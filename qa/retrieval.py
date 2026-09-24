# -*- coding: utf-8 -*-
"""S4 混合检索 + S5 页级归并与重排序（非 LLM）。

- 向量路：Milvus（复用 step5 的向量化原语）
- BM25 路：本地 rank_bm25 索引（复用 step6 的检索原语），filters 命中的报告集合内检索
- 融合：RRF；归并：chunk ->（文件, 页码）去重回查整页（data/03_flattened）
- 重排序：Agicto qwen3-rerank（/rerank 接口），打分文本 = 整页原文
"""
import os
import time

from qa_common import abspath, cfg_get, get_qa_cfg


# ---------- 报告集合（filters -> 报告 id 列表） ----------

def select_reports(scope, filters):
    """按 filters（company/stock_code/report_type/period，null 维度不过滤）选报告集合。"""
    sel = []
    for rep in scope["reports"]:
        if filters.get("company") and rep["company"] != filters["company"]:
            continue
        if filters.get("stock_code") and (rep["stock_code"] or "") != filters["stock_code"]:
            continue
        if filters.get("report_type") and rep["report_type"] != filters["report_type"]:
            continue
        if filters.get("period") and rep["period"] != filters["period"]:
            continue
        sel.append(rep)
    return sel


# ---------- 向量路 ----------

def _escape(v):
    return str(v).replace("\\", "\\\\").replace('"', '\\"')


def milvus_search(cfg, query, filters, top_n):
    """单条查询的向量召回；null 维度不拼入 expr。返回 (hits, timings)。"""
    from step5_milvus import embed_batches, get_client
    from pymilvus import Collection, connections

    mil = cfg["milvus"]
    conns = connections
    # 同进程内复用连接（has_connection 幂等）
    if not conns.has_connection("default"):
        conns.connect(host=mil["host"], port=str(mil["port"]))

    exprs = []
    for key in ("company", "stock_code", "report_type", "period"):
        if filters.get(key):
            exprs.append('%s == "%s"' % (key, _escape(filters[key])))
    expr = " and ".join(exprs) if exprs else None

    client = get_client(cfg)
    t0 = time.time()
    vec = embed_batches(cfg, client, [query])[0]
    t_embed = time.time() - t0
    col = Collection(mil["collection"])
    t0 = time.time()
    col.load()
    t_load = time.time() - t0
    t0 = time.time()
    res = col.search([vec], "vector", param={"metric_type": mil["metric_type"],
                                             "params": {"ef": 128}},
                     limit=top_n, expr=expr,
                     output_fields=["chunk_id", "text", "report_name", "page", "chunk_index"])
    t_search = time.time() - t0
    hits = []
    for rank, r in enumerate(res[0]):
        e = r.entity
        hits.append({
            "chunk_id": e.get("chunk_id"), "text": e.get("text"),
            "report_name": e.get("report_name"), "page": int(e.get("page")),
            "chunk_index": int(e.get("chunk_index")),
            "vec_rank": rank, "vec_score": float(r.score),
        })
    return hits, {"embed_s": round(t_embed, 3), "load_s": round(t_load, 3),
                  "search_s": round(t_search, 3)}


# ---------- BM25 路 ----------

def bm25_search(cfg, reports, query, top_n):
    """filters 命中的报告集合内 BM25 检索，跨报告按分数归并取 top_n。"""
    from step6_bm25 import search as bm25_report_search
    all_hits = []
    for rep in reports:
        for h in bm25_report_search(cfg, rep["report_id"], query, top_n):
            if h["score"] <= 0:  # 零分命中 = 无词项匹配，不该占 RRF 融合名额
                continue
            h["report_name"] = rep["report_id"]  # 与 Milvus report_name 同一逻辑 id
            all_hits.append(h)
    all_hits.sort(key=lambda h: -h["score"])
    for rank, h in enumerate(all_hits[:top_n]):
        h["bm25_rank"] = rank
    return all_hits[:top_n]


# ---------- RRF 融合 ----------

def rrf_fuse(vec_hits, bm25_hits, rrf_k):
    """按 chunk_id 融合两路排名（缺一路时按另一路参与）。"""
    pool = {}

    def put(hit, rank, tag):
        cid = hit["chunk_id"]
        e = pool.setdefault(cid, {"hit": hit, "scores": {}})
        e["scores"][tag] = rank + 1
        e["hit"] = {**e["hit"], **{k: v for k, v in hit.items() if v is not None}}

    for rank, h in enumerate(vec_hits):
        put(h, rank, "vec")
    for rank, h in enumerate(bm25_hits):
        put(h, rank, "bm25")

    fused = []
    for cid, e in pool.items():
        score = 0.0
        for tag, r in e["scores"].items():
            score += 1.0 / (rrf_k + r)
        fused.append({**e["hit"], "rrf_score": score, "rrf_tags": sorted(e["scores"])})
    fused.sort(key=lambda h: -h["rrf_score"])
    return fused


# ---------- 页级归并 ----------

class PageStore:
    """data/03_flattened/{rid}.json 的整页文本缓存。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self._cache = {}

    def get(self, report_id):
        if report_id not in self._cache:
            path = os.path.join(abspath(cfg_get(self.cfg, "paths.flattened")),
                                report_id + ".json")
            if not os.path.isfile(path):
                raise FileNotFoundError("整页 JSON 不存在（先运行 step3）: %s" % path)
            import json
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._cache[report_id] = {int(p["page"]): p["text"] for p in data["pages"]}
        return self._cache[report_id]


def merge_to_pages(cfg, fused_candidates, page_store):
    """chunk 候选按（文件, 页码）去重 -> 回查整页 -> 按 RRF 预排序截取 rerank_candidates。

    同页命中多条子查询时 source_sub_query_ids 取并集（分桶依据，不可丢弃）。
    """
    top = int(get_qa_cfg(cfg)["retrieval"]["rerank_candidates"])
    pages = {}  # key -> {"cand": 最高分候选, "sq_ids": 并集}
    for cand in fused_candidates:
        key = (cand["report_name"], cand["page"])
        entry = pages.setdefault(key, {"cand": cand, "sq_ids": set()})
        if cand["rrf_score"] > entry["cand"]["rrf_score"]:
            entry["cand"] = cand
        entry["sq_ids"].update(str(s) for s in (cand.get("source_sub_query_ids") or []))
    out = []
    for (rid, page), entry in pages.items():
        cand = entry["cand"]
        texts = page_store.get(rid)
        if page not in texts:
            raise KeyError("整页原文缺失: %s P%s" % (rid, page))
        out.append({
            "page_key": "%s::P%d" % (rid, page),
            "report_name": rid, "page": page,
            "text": texts[page],
            "rrf_score": cand["rrf_score"], "rrf_tags": cand["rrf_tags"],
            "source_sub_query_ids": sorted(entry["sq_ids"]),
        })
    out.sort(key=lambda p: -p["rrf_score"])
    return out[:top]


# ---------- 重排序 ----------

def rerank_pages(cfg, pages, query):
    """Agicto qwen3-rerank：query vs 整页原文；返回每页 relevance_score（就地写入）。

    失败不静默：重试用尽后抛异常（宁可不答，不给未精排结果）。
    """
    import os as _os
    import requests
    rcfg = get_qa_cfg(cfg)["retrieval"]
    base = rcfg["rerank_base_url"].rstrip("/")
    key_env = rcfg["rerank_api_key_env"]
    key = _os.environ.get(key_env)
    if not key:
        raise RuntimeError("环境变量 %s 未设置（rerank API key 不兜底）" % key_env)

    documents = [p["text"] for p in pages]
    max_retries = int(cfg_get(cfg, "llm.max_retries"))
    wait_s = int(cfg_get(cfg, "llm.retry_wait_seconds"))
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(base + "/rerank",
                                 headers={"Authorization": "Bearer " + key},
                                 json={"model": rcfg["rerank_model"],
                                       "query": query, "documents": documents},
                                 timeout=120)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results") or data.get("data")
            if results is None:
                raise ValueError("rerank 响应缺少 results 字段: %s" % str(data)[:200])
            for item in results:
                score = item.get("relevance_score", item.get("score"))
                pages[int(item["index"])]["rerank_score"] = float(score)
            return pages
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                import time
                time.sleep(wait_s)
    raise RuntimeError("rerank 调用失败（已重试 %d 次）: %s" % (max_retries, last_err))


# ---------- 单轮检索完整流程 ----------

def normalize_filters(scope, filters):
    """filters 与语料库对齐：全库无 stock_code 元数据时该维度整体失效。

    否则 P3 给出 stock_code（语料库实际为空）会导致双路检索 0 命中、白烧全部循环轮。
    """
    f = dict(filters or {})
    if f.get("stock_code") and not any(r.get("stock_code") for r in scope["reports"]):
        f.pop("stock_code")
    return f


def retrieve_loop_round(cfg, scope, sub_queries, page_store, log_ctx):
    """一轮检索：每条子查询独立 双路召回 + RRF -> 页级归并 -> rerank -> top_pages_per_loop。

    子查询间并行（线程池，max_workers = retrieval.max_workers）：
    - 召回阶段每条子查询的 双路检索 各自并发执行；
    - rerank 阶段每条子查询的精排调用并发执行。
    sub_queries 元素形如 {"query": str, "filters": {...}, "id": int}（首轮为 P3 的
    sub_queries 映射，补充轮为 P45 的 additional_queries 映射）。
    返回本轮 top 页列表（含 rerank_score、source_sub_query_ids）。
    """
    from concurrent.futures import ThreadPoolExecutor
    rcfg = get_qa_cfg(cfg)["retrieval"]
    workers = max(1, int(rcfg.get("max_workers", 4)))

    def _search_one(sq):
        filters = normalize_filters(scope, sq.get("filters") or {})
        reports = select_reports(scope, filters)
        # keywords 消费点：并入 BM25 查询扩展（向量路仍用原句，避免稀释语义）
        bm_query = sq["query"]
        kws = [k for k in (sq.get("keywords") or []) if k]
        if kws:
            bm_query = bm_query + " " + " ".join(kws)
        vec, vec_timing = milvus_search(cfg, sq["query"], filters, int(rcfg["vector_top_n"]))
        t0 = time.time()
        bm = bm25_search(cfg, reports, bm_query, int(rcfg["bm25_top_n"]))
        t_bm = time.time() - t0
        t0 = time.time()
        fused = rrf_fuse(vec, bm, int(rcfg["rrf_k"]))
        t_rrf = time.time() - t0
        for f in fused:
            f.setdefault("source_sub_query_ids", []).append(sq.get("id"))
        timing = {"milvus": vec_timing, "bm25_s": round(t_bm, 3), "rrf_s": round(t_rrf, 3)}
        return sq, filters, reports, vec, bm, fused, timing

    fused_all = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for sq, filters, reports, vec, bm, fused, timing in pool.map(_search_one, sub_queries):
            fused_all.extend(fused)
            log_ctx["sub_query_logs"].append({
                "id": sq.get("id"), "query": sq["query"], "filters": filters,
                "reports_selected": [r["report_id"] for r in reports],
                "vec_hits": len(vec), "bm25_hits": len(bm), "fused": len(fused),
                "timing": timing,
            })

    t0 = time.time()
    pages = merge_to_pages(cfg, fused_all, page_store)
    merge_s = time.time() - t0
    if not pages:
        return []
    # rerank 打分（并行）：query 用该页命中子查询的 rewritten_query；
    # 一页命中多条子查询时取最高分
    per_page_scores = {}  # page_key -> {sq_id: score}
    rerank_timings = []

    def _rerank_one(sq):
        t0 = time.time()
        scored = rerank_pages(cfg, [dict(p) for p in pages], sq["query"])
        return sq, scored, time.time() - t0

    t_rerank0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for sq, scored, t_sq in pool.map(_rerank_one, sub_queries):
            rerank_timings.append({"id": sq.get("id"), "seconds": round(t_sq, 3),
                                   "pages": len(pages)})
            for p in scored:
                per_page_scores.setdefault(p["page_key"], {})[sq.get("id")] = p.get("rerank_score", 0.0)
    rerank_s = time.time() - t_rerank0
    for p in pages:
        scores = per_page_scores.get(p["page_key"], {})
        p["rerank_scores_by_sq"] = scores
        p["rerank_score"] = max(scores.values()) if scores else 0.0
    pages.sort(key=lambda p: -p["rerank_score"])
    out = pages[:int(rcfg["top_pages_per_loop"])]
    log_ctx["round_timing"] = {
        "merge_page_s": round(merge_s, 3),
        "rerank_total_s": round(rerank_s, 3), "rerank_per_sq": rerank_timings,
    }
    return out
