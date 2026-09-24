# -*- coding: utf-8 -*-
"""S1~S7 编排：一次提问的完整生命周期。

架构 = LangChain map_reduce 模式：
- 拆解（P3）：多意图/对比/反问 → N 个独立子任务（每条自包含查询 + filters）；
- Map（S4/S5/S6 + P5）：每个子任务独立执行自己的检索循环（不足时在任务内部补充检索，
  最多 max_loops 轮）并独立作答，证据池任务内隔离，互不挤占；
- Reduce（P6）：大模型汇总各子任务答案，回答原始问题（对比综合、计算值规则在此生效）。

例："A 公司比 B 公司上半年多盈利多少" → 子任务 A（A 上半年盈利）+ 子任务 B
（B 上半年盈利）各自检索作答 → P6 汇总做差给出结论。
"""
import json
import re
import time

from common import count_tokens
from qa_common import corpus_scope, get_qa_cfg, log_turn
import stages
import steplog
from memory import SessionStore
from retrieval import PageStore, retrieve_loop_round


class QAService:
    def __init__(self, cfg):
        self.cfg = cfg
        self.store = SessionStore(cfg)
        self.scope = corpus_scope(cfg)
        self.page_store = PageStore(cfg)
        self._report_meta = {r["report_id"]: r for r in self.scope["reports"]}

    def startup_check(self):
        """启动自检：Milvus 连接 + BM25 索引存在性，不通过拒绝启动。"""
        import os
        from pymilvus import connections
        mil = self.cfg["milvus"]
        connections.connect(host=mil["host"], port=str(mil["port"]))
        from pymilvus import utility
        if not utility.has_collection(mil["collection"]):
            raise RuntimeError("Milvus 集合不存在: %s" % mil["collection"])
        from qa_common import abspath
        for rep in self.scope["reports"]:
            pkl = os.path.join(abspath(self.cfg["bm25"]["index_dir"]),
                               rep["report_id"] + ".pkl")
            if not os.path.isfile(pkl):
                raise RuntimeError("BM25 索引缺失（先运行 step6）: %s" % pkl)

    # ---------- 证据池 ----------

    def _enrich(self, page):
        meta = self._report_meta.get(page["report_name"], {})
        page["company"] = meta.get("company", "")
        page["stock_code"] = meta.get("stock_code") or ""
        page["report_type"] = meta.get("report_type", "")
        page["period"] = meta.get("period", "")
        return page

    def _evidence_block(self, evidence_pool):
        """注入 LLM 的证据文本：单页按 max_page_tokens 截断；受 evidence_token_budget。

        桶 = 任务内检索通道（首轮子查询 / 各条补充查询），每桶保底 bucket_min_pages 页，
        防止任务内补充查询与首轮互相挤占。
        """
        rcfg = get_qa_cfg(self.cfg)["retrieval"]
        max_page = int(rcfg["max_page_tokens"])
        budget = int(rcfg["evidence_token_budget"])
        min_quota = int(rcfg["bucket_min_pages"])

        buckets = {}
        for p in evidence_pool:
            keys = p.get("buckets") or ["common"]
            for b in keys:
                buckets.setdefault(b, []).append(p)
        for lst in buckets.values():
            lst.sort(key=lambda p: -p["rerank_score"])

        # 选取顺序：先每桶保底，再轮转补齐
        selected, sel_keys = [], set()

        def take(p):
            if p["page_key"] not in sel_keys:
                selected.append(p)
                sel_keys.add(p["page_key"])

        for lst in buckets.values():
            for p in lst[:min_quota]:
                take(p)
        rest = {b: lst[min_quota:] for b, lst in buckets.items()}
        while True:
            added = False
            for b in sorted(rest):
                while rest[b]:
                    p = rest[b].pop(0)
                    if p["page_key"] not in sel_keys:
                        take(p)
                        added = True
                        break
            if not added:
                break
        pool = sorted(selected, key=lambda p: -p["rerank_score"])
        lines, used, dropped = [], 0, 0
        for i, p in enumerate(pool, 1):
            text = p["text"]
            if count_tokens(self.cfg, text) > max_page:
                import tiktoken
                enc = tiktoken.get_encoding("o200k_base")
                text = enc.decode(enc.encode(text)[:max_page])
            header = "[证据%d] %s（%s %s %s）P%d 来源子查询: %s" % (
                i, p["report_name"], p["company"], p["period"], p["report_type"],
                p["page"], p.get("source_sub_query_ids", []))
            t = count_tokens(self.cfg, header) + count_tokens(self.cfg, text)
            if used + t > budget:
                dropped += 1
                continue
            lines.append(header + "\n" + text)
            used += t
        if dropped:
            lines.append("（另有 %d 页因证据池 token 预算被舍弃，优先保留了 rerank 高分页）" % dropped)
        return "\n\n".join(lines) if lines else "（无证据）"

    def _merge_pages_into_pool(self, evidence_pool, pages):
        """页并入任务内证据池：去重、高分替换（继承 buckets）、桶归属累积。"""
        for p in pages:
            self._enrich(p)
            cur = evidence_pool.get(p["page_key"])
            if cur is None:
                evidence_pool[p["page_key"]] = p
                cur = p
            elif p["rerank_score"] > cur.get("rerank_score", 0.0):
                p["buckets"] = cur.get("buckets") or []  # 替换时继承旧页桶归属
                evidence_pool[p["page_key"]] = p
                cur = p
            sq_ids = p.get("source_sub_query_ids") or []
            merged = set(cur.get("buckets") or []) | {str(s) for s in sq_ids}
            cur["buckets"] = sorted(merged) or ["common"]

    # ---------- 来源标注程序级校验（无提示词参与） ----------

    _CITE_RE = re.compile(r"\[来源[:：]\s*([^\[\]]+?)\s*[Pp](\d+)\s*\]")

    def _validate_citations(self, answer, pool):
        """解析答案中的 [来源: 报告名 P页码]，与证据池 page_key 求交集：
        越界标注整段剔除；剔除后无任何有效标注时降级为按 rerank 分附证据池来源列表。
        返回 (清洗后的回答, 有效来源列表, 被剔除的标注)。"""
        pool_keys = {(p["report_name"], int(p["page"])) for p in pool}
        valid_cites, invalid_cites = set(), []
        pool_by_key = {(p["report_name"], int(p["page"])): p for p in pool}

        def _check(m):
            name, page = m.group(1).strip(), int(m.group(2))
            key = (name, page)
            if key in pool_keys:
                valid_cites.add(key)
                return m.group(0)
            # 报告名模糊匹配（模型可能缩写/扩写报告名）
            for rn, pg in pool_keys:
                if pg == page and (name in rn or rn in name):
                    valid_cites.add((rn, pg))
                    return m.group(0)
            invalid_cites.append(m.group(0))
            return ""

        cleaned = self._CITE_RE.sub(_check, answer)

        if valid_cites:
            sources = [{"report_name": rn, "company": pool_by_key[(rn, pg)]["company"],
                        "report_type": pool_by_key[(rn, pg)]["report_type"],
                        "period": pool_by_key[(rn, pg)]["period"], "page": pg}
                       for rn, pg in valid_cites]
        else:
            # 降级：答案没带（或全被剔除）标注时，附证据池高分页供用户自行核对
            sources = [{"report_name": p["report_name"], "company": p["company"],
                        "report_type": p["report_type"], "period": p["period"],
                        "page": p["page"]} for p in
                       sorted(pool, key=lambda x: -x["rerank_score"])[:5]]
        return cleaned, sources, invalid_cites

    # ---------- Map：单个子任务 = 独立检索循环 + 独立作答 ----------

    def _run_task(self, task, origin_question, tlog):
        """一个子任务的完整 map 流程（P45 合并节点：判断+作答一次调用）。

        - 检索循环是任务内部的：证据不足时 P45 给补充查询，只服务于本任务；
        - 证据池任务内隔离，多任务互不挤占 token 预算；多任务之间并行执行；
        - 充分性结论随最终回答一并产出（reply 中保留 insufficient/sufficient 语义）。
        返回 {task_id, query, insufficient, answer, missing_info, sources,
              invalid_citations, pool, sufficient}。
        """
        rcfg = get_qa_cfg(self.cfg)["retrieval"]
        max_loops = int(rcfg["max_loops"])
        evidence_pool = {}
        current = [task]
        loops = []
        result = None
        for loop in range(1, max_loops + 1):
            t1 = steplog.step("任务%s · 检索循环 第%d/%d轮（双路召回+RRF+页合并+重排，子查询并行）"
                              % (task["id"], loop, max_loops),
                              inputs={"查询列表": current,
                                      "证据池现有页数": len(evidence_pool)})
            pages = retrieve_loop_round(self.cfg, self.scope, current,
                                        self.page_store, tlog)
            self._merge_pages_into_pool(evidence_pool, pages)
            loops.append({"loop": loop, "new_pages": [p["page_key"] for p in pages],
                          "pool_size": len(evidence_pool)})
            steplog.done(t1, {"本轮新增页": [p["page_key"] for p in pages],
                              "本轮top分": max((p["rerank_score"] for p in pages), default=None),
                              "证据池累计页数": len(evidence_pool)})
            t2 = steplog.step("任务%s · 充分性判断+作答（P45 合并节点，第%d轮）"
                              % (task["id"], loop),
                              inputs={"任务问题": task["query"]})
            result = stages.answer_with_check(
                self.cfg, task["query"],
                [{**task, "original_question": origin_question}],
                self._evidence_block(list(evidence_pool.values())),
                loop, max_loops, self.scope)
            loops[-1]["verdict"] = {
                "sufficient": result["sufficient"],
                "missing_points": result["missing_points"],
                "reason": result.get("reason", ""),
            }
            steplog.done(t2, {"sufficient": result["sufficient"],
                              "reason": result.get("reason", ""),
                              "missing_points": result["missing_points"]})
            if result["sufficient"]:
                steplog.note(">>> 判定充分，采用本轮作答")
                break
            if loop == max_loops:
                steplog.note(">>> 已达最大轮数 %d，带现有证据作答（sufficient=false）" % max_loops)
                break
            if not result["additional_queries"]:
                steplog.note(">>> 判定不充分但无补充查询建议，带现有证据作答")
                break
            steplog.note(">>> 判定不充分，将在下一轮补充检索 %d 条"
                         % len(result["additional_queries"]))
            current = [{"id": "sup%d_%d" % (loop, i + 1), "query": q["query"],
                        "filters": q.get("filters") or {}}
                       for i, q in enumerate(result["additional_queries"])]
        tlog["loops"] = loops
        tlog["result"] = result

        pool = list(evidence_pool.values())
        base = {"task_id": task["id"], "query": task["query"], "pool": pool,
                "sufficient": bool(result["sufficient"])}
        if result.get("insufficient") or not (result.get("answer") or "").strip():
            steplog.note(">>> 任务%s 作答结论：insufficient（missing_info=%s）"
                         % (task["id"], result.get("missing_info")))
            return {**base, "insufficient": True, "answer": None,
                    "missing_info": result.get("missing_info") or [],
                    "sources": [], "invalid_citations": []}
        reply, sources, invalid = self._validate_citations(result["answer"], pool)
        steplog.note(">>> 任务%s 作答完成：sufficient=%s, 来源 %d 页, 虚构标注剔除 %d 条"
                     % (task["id"], result["sufficient"], len(sources), len(invalid)))
        return {**base, "insufficient": False, "answer": reply,
                "missing_info": result.get("missing_info") or [],
                "sources": sources, "invalid_citations": invalid}

    # ---------- 主入口 ----------

    def answer(self, session, message):
        """返回 {reply, sources, insufficient, elapsed_ms}。调用方负责持会话锁。"""
        t0 = time.time()
        qa_cfg = get_qa_cfg(self.cfg)
        turn_no = session.next_turn_no()
        log = {"session_id": session.session_id, "turn_no": turn_no,
               "question": message, "tasks": []}
        steplog.reset()
        print("\n" + "█" * 66)
        print("█ ▶▶▶ 新提问  会话=%s 轮次=%d" % (session.session_id[:8], turn_no))
        print("█" * 66)

        memory_context = session.context_for_llm()

        # S2 意图识别
        t1 = steplog.step("S2 意图识别（P1）",
                          inputs={"问题": message,
                                  "对话记忆": memory_context or "（空）"})
        intent = stages.detect_intent(self.cfg, message, memory_context)
        log["intent"] = intent
        steplog.done(t1, intent)
        if not intent["is_financial_qa"] and intent["confidence"] >= 0.5:
            steplog.note(">>> 非财报问题，命中拒绝话术")
            reply = qa_cfg["replies"]["reject_template"]
            session.add_message(turn_no, "user", message)
            session.add_message(turn_no, "assistant", reply)
            session.compress(lambda s, texts: stages.compress_memory(self.cfg, s, texts))
            log["rejected"] = True
            log["step_timings"] = steplog.summary()
            log_turn(self.cfg, log)
            steplog.dump(self.cfg, session.session_id, turn_no,
                         time.time() - t0, message)
            return {"reply": reply, "sources": [], "insufficient": False,
                    "elapsed_ms": int((time.time() - t0) * 1000)}

        # S3 任务拆解
        t2 = steplog.step("S3 任务拆解/查询改写（P3，map_reduce 拆解层）",
                          inputs={"问题": message,
                                  "对话记忆": memory_context or "（空）"})
        plan = stages.rewrite_query(self.cfg, message, memory_context, self.scope)
        tasks = [{"id": str(sq["id"]), "query": sq["rewritten_query"],
                  "filters": sq.get("filters") or {},
                  "keywords": sq.get("keywords") or []} for sq in plan["sub_queries"]]
        log["query_plan"] = plan
        steplog.done(t2, {"query_type": plan.get("query_type"),
                          "子任务数": len(tasks),
                          "任务列表": [{"id": t["id"], "query": t["query"],
                                        "filters": t["filters"]} for t in tasks]})

        # Map：子任务并行（各自独立的检索循环 + P45 判断作答，证据池任务内隔离）
        task_results = [None] * len(tasks)

        def _map_one(idx, task):
            steplog.note("■■ MAP 开始：子任务%s（共 %d 个子任务，并行执行）"
                         % (task["id"], len(tasks)))
            tlog = {"task_id": task["id"], "task_query": task["query"],
                    "sub_query_logs": []}
            res = self._run_task(task, message, tlog)
            tlog["answer"] = res["answer"]
            tlog["insufficient"] = res["insufficient"]
            tlog["sufficient"] = res["sufficient"]
            tlog["invalid_citations"] = res["invalid_citations"]
            tlog["pool"] = [p["page_key"] for p in res["pool"]]
            log["tasks"].append(tlog)
            steplog.note("■■ MAP 结束：子任务%s（%s，依据%s）"
                         % (task["id"], "信息不足" if res["insufficient"] else "已作答",
                            "充分" if res["sufficient"] else "不充分"))
            return idx, res

        from concurrent.futures import ThreadPoolExecutor
        map_workers = min(len(tasks), max(1, int(qa_cfg["retrieval"].get("map_workers", 4))))
        with ThreadPoolExecutor(max_workers=map_workers) as pool:
            for idx, res in pool.map(lambda p: _map_one(*p), enumerate(tasks)):
                task_results[idx] = res
        log["tasks"].sort(key=lambda t: str(t["task_id"]))  # 并行完成序 -> 按任务序

        # Reduce：汇总各子任务答案回答原始问题
        print("\n" + "◆" * 66)
        print("◆ ◆◆ REDUCE 开始：汇总 %d 个子任务答案" % len(task_results))
        print("◆" * 66)
        t4 = steplog.step("REDUCE 答案汇总（P6）",
                          inputs={"原始问题": message,
                                  "子任务答案": [
                                      {"task_id": r["task_id"],
                                       "answer": (r["answer"] or "未在现有资料中找到")[:300],
                                       "sources": ["%s P%d" % (s["report_name"], s["page"])
                                                   for s in r["sources"]]}
                                      for r in task_results]})
        if all(r["insufficient"] for r in task_results):
            result = {"insufficient": True, "missing_info":
                      sorted({m for r in task_results for m in r["missing_info"]})}
        else:
            payload = [{"task_id": r["task_id"], "task_query": r["query"],
                        "answer": r["answer"] or "未在现有资料中找到",
                        "sufficient": r["sufficient"],
                        "sources": ["%s P%d" % (s["report_name"], s["page"])
                                    for s in r["sources"]],
                        "missing_info": r["missing_info"]}
                       for r in task_results]
            result = stages.aggregate_answers(self.cfg, message, payload)
        steplog.done(t4, result)
        log["aggregate"] = result

        if result.get("insufficient"):
            reply = qa_cfg["replies"]["insufficient_template"]
            if result.get("missing_info"):
                reply += "\n（未找到的信息点：" + "；".join(result["missing_info"]) + "）"
            sources = []
            insufficient = True
        else:
            # 校验对象 = 全部子任务证据池的并集（P6 只能引用子任务给出的来源页）
            union_pool, seen = [], set()
            for r in task_results:
                for p in r["pool"]:
                    if p["page_key"] not in seen:
                        seen.add(p["page_key"])
                        union_pool.append(p)
            reply, sources, invalid = self._validate_citations(result["answer"], union_pool)
            if invalid:
                log["invalid_citations"] = invalid
            insufficient = False
            t5 = steplog.step("来源标注程序级校验（Reduce 层，对子任务证据池并集）")
            steplog.done(t5, {"有效来源": ["%s P%d" % (s["report_name"], s["page"])
                                            for s in sources],
                              "被剔除的虚构标注": invalid or "无"})

        session.add_message(turn_no, "user", message)
        session.add_message(turn_no, "assistant", reply)
        session.compress(lambda s, texts: stages.compress_memory(self.cfg, s, texts))
        log["reply"] = reply
        log["sources"] = sources
        log["elapsed_ms"] = int((time.time() - t0) * 1000)
        log["step_timings"] = steplog.summary()
        log_turn(self.cfg, log)
        steplog.dump(self.cfg, session.session_id, turn_no,
                     time.time() - t0, message)
        print("\n" + "█" * 66)
        print("█ ✔✔ 提问完成  总耗时 %.1fs  来源 %d 页  %s"
              % (log["elapsed_ms"] / 1000.0, len(sources),
                 "信息不足" if insufficient else "已作答"))
        print("█" * 66)
        return {"reply": reply, "sources": sources,
                "insufficient": insufficient, "elapsed_ms": log["elapsed_ms"]}
