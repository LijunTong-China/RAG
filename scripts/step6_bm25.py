# -*- coding: utf-8 -*-
"""步骤6：BM25 检索索引构建（与步骤5 向量入库同规格）。

- 输入：data/04_chunks/{report_id}.json（与向量同源）
- 分词：jieba（配置 bm25.tokenizer）；stopwords 文件可选（配置了就必须存在，不兜底）
- 索引：rank_bm25.BM25Okapi，k1/b 可配置；单份一个 pickle 文件
- 单份重灌：删除旧 pkl 再建（与向量按 report_name 删旧同义）
- 对账：索引词条数 == 步骤4切片数，不等 = 失败
- 报告：data/logs/step6_bm25_report.json（追加式）
"""
import argparse
import json
import os
import pickle

from common import (abspath, append_log, cfg_get, ensure_dir, fail,
                    load_config, require_file)


def _tokenize(cfg, text):
    engine = cfg_get(cfg, "bm25.tokenizer")
    if engine != "jieba":
        fail("不支持的 bm25.tokenizer: %s" % engine)
    import jieba
    tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]
    sw_path = cfg["bm25"].get("stopwords", "")  # 可选项：不配置 = 不用停用词
    if sw_path:
        sw_file = abspath(sw_path)
        if not os.path.isfile(sw_file):
            fail("bm25.stopwords 文件不存在: %s" % sw_file)
        global _SW
        if _SW is None:
            with open(sw_file, "r", encoding="utf-8") as f:
                _SW = {w.strip() for w in f if w.strip()}
        tokens = [t for t in tokens if t not in _SW]
    return tokens


_SW = None


def build_bm25(cfg, rid):
    """单份构建 BM25 索引（graph 与 CLI 共用）。"""
    src = require_file(os.path.join(abspath(cfg_get(cfg, "paths.chunks")), rid + ".json"),
                       "切片 JSON（先运行 step4）")
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]

    from rank_bm25 import BM25Okapi
    corpus = [_tokenize(cfg, c["text"]) for c in chunks]
    empty = [i for i, t in enumerate(corpus) if not t]
    if empty:
        fail("切片 %s 分词后为空（%d 条），无法建索引: %s"
             % (rid, len(empty), [chunks[i]["chunk_id"] for i in empty[:5]]))
    bm25 = BM25Okapi(corpus,
                     k1=float(cfg_get(cfg, "bm25.k1")),
                     b=float(cfg_get(cfg, "bm25.b")))

    index_dir = ensure_dir(abspath(cfg_get(cfg, "bm25.index_dir")))
    dst = os.path.join(index_dir, rid + ".pkl")
    if os.path.exists(dst):  # 单份重灌：先删旧索引
        os.remove(dst)
    with open(dst, "wb") as f:
        pickle.dump({
            "report_id": rid,
            "bm25": bm25,
            "chunk_ids": [c["chunk_id"] for c in chunks],
            "pages": [c["page"] for c in chunks],
            "chunk_index": [c["chunk_index"] for c in chunks],
            "texts": [c["text"] for c in chunks],
            "params": {"k1": float(cfg_get(cfg, "bm25.k1")), "b": float(cfg_get(cfg, "bm25.b"))},
        }, f)

    # 逐份对账
    with open(dst, "rb") as f:
        loaded = pickle.load(f)
    expected = len(chunks)
    ok = len(loaded["chunk_ids"]) == expected and loaded["bm25"].corpus_size == expected
    record = {"report_id": rid, "chunks": expected, "indexed": len(loaded["chunk_ids"]),
              "reconciled": ok, "index_file": dst}
    if not ok:
        append_log(cfg, "step6_bm25_report", record)
        fail("对账失败：%s BM25 词条数 %d != 切片数 %d" % (rid, len(loaded["chunk_ids"]), expected))
    append_log(cfg, "step6_bm25_report", record)
    print("BM25 索引完成 %s：%d 条，对账一致 -> %s" % (rid, expected, dst))
    return record


def load_index(cfg, rid):
    """查询侧加载单份索引（后续召回阶段用）。"""
    path = os.path.join(abspath(cfg_get(cfg, "bm25.index_dir")), rid + ".pkl")
    require_file(path, "BM25 索引（先运行 step6）")
    with open(path, "rb") as f:
        return pickle.load(f)


def search(cfg, rid, query, top_k):
    """单份 BM25 检索（与向量同一把分词尺子）。"""
    idx = load_index(cfg, rid)
    scores = idx["bm25"].get_scores(_tokenize(cfg, query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [{"chunk_id": idx["chunk_ids"][i], "page": idx["pages"][i],
             "score": scores[i], "text": idx["texts"][i]} for i in order]


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_id", required=True, help="单份重灌粒度 = 单份财报")
    args = ap.parse_args()
    build_bm25(cfg, args.report_id)


if __name__ == "__main__":
    main()
