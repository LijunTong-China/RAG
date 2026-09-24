# -*- coding: utf-8 -*-
"""步骤5：切片 JSON -> Agicto text-embedding-v4 向量化 -> 写入 Milvus（Docker Standalone）。

- API key 只从配置指定的环境变量读取，缺失即报错（不兜底）
- 单份重灌 = 按 report_name 删除旧数据再插入
- 入库后逐份对账：Milvus 查询数 == 步骤4切片数，不等 = 失败
"""
import argparse
import json
import os
import time

from common import (abspath, append_log, cfg_get, ensure_dir, fail,
                    load_config, logical_report_id, read_manifest, require_file)


def get_client(cfg):
    from openai import OpenAI
    env_key = cfg_get(cfg, "embedding.api_key_env")
    key = os.environ.get(env_key)
    if not key:
        fail("环境变量 %s 未设置（embedding API key 不兜底）" % env_key)
    return OpenAI(base_url=cfg_get(cfg, "embedding.base_url"), api_key=key)


def embed_batches(cfg, client, texts):
    batch_size = int(cfg_get(cfg, "embedding.batch_size"))
    max_retries = int(cfg_get(cfg, "embedding.max_retries"))
    wait_s = int(cfg_get(cfg, "embedding.retry_wait_seconds"))
    payload_base = {
        "model": cfg_get(cfg, "embedding.model"),
        "dimensions": int(cfg_get(cfg, "embedding.dimensions")),
    }
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        resp = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.embeddings.create(input=batch, **payload_base)
                break
            except Exception as e:
                if attempt == max_retries:
                    raise RuntimeError("向量化失败（已重试 %d 次，批次起点 %d）: %s" % (max_retries, start, e))
                print("[重试] 批次起点 %d 第 %d 次失败: %s，%ds 后重试" % (start, attempt, e, wait_s))
                time.sleep(wait_s)
        got = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        if len(got) != len(batch):
            raise RuntimeError("向量化返回条数 %d != 请求条数 %d（起点 %d）" % (len(got), len(batch), start))
        dim = int(cfg_get(cfg, "embedding.dimensions"))
        if any(len(v) != dim for v in got):
            raise RuntimeError("向量维度不为 %d（起点 %d）" % (dim, start))
        vectors.extend(got)
    return vectors


def ensure_collection(cfg, mil):
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
    connections.connect(host=mil["host"], port=str(mil["port"]))
    name = mil["collection"]
    if utility.has_collection(name):
        col = Collection(name)
        # 旧 schema 兼容检查：缺 stock_code/report_type/period 字段 = 必须删集合重灌
        existing = {f.name for f in col.schema.fields}
        missing = {"stock_code", "report_type", "period"} - existing
        if missing:
            fail("Milvus 集合 %s 缺少字段 %s（旧 schema）。请删除该集合并重跑 step5 重灌全部报告："
                 "pymilvus 中 utility.drop_collection(\"%s\") 后重新执行本脚本" % (name, sorted(missing), name))
        return col
    lim = mil["varchar_limits"]  # 整块取用，缺项由 cfg_get 校验不了 dict，故逐项检查
    for k in ("chunk_id", "text", "report_name", "company", "stock_code",
              "report_type", "period", "source_path"):
        if k not in lim:
            fail("配置缺失: milvus.varchar_limits.%s" % k)
    fields = [
        FieldSchema("chunk_id", DataType.VARCHAR, max_length=lim["chunk_id"], is_primary=True),
        FieldSchema("vector", DataType.FLOAT_VECTOR, dim=int(cfg_get(cfg, "embedding.dimensions"))),
        FieldSchema("text", DataType.VARCHAR, max_length=lim["text"]),
        FieldSchema("report_name", DataType.VARCHAR, max_length=lim["report_name"]),
        FieldSchema("company", DataType.VARCHAR, max_length=lim["company"]),
        FieldSchema("stock_code", DataType.VARCHAR, max_length=lim["stock_code"]),
        FieldSchema("report_type", DataType.VARCHAR, max_length=lim["report_type"]),
        FieldSchema("period", DataType.VARCHAR, max_length=lim["period"]),
        FieldSchema("source_path", DataType.VARCHAR, max_length=lim["source_path"]),
        FieldSchema("page", DataType.INT16),
        FieldSchema("chunk_index", DataType.INT16),
    ]
    col = Collection(name, CollectionSchema(fields, description="financial_reports"))
    ip = mil["index_params"]
    col.create_index("vector", {"index_type": mil["index_type"], "metric_type": mil["metric_type"],
                                "params": {"M": ip["M"], "efConstruction": ip["efConstruction"]}})
    return col


def index_report(cfg, rid):
    """单份向量化入库（graph 与 CLI 共用）。返回对账 record；对账不一致抛异常。"""

    src = require_file(os.path.join(abspath(cfg_get(cfg, "paths.chunks")), rid + ".json"),
                       "切片 JSON（先运行 step4）")
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]
    md = data["metadata"]

    rows = read_manifest(cfg)
    mrows = ([r for r in rows if os.path.splitext(r["file_name"])[0] == rid]
             or [r for r in rows
                 if logical_report_id(os.path.splitext(r["file_name"])[0]) == rid])
    if not mrows:
        fail("对照表中无此 report_id: %s" % rid)

    mil = cfg["milvus"]  # 整块取用
    for k in ("host", "port", "collection", "metric_type", "index_type", "index_params", "varchar_limits"):
        if k not in mil:
            fail("配置缺失: milvus.%s" % k)

    client = get_client(cfg)
    print("向量化 %d 片..." % len(chunks))
    vectors = embed_batches(cfg, client, [c["text"] for c in chunks])

    col = ensure_collection(cfg, mil)
    col.load()  # delete/insert/query 均要求 collection 处于 loaded 状态
    # 单份重灌：先删旧数据
    col.delete("report_name == \"%s\"" % rid)

    from pymilvus import Collection
    batch = 500
    for start in range(0, len(chunks), batch):
        part = chunks[start:start + batch]
        col.insert([{
            "chunk_id": c["chunk_id"],
            "vector": vectors[start + k],
            "text": c["text"],
            "report_name": md["report_name"],
            "company": md["company"],
            "stock_code": md.get("stock_code", "") or "",
            "report_type": md["report_type"],
            "period": md["period"],
            "source_path": md["source_pdf_path"],
            "page": c["page"],
            "chunk_index": c["chunk_index"],
        } for k, c in enumerate(part)])
    col.flush()

    # 逐份对账
    col.load()
    in_db = col.query("report_name == \"%s\"" % rid, output_fields=["chunk_id"])
    inserted = len(in_db)
    expected = len(chunks)
    ok = inserted == expected
    record = {
        "report_id": rid,
        "chunks": expected,
        "inserted": inserted,
        "reconciled": ok,
    }
    if not ok:
        append_log(cfg, "step5_milvus_report", record)
        raise RuntimeError("对账失败：Milvus 中 %s 入库 %d 条 != 切片数 %d（已记入报告）"
                           % (rid, inserted, expected))
    append_log(cfg, "step5_milvus_report", record)
    print("入库完成 %s：%d 条，对账一致，报告已追加 step5_milvus_report.json" % (rid, inserted))
    return record


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_id", required=True, help="单份重灌粒度 = 单份财报")
    args = ap.parse_args()
    index_report(cfg, args.report_id)


if __name__ == "__main__":
    main()
