# -*- coding: utf-8 -*-
"""步骤4：摊平 JSON -> 逐页切片（≤300 token/片，重叠 100 token，token 口径见配置）。

- 切片边界不越过页；页尾不足一片并入最后一片，不跨页拼接
- 按句子边界聚合封片；单句超上限时句内二分
- 相邻片重叠 = 配置 overlap_tokens，回退落在句子边界
- 质量校验：无空切片；单片 ≤ max_tokens；chunk_id/page 与 pages 对应
"""
import argparse
import json
import os
import re

from common import (abspath, append_log, cfg_get, count_tokens, ensure_dir,
                    fail, get_token_counter, load_config, require_file)


def split_sentences(cfg, text):
    look = cfg_get(cfg, "chunking.sentence_split_lookbehind")
    parts = re.split("(?<=%s)" % look, text)
    return [p for p in parts if p]


def bisect_long_sentence(cfg, sent, max_tokens, counter):
    """单句超上限时按 token 二分，直到 ≤ max_tokens。"""
    if counter.encode(sent) and len(counter.encode(sent)) <= max_tokens:
        return [sent]
    if len(sent) == 1:
        raise ValueError("单字符即超 token 上限，无法二分: %r" % sent)
    mid = len(sent) // 2
    return (bisect_long_sentence(cfg, sent[:mid], max_tokens, counter)
            + bisect_long_sentence(cfg, sent[mid:], max_tokens, counter))


def chunk_page(cfg, text, max_tokens, overlap_tokens):
    counter = get_token_counter(cfg)
    sents = []
    for s in split_sentences(cfg, text):
        n = len(counter.encode(s))
        if n > max_tokens:
            sents.extend((x, len(counter.encode(x))) for x in bisect_long_sentence(cfg, s, max_tokens, counter))
        elif s:
            sents.append((s, n))

    chunks = []
    i = 0
    while i < len(sents):
        tok = 0
        j = i
        while j < len(sents) and tok + sents[j][1] <= max_tokens:
            tok += sents[j][1]
            j += 1
        if j == i:  # 理论上不会发生（超长句已二分），防御性终止而非兜底
            fail("切片逻辑异常：第 %d 句无法封片" % i)
        piece = "".join(s[0] for s in sents[i:j])
        chunks.append((piece, tok))
        if j >= len(sents):
            break  # 页内句子已用尽，页尾不足一片的部分已并入本片，不再产出纯重叠片
        # 回退约 overlap_tokens，落在句子边界，保证重叠
        back = 0
        k = j - 1
        while k > i and back < overlap_tokens:
            back += sents[k][1]
            k -= 1
        i = k if k > i else i + 1
    return chunks


def chunk_report(cfg, rid):
    """切换单份财报（graph 与 CLI 共用）。质量校验不过直接 fail。"""

    src = require_file(os.path.join(abspath(cfg_get(cfg, "paths.flattened")), rid + ".json"),
                       "摊平 JSON（先运行 step3）")
    with open(src, "r", encoding="utf-8") as f:
        flat = json.load(f)

    max_tokens = int(cfg_get(cfg, "chunking.max_tokens"))
    overlap_tokens = int(cfg_get(cfg, "chunking.overlap_tokens"))
    if overlap_tokens >= max_tokens:
        fail("配置错误: overlap_tokens(%d) 必须小于 max_tokens(%d)" % (overlap_tokens, max_tokens))

    chunks = []
    for pg in flat["pages"]:
        for seq, (text, tok) in enumerate(chunk_page(cfg, pg["text"], max_tokens, overlap_tokens)):
            chunks.append({
                "chunk_id": "%s::p%d::c%03d" % (rid, pg["page"], seq),
                "chunk_index": seq,
                "page": pg["page"],
                "token_count": tok,
                "text": text,
            })

    # 质量约束
    empty = [c["chunk_id"] for c in chunks if not c["text"].strip()]
    if empty:
        fail("存在空切片，违反质量约束: %s" % empty[:10])
    over = [c["chunk_id"] for c in chunks if c["token_count"] > max_tokens]
    if over:
        fail("存在超限切片: %s" % over[:10])
    page_set = {pg["page"] for pg in flat["pages"]}
    bad = [c["chunk_id"] for c in chunks if c["page"] not in page_set]
    if bad:
        fail("切片页码与 pages 对不上: %s" % bad[:10])

    out = {
        "report_id": rid,
        "metadata": dict(flat["metadata"], total_chunks=len(chunks)),
        "pages": flat["pages"],
        "chunks": chunks,
    }
    out_dir = ensure_dir(abspath(cfg_get(cfg, "paths.chunks")))
    dst = os.path.join(out_dir, rid + ".json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    toks = sorted(c["token_count"] for c in chunks)
    n = len(toks)
    record = {
        "report_id": rid,
        "total_chunks": n,
        "token_mean": round(sum(toks) / n, 1) if n else 0,
        "token_median": toks[n // 2] if n else 0,
        "token_max": toks[-1] if n else 0,
        "pages": len(flat["pages"]),
        "chunks_per_page_mean": round(n / len(flat["pages"]), 2) if flat["pages"] else 0,
    }
    log = append_log(cfg, "step4_chunk_report", record)
    print("切片完成 %s：%d 片（均值 %s / 中位 %s / 最大 %s token），输出 %s\n报告: %s"
          % (rid, n, record["token_mean"], record["token_median"], record["token_max"], dst, log))
    return record


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_id", required=True)
    args = ap.parse_args()
    chunk_report(cfg, args.report_id)


if __name__ == "__main__":
    main()
