# -*- coding: utf-8 -*-
"""步骤2：content_list.json -> 校验 MD + 人工核对记录模板。

- 渲染：text->段落；table->markdown 表格(含标题/脚注)；image->占位；每页开头插入页标记
- 抽检页（按配置）：第1页、目录页、随机正文2页、表格最多的1页 -> 写入 qa_records 模板
- 通过/不通过由人工在 qa_records 中填写「抽检结论: 通过/不通过」，脚本不代判
"""
import argparse
import json
import os
import random
import re

from common import (abspath, append_log, cfg_get, ensure_dir, fail,
                    get_llm_client, llm_chat, load_config, require_file)
from prompts import load_prompt


def _slice_page_md(md_text, page):
    """从渲染 MD 中截取指定页的片段（页标记到下一个页标记之间）。"""
    marker = "<!-- ===== 第 %d 页 ===== -->" % page
    start = md_text.find(marker)
    if start < 0:
        return ""
    nxt = md_text.find("<!-- ===== 第", start + len(marker))
    return md_text[start:nxt if nxt > 0 else len(md_text)]


def autocheck_pages(cfg, rid, blocks, md_path, sample_pages):
    """LLM 自动校验：抽检页的原始块 vs 渲染 MD 逐页对比（数字/表格结构/完整性）。
    结论写入 qa_records 供参考，人工结论行不受影响。"""
    client = get_llm_client(cfg)
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    results = []
    for pno in sample_pages:
        orig = [b for b in blocks if b.get("page_idx") == pno - 1]
        sys_paged = load_prompt("md_autocheck_system", cfg).format(page=pno)
        user = ("页码：%d\nA) 原始块：\n%s\nB) Markdown：\n%s"
                % (pno, json.dumps(orig, ensure_ascii=False), _slice_page_md(md_text, pno)))
        raw = llm_chat(cfg, client, sys_paged, user)
        m = re.search(r"```(?:json)?\s*(.+?)\s*```", raw, re.S)
        try:
            obj = json.loads(m.group(1) if m else raw)
        except json.JSONDecodeError:
            obj = {"page": pno, "verdict": "解析失败", "issues": ["LLM 输出非 JSON: %s" % raw[:200]]}
        results.append(obj)
    return results


def append_autocheck(qa_path, results):
    """自动校验结论追加到 qa_records（不改动人工结论行）。"""
    from datetime import date
    verdicts = [r.get("verdict", "?") for r in results]
    overall = "不通过" if "不通过" in verdicts else ("通过" if "通过" in verdicts else "未完成")
    with open(qa_path, "a", encoding="utf-8") as f:
        f.write("\n## LLM 自动校验（%s，仅供参考，不阻塞流程）\n\n" % date.today().isoformat())
        for r in results:
            issues = "；".join(r.get("issues", [])) or "未发现问题"
            f.write("- 第 %s 页：%s（%s）\n" % (r.get("page"), r.get("verdict"), issues))
        f.write("\n- 总体建议：%s\n" % overall)


def render_block(b):
    t = b.get("type")
    if t in ("text", "header"):
        # header（页眉）按正文段落渲染，便于与原 PDF 逐页对照
        return b.get("text", "")
    if t == "page_number":
        return "<!-- 页码: %s -->" % b.get("text", "")
    if t == "table":
        lines = []
        cap = b.get("table_caption")
        if isinstance(cap, list):
            cap = " ".join(cap)
        if cap:
            lines.append("**%s**" % cap)
        lines.append(b.get("table_body", ""))
        foot = b.get("table_footnote")
        if isinstance(foot, list):
            foot = " ".join(foot)
        if foot:
            lines.append("*%s*" % foot)
        return "\n\n".join(lines)
    if t == "image":
        cap = b.get("img_caption") or b.get("image_caption") or ""
        if isinstance(cap, list):
            cap = " ".join(cap)
        path = b.get("img_path") or b.get("image_path") or ""
        return "![%s](%s)" % (cap, path)
    if t == "equation":
        return "$$%s$$" % b.get("text", "")
    return json_dumps_safe(b)


def json_dumps_safe(b):
    import json
    return "<!-- 未识别块类型: %s -->" % json.dumps(b, ensure_ascii=False)[:200]


def render_md(cfg, blocks):
    lines = []
    cur_page = None
    for b in blocks:
        page = b.get("page_idx")
        if page is None:
            continue
        if page != cur_page:
            if cur_page is not None:
                lines.append("")
            lines.append("<!-- ===== 第 %d 页 ===== -->" % (page + 1))  # page_idx 0基 -> 页码 1基
            cur_page = page
        rendered = render_block(b)
        if rendered:
            lines.append(rendered)
            lines.append("")
    return "\n".join(lines)


def pick_sample_pages(cfg, blocks):
    sp = cfg["qa"]["sample_pages"]  # 整块取用，缺项由校验器报
    for k in ("first_page", "toc_search_max_page", "toc_keyword", "random_body_pages",
              "max_table_pages", "random_seed"):
        if k not in sp:
            fail("配置缺失: qa.sample_pages.%s" % k)
    pages = {}
    tables_per_page = {}
    for b in blocks:
        p = b.get("page_idx")
        if p is None:
            continue
        if b.get("type") == "table":
            tables_per_page[p] = tables_per_page.get(p, 0) + 1

    chosen = {sp["first_page"] - 1}  # 0基
    toc_page = None
    for b in blocks:
        if b.get("type") == "text" and sp["toc_keyword"] in (b.get("text") or ""):
            if b.get("page_idx", 10**9) < sp["toc_search_max_page"]:
                toc_page = b["page_idx"]
                break
    if toc_page is not None:
        chosen.add(toc_page)
    if tables_per_page:
        max_table_pages = sorted(tables_per_page, key=lambda p: (-tables_per_page[p], p))[:sp["max_table_pages"]]
        chosen.update(max_table_pages)
    body = sorted(set(tables_per_page) | {b.get("page_idx") for b in blocks if b.get("type") == "text"})
    body = [p for p in body if p is not None and p not in chosen]
    rng = random.Random(sp["random_seed"])
    chosen.update(rng.sample(body, min(sp["random_body_pages"], len(body))))
    for p in chosen:
        pages[p + 1] = "第 %d 页" % (p + 1)
    return sorted(pages), toc_page


def render_report(cfg, rid):
    """渲染单份校验 MD + qa_records 模板（graph 与 CLI 共用）。"""

    src = require_file(
        os.path.join(abspath(cfg_get(cfg, "paths.mineru_json")), rid, "%s_content_list.json" % rid),
        "content_list.json（先运行 step1）")
    with open(src, "r", encoding="utf-8") as f:
        blocks = json.load(f)

    md_dir = ensure_dir(abspath(cfg_get(cfg, "paths.check_md")))
    qa_dir = ensure_dir(abspath(cfg_get(cfg, "paths.qa_records")))
    md_path = os.path.join(md_dir, rid + ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_md(cfg, blocks))

    sample_pages, toc_page = pick_sample_pages(cfg, blocks)
    conclusion_field = cfg_get(cfg, "qa.conclusion_field")
    qa_path = os.path.join(qa_dir, rid + ".md")
    with open(qa_path, "w", encoding="utf-8") as f:
        f.write("# 核对记录：%s\n\n" % rid)
        f.write("- 核对日期：（人工填写）\n")
        f.write("- 抽检页：%s\n" % "、".join(str(p) for p in sample_pages))
        f.write("- 目录页：%s\n\n" % (("第 %d 页" % (toc_page + 1)) if toc_page is not None else "未找到（人工指定）"))
        f.write("## 核对方法\n\n每份固定抽 5 页（第 1 页、目录页、随机正文 2 页、表格最多的 1 页）对照原 PDF。\n\n")
        f.write("## 通过标准\n\n任一抽检页出现数字错误或表格结构错乱 = 不通过 -> 调 MinerU 参数重跑该份（回步骤1）再抽检。只通过/不通过。\n\n")
        f.write("## 抽检记录\n\n（人工填写各抽检页核对情况）\n\n")
        f.write("## %s\n\n%s: 待定（填写「%s」或「不通过」）\n"
                % (conclusion_field, conclusion_field, cfg_get(cfg, "qa.pass_keyword")))

    append_log(cfg, "step2_render_report", {"report_id": rid, "md": md_path, "qa_record": qa_path,
                                            "sample_pages": sample_pages})

    # LLM 自动校验（可配置开关 prompts.autocheck）：结论追加进 qa_records，不阻塞
    if cfg_get(cfg, "prompts.autocheck"):
        results = autocheck_pages(cfg, rid, blocks, md_path, sample_pages)
        append_autocheck(qa_path, results)
        print("LLM 自动校验完成（%d 页），结论已追加到 %s" % (len(results), qa_path))
    print("已生成 %s\n已生成 %s（抽检页: %s）——请人工核对并填写「%s: 通过/不通过」"
          % (md_path, qa_path, sample_pages, conclusion_field))
    return {"report_id": rid, "md": md_path, "qa_record": qa_path, "sample_pages": sample_pages}


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_id", required=True)
    args = ap.parse_args()
    render_report(cfg, args.report_id)


if __name__ == "__main__":
    main()
