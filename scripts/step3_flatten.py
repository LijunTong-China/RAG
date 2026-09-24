# -*- coding: utf-8 -*-
"""步骤3：content_list.json -> 摊平 JSON（财报级元数据 + 页级纯文本）。

- 闸门：qa_records/{report_id}.md 中「抽检结论: 通过」方可运行
- 摊平：块按 page_idx 分页；页内原顺序拼接；table->逐行 markdown 文本（含表标题）；image->[图: 图题]
- 空页保留并标记，报告列清单；不跳过不编造
"""
import argparse
import json
import os
import re
from html.parser import HTMLParser

from common import (abspath, append_log, cfg_get, count_tokens, ensure_dir,
                    fail, load_config, logical_report_id, read_manifest,
                    require_file)


class _TableHTMLParser(HTMLParser):
    """把 HTML 表格解析为行数组，不猜任何结构。"""
    def __init__(self):
        super().__init__()
        self.rows = []
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag):
        if tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None
        elif tag in ("td", "th") and self._cell is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_table_to_text(html):
    if not html:
        return ""
    p = _TableHTMLParser()
    try:
        p.feed(html)
    except Exception as e:
        raise ValueError("表格 HTML 解析失败（不做兜底）: %s" % e)
    return "\n".join("| " + " | ".join(row) + " |" for row in p.rows)


def page_text_from_blocks(blocks, page, skipped):
    parts = []
    for b in blocks:
        if b.get("page_idx") != page:
            continue
        t = b.get("type")
        if t == "text":
            parts.append(b.get("text", ""))
        elif t == "header":
            parts.append(b.get("text", ""))
        elif t == "page_number":
            continue  # 页码已在 page 元数据中，不重复进正文
        elif t == "page_footnote":
            parts.append(b.get("text", ""))  # 脚注常含真实数据口径，进正文
        elif t in ("footer", "aside_text"):
            # 页脚/侧边栏装饰性文字（如每页重复的"后附财务报表附注…"），与页码同款跳过
            skipped[t] = skipped.get(t, 0) + 1
            continue
        elif t == "table":
            cap = b.get("table_caption")
            if isinstance(cap, list):
                cap = " ".join(cap)
            foot = b.get("table_footnote")
            if isinstance(foot, list):
                foot = " ".join(foot)
            if cap:
                parts.append(str(cap))
            body = html_table_to_text(b.get("table_body", ""))
            if body:
                parts.append(body)
            if foot:
                parts.append(str(foot))
        elif t == "image":
            cap = b.get("img_caption") or b.get("image_caption") or ""
            if isinstance(cap, list):
                cap = " ".join(cap)
            parts.append("[图: %s]" % cap)
        elif t == "chart":
            cap = b.get("chart_caption") or ""
            if isinstance(cap, list):
                cap = " ".join(cap)
            parts.append("[图: %s]" % cap)
        elif t == "equation":
            parts.append(b.get("text", ""))
        else:
            raise ValueError("未识别块类型 type=%r（page %d），不做兜底，请扩展 step3" % (t, page))
    return "\n".join(x for x in parts if x)


def qa_status(cfg, rid):
    """读取人工抽检状态（仅供参考提醒，不做闸门，不阻塞后续流程）。"""
    import re
    qa_path = os.path.join(abspath(cfg_get(cfg, "paths.qa_records")), rid + ".md")
    if not os.path.isfile(qa_path):
        return "未生成"
    with open(qa_path, "r", encoding="utf-8") as f:
        content = f.read()
    field = cfg_get(cfg, "qa.conclusion_field")
    m = re.search(r"%s[:：]\s*(\S+)" % re.escape(field), content)
    if not m:
        return "未填写"
    if m.group(1) == cfg_get(cfg, "qa.pass_keyword"):
        return "通过"
    return m.group(1)   # 原样显示（待定/不通过等），不猜状态


def flatten_report(cfg, rid):
    """摊平单份财报（graph 与 CLI 共用）。人工抽检状态只提示，不阻塞。"""

    status = qa_status(cfg, rid)
    if status != "通过":
        print("[提醒] 人工抽检状态为「%s」，不阻塞流程，建议尽快抽检" % status)

    src = require_file(
        os.path.join(abspath(cfg_get(cfg, "paths.mineru_json")), rid, "%s_content_list.json" % rid),
        "content_list.json（先运行 step1）")
    with open(src, "r", encoding="utf-8") as f:
        blocks = json.load(f)

    rows = read_manifest(cfg)
    mrows = ([r for r in rows if os.path.splitext(r["file_name"])[0] == rid]
             or [r for r in rows
                 if logical_report_id(os.path.splitext(r["file_name"])[0]) == rid])
    if not mrows:
        fail("对照表中无此 report_id: %s（元数据以对照表为唯一来源）" % rid)
    mrow = mrows[0]
    filled_status = {cfg_get(cfg, "manifest_status.manual_filled"), cfg_get(cfg, "manifest_status.llm_filled")}
    if mrow["parse_status"] in filled_status and not mrow["parsed_company"]:
        fail("对照表条目 %s 状态为补填但字段为空: %s" % (rid, mrow))

    page_ids = sorted({b.get("page_idx") for b in blocks if b.get("page_idx") is not None})
    skipped = {}
    pages = [{"page": p + 1, "text": page_text_from_blocks(blocks, p, skipped)} for p in page_ids]
    empty_pages = [pg["page"] for pg in pages if not pg["text"].strip()]
    if skipped:
        print("[跳过装饰块] %s：%s（页脚/侧边栏重复性文字，不入正文）"
              % (rid, "、".join("%s×%d" % kv for kv in sorted(skipped.items()))))

    out = {
        "report_id": rid,
        "metadata": {
            "report_name": rid,
            "company": mrow["parsed_company"],
            "total_pages": len(pages),
            "source_pdf_path": mrow["pdf_path"],
            "mineru_json_path": os.path.relpath(src, os.path.dirname(abspath("."))).replace("\\", "/"),
            "stock_code": mrow["parsed_stock_code"],
            "report_type": mrow["parsed_report_type"],
            "period": mrow["parsed_period"],
        },
        "pages": pages,
    }
    out_dir = ensure_dir(abspath(cfg_get(cfg, "paths.flattened")))
    dst = os.path.join(out_dir, rid + ".json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    page_chars = [{"page": pg["page"], "chars": len(pg["text"])} for pg in pages]
    table_pages = sorted({b["page_idx"] + 1 for b in blocks if b.get("type") == "table"})
    record = {
        "report_id": rid,
        "total_pages": len(pages),
        "total_tokens_approx": sum(count_tokens(cfg, pg["text"]) for pg in pages),
        "empty_pages": empty_pages,
        "table_pages": table_pages,
        "page_chars": page_chars,
    }
    log = append_log(cfg, "step3_flatten_report", record)
    print("摊平完成 %s：%d 页（空页 %d 个：\n%s\n），输出 %s\n报告: %s"
          % (rid, len(pages), len(empty_pages), empty_pages if empty_pages else "无", dst, log))
    return record


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_id", required=True)
    args = ap.parse_args()
    flatten_report(cfg, args.report_id)


if __name__ == "__main__":
    main()
