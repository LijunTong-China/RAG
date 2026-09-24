# -*- coding: utf-8 -*-
"""步骤0：扫描源 PDF 目录（paths.original_pdfs，如 data/original_pdfs/），
整理出可处理的工作副本到 data/00_raw_pdfs/ 并登记 file_manifest.csv。

- 源目录只读不动：超 mineru.max_pages 的 PDF 拆成 _partN 分卷，未超限的整本复制，
  产物统一落到 paths.raw_pdfs（派生工作目录，可随时清空重生成）
- 每次运行先清空 raw_pdfs 里的旧 PDF，避免改名/换文件后残留
- 文件名可按配置正则解析 -> parse_status=ok，元数据自动获取
- 解析不了的 -> 调 LLM（Agicto glm-5.3-flash）识别文件名中的元数据；
  识别成功 -> parse_status=llm_ok；LLM 也给不出有效字段 -> parse_status=need_manual
- 校验：股票代码 6 位数字、报告期 4 位数字，不符合 = 识别失败（不猜值）
- report_id（文件名去扩展名）重名即报错退出
- 运行前置：环境变量 AGICTO_API_KEY（仅当存在不可解析文件名时需要）
"""
import csv
import json
import os
import re

from common import (PROJECT_ROOT, abspath, append_log, cfg_get, ensure_dir,
                    fail, get_llm_client, llm_chat, load_config, parse_filename)
from prompts import load_prompt


def llm_parse_filename(cfg, client, file_name):
    stem = os.path.splitext(file_name)[0]
    system = load_prompt("filename_parse_system", cfg)
    if cfg_get(cfg, "llm.cot"):
        system += "\n" + load_prompt("filename_parse_cot", cfg)
    raw = llm_chat(cfg, client, system, "文件名：%s" % stem)
    # 兼容 ```json 围栏输出
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", raw, re.S)
    text = m.group(1) if m else raw
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    company = str(obj.get("company", "")).strip()
    code = str(obj.get("stock_code", "")).strip()
    rtype = str(obj.get("report_type", "")).strip()
    period = str(obj.get("period", "")).strip()
    # 校验：公司名必须有；其余字段有则须合法，没有则留空（不猜值不编造）
    if not company:
        return None
    if (code and not re.fullmatch(r"\d{6}", code)) or \
       (period and not re.fullmatch(r"\d{4}", period)) or \
       (rtype and rtype not in ("年报", "半年报", "一季报", "三季报", "季报")):
        return None
    return {"parsed_company": company, "parsed_stock_code": code,
            "parsed_report_type": rtype, "parsed_period": period}


def main():
    cfg = load_config()
    src_dir = abspath(cfg_get(cfg, "paths.original_pdfs"))  # 源 PDF 目录（只读）
    raw_dir = abspath(cfg_get(cfg, "paths.raw_pdfs"))       # 派生工作目录（可清）
    if not os.path.isdir(src_dir):
        fail("源 PDF 目录不存在: %s" % src_dir)

    columns = cfg_get(cfg, "manifest_columns")
    ok_status = cfg_get(cfg, "manifest_status.ok")
    need_manual_status = cfg_get(cfg, "manifest_status.need_manual")
    llm_status = cfg_get(cfg, "manifest_status.llm_filled")

    # 收集源 PDF（递归，"_" 开头目录跳过）
    ignore_prefixes = tuple(cfg_get(cfg, "paths.raw_ignore_dir_prefixes"))
    auto_split = bool(cfg_get(cfg, "scan.auto_split"))
    max_pages = int(cfg_get(cfg, "mineru.max_pages"))
    src_pdfs = []
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if not d.startswith(ignore_prefixes)]
        for name in files:
            if name.lower().endswith(".pdf"):
                src_pdfs.append(os.path.join(root, name))
    src_pdfs.sort()
    if not src_pdfs:
        fail("目录下未发现任何 PDF: %s" % src_dir)

    # 重建工作目录：清掉旧副本，超限拆卷 / 未超限复制，源文件不动
    import glob
    import shutil
    os.makedirs(raw_dir, exist_ok=True)
    for old in glob.glob(os.path.join(raw_dir, "*.pdf")):
        os.remove(old)
    pdfs = []
    if auto_split:
        from split_pdf import pdf_page_count, split_pdf
        for p in src_pdfs:
            n = pdf_page_count(p)
            if n > max_pages:
                print("[超限] %s（%d 页 > %d），自动拆分：" % (os.path.relpath(p, src_dir), n, max_pages))
                pdfs.extend(split_pdf(p, max_pages, raw_dir))
            else:
                dst = os.path.join(raw_dir, os.path.basename(p))
                shutil.copy2(p, dst)
                pdfs.append(dst)
    else:
        for p in src_pdfs:
            dst = os.path.join(raw_dir, os.path.basename(p))
            shutil.copy2(p, dst)
            pdfs.append(dst)
    pdfs = sorted(pdfs)
    print("[整理] 源 %d 份 -> 工作副本 %d 份（%s）" % (len(src_pdfs), len(pdfs), raw_dir))

    client = None
    rows = []
    seen_ids = {}
    for pdf in pdfs:
        rel = os.path.relpath(pdf, PROJECT_ROOT).replace("\\", "/")
        name = os.path.basename(pdf)
        parsed = parse_filename(cfg, name)
        status = ok_status
        if parsed is None:
            if client is None:
                client = get_llm_client(cfg)
            llm = llm_parse_filename(cfg, client, name)
            if llm is None:
                status = need_manual_status
                parsed = {}
            else:
                status = llm_status
                parsed = dict(llm, report_id=os.path.splitext(name)[0])
        rid = parsed["report_id"] if parsed else ""
        if rid and rid in seen_ids:
            fail("report_id 重名: %s 与 %s （对照表要求全局唯一）" % (pdf, seen_ids[rid]))
        if rid:
            seen_ids[rid] = pdf
        rows.append({"pdf_path": rel, "file_name": name,
                     **{k: parsed.get(k, "") for k in
                        ("parsed_company", "parsed_stock_code", "parsed_report_type", "parsed_period")},
                     "parse_status": status})

    manifest_dir = ensure_dir(abspath(cfg_get(cfg, "paths.manifest_dir")))
    out = os.path.join(manifest_dir, os.path.basename(cfg_get(cfg, "paths.file_manifest")))
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)

    need_manual = [r for r in rows if r["parse_status"] == need_manual_status]
    llm_cnt = len(rows) - len(need_manual) - len([r for r in rows if r["parse_status"] == ok_status])
    record = {
        "total_pdfs": len(rows),
        "ok": len(rows) - len(need_manual) - llm_cnt,
        "llm_ok": llm_cnt,
        "need_manual": [r["pdf_path"] for r in need_manual],
    }
    log = append_log(cfg, "step0_scan_report", record)
    print("已登记 %d 份 PDF -> %s" % (len(rows), out))
    print("ok=%d, llm_ok=%d, need_manual=%d（明细见 %s）" % (record["ok"], llm_cnt, len(need_manual), log))
    if need_manual:
        print("\n以下 PDF 正则与 LLM 均无法识别元数据，停在流水线外，请人工在对照表补填并将 parse_status 改为 %s："
              % cfg_get(cfg, "manifest_status.manual_filled"))
        for r in need_manual:
            print("  - " + r["pdf_path"])


if __name__ == "__main__":
    main()
