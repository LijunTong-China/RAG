# -*- coding: utf-8 -*-
"""超限 PDF 拆分工具：把超过 MinerU 平台页数上限（默认 200 页）的 PDF
拆成多个 ≤max_pages 的分卷 PDF，写入指定输出目录，命名 {原文件名去扩展名}_partN.pdf。
源文件始终不动。

用法：
    python split_pdf.py --pdf "data/original_pdfs/xxx.pdf"        # 按配置上限拆分
    python split_pdf.py --pdf xxx.pdf --max_pages 200             # 显式指定
分卷统一输出到 data/00_raw_pdfs/（派生工作目录，与源目录分离）。
"""
import argparse
import os

from common import PROJECT_ROOT, abspath, cfg_get, fail, load_config


def pdf_page_count(path):
    from pypdf import PdfReader
    return len(PdfReader(path).pages)


def split_pdf(pdf_path, max_pages, out_dir):
    """把超限 PDF 拆成 _partN.pdf（≤max_pages），分卷写入 out_dir（源文件不动）。
    返回分卷路径列表；未超限返回 [原路径]。"""
    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(pdf_path)
    total = len(reader.pages)
    if total <= max_pages:
        return [pdf_path]

    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    parts = []
    part = 1
    for start in range(0, total, max_pages):
        w = PdfWriter()
        for i in range(start, min(start + max_pages, total)):
            w.add_page(reader.pages[i])
        out = os.path.join(out_dir, "%s_part%d.pdf" % (stem, part))
        with open(out, "wb") as f:
            w.write(f)
        print("  [拆卷] %s_part%d.pdf（第 %d-%d 页）"
              % (stem, part, start + 1, min(start + max_pages, total)))
        parts.append(out)
        part += 1
    return parts


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--max_pages", type=int, default=None,
                    help="默认取配置 mineru.max_pages")
    args = ap.parse_args()
    max_pages = args.max_pages or int(cfg_get(cfg, "mineru.max_pages"))

    pdf_path = abspath(args.pdf)
    if not os.path.isfile(pdf_path):
        fail("文件不存在: %s" % pdf_path)
    total = pdf_page_count(pdf_path)
    if total <= max_pages:
        print("共 %d 页，未超上限 %d，无需拆分" % (total, max_pages))
        return
    print("共 %d 页，超上限 %d，开始拆分：" % (total, max_pages))
    split_pdf(pdf_path, max_pages, abspath(cfg_get(cfg, "paths.raw_pdfs")))


if __name__ == "__main__":
    main()
