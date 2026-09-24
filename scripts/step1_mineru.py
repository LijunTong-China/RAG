# -*- coding: utf-8 -*-
"""步骤1：通过 MinerU 云端 API（https://mineru.net/api/v4）解析 PDF -> 01_mineru_json/{report_id}/。

- 鉴权：Authorization: Bearer {mineru.api_key}（配置文件，缺失即报错不兜底）
- 批量：POST /api/v4/file-urls/batch 申请上传链接（单批 ≤ mineru.batch_limit）+ PUT 上传，系统自动提交解析
- 轮询：GET /api/v4/extract-results/batch/{batch_id}，state=done 后下载 full_zip_url
- 解包：{report_id}_content_list.json（主用）、layout.json -> {report_id}_middle.json（备用）、images/
- 单份失败只记失败不影响其他份；重跑粒度 = 单份（--report_id，也走批量接口单文件）
"""
import argparse
import io
import json
import os
import re
import shutil
import sys
import time
import zipfile

import requests

from common import (abspath, append_log, cfg_get, ensure_dir, fail,
                    load_config, logical_report_id, part_number, read_manifest)


class MinerUAPI:
    def __init__(self, cfg):
        self.base = cfg_get(cfg, "mineru.api_base").rstrip("/")
        self.token = cfg_get(cfg, "mineru.api_key")
        self.poll_interval = int(cfg_get(cfg, "mineru.poll_interval_seconds"))
        self.poll_timeout = int(cfg_get(cfg, "mineru.poll_timeout_seconds"))

    def _headers(self, json_body=True):
        h = {"Authorization": "Bearer " + self.token}
        if json_body:
            h["Content-Type"] = "application/json"
        return h

    def _check(self, r, what):
        if r.status_code != 200:
            raise RuntimeError("%s HTTP %d: %s" % (what, r.status_code, r.text[:500]))
        body = r.json()
        if body.get("code") != 0:
            raise RuntimeError("%s 业务失败 code=%s msg=%s" % (what, body.get("code"), body.get("msg")))
        return body["data"]

    def apply_upload_urls(self, files_payload):
        """POST /file-urls/batch -> (batch_id, file_urls)。"""
        r = requests.post(self.base + "/file-urls/batch", headers=self._headers(),
                          json=files_payload, timeout=60)
        data = self._check(r, "申请上传链接")
        return data["batch_id"], data["file_urls"]

    def upload_file(self, upload_url, pdf_path):
        """PUT 上传本地文件（无须 Content-Type）。"""
        with open(pdf_path, "rb") as f:
            r = requests.put(upload_url, data=f, timeout=600)
        if r.status_code != 200:
            raise RuntimeError("上传失败 HTTP %d: %s" % (r.status_code, pdf_path))

    def poll_batch(self, batch_id):
        """轮询直到批内全部 done/failed，返回 extract_result 列表。"""
        deadline = time.time() + self.poll_timeout
        while True:
            r = requests.get(self.base + "/extract-results/batch/" + batch_id,
                             headers=self._headers(False), timeout=60)
            results = self._check(r, "查询批量结果")["extract_result"]
            states = {x["state"] for x in results}
            if states <= {"done", "failed"}:
                return results
            if time.time() > deadline:
                raise RuntimeError("轮询超时（%ds），当前状态: %s"
                                   % (self.poll_timeout, {x["file_name"]: x["state"] for x in results}))
            time.sleep(self.poll_interval)


def extract_zip_outputs(cfg, zip_bytes, report_id, final_dir):
    """从结果 zip 严格取出 content_list / layout.json / images，落成标准目录布局。"""
    content_suffix = cfg_get(cfg, "mineru.zip_content_list_suffix")
    middle_name = cfg_get(cfg, "mineru.zip_middle_name")
    images_name = cfg_get(cfg, "mineru.images_dir_name")

    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    names = zf.namelist()
    content_entries = [n for n in names if n.endswith(content_suffix)]
    if len(content_entries) != 1:
        raise RuntimeError("zip 内 %s 文件应恰为 1 个，实际 %d 个: %s"
                           % (content_suffix, len(content_entries), names))
    middle_entries = [n for n in names if os.path.basename(n) == middle_name]
    if len(middle_entries) != 1:
        raise RuntimeError("zip 内 %s 文件应恰为 1 个，实际 %d 个: %s"
                           % (middle_name, len(middle_entries), names))

    content_dst = os.path.join(final_dir, "%s_content_list.json" % report_id)
    middle_dst = os.path.join(final_dir, "%s_middle.json" % report_id)
    with open(content_dst, "wb") as f:
        f.write(zf.read(content_entries[0]))
    with open(middle_dst, "wb") as f:
        f.write(zf.read(middle_entries[0]))

    images_rel = ""
    for n in names:
        parts = n.split("/")
        if len(parts) >= 2 and parts[0] == images_name and not n.endswith("/"):
            dst = os.path.join(final_dir, *parts)
            ensure_dir(os.path.dirname(dst))
            with open(dst, "wb") as f:
                f.write(zf.read(n))
            images_rel = os.path.join(final_dir, images_name)
    return content_dst, middle_dst, images_rel


def process_batch(cfg, api, rows):
    """一批 PDF：申请链接 -> 上传 -> 轮询 -> 解包落盘。返回 {report_id: info}，异常进 failed。"""
    limit = int(cfg_get(cfg, "mineru.batch_limit"))
    if len(rows) > limit:
        fail("单批文件数 %d 超过配置 mineru.batch_limit=%d" % (len(rows), limit))

    payload = {
        "model_version": cfg_get(cfg, "mineru.model_version"),
        "enable_table": cfg_get(cfg, "mineru.enable_table"),
        "enable_formula": cfg_get(cfg, "mineru.enable_formula"),
        "language": cfg_get(cfg, "mineru.language"),
        "files": [{"name": r["file_name"], "data_id": os.path.splitext(r["file_name"])[0]} for r in rows],
    }
    batch_id, urls = api.apply_upload_urls(payload)
    if len(urls) != len(rows):
        raise RuntimeError("上传链接数 %d != 文件数 %d" % (len(urls), len(rows)))
    for row, url in zip(rows, urls):
        api.upload_file(url, abspath(row["pdf_path"]))
        print("[已上传] %s" % row["file_name"])

    results = api.poll_batch(batch_id)
    out_root = abspath(cfg_get(cfg, "paths.mineru_json"))
    infos = {}
    for row, res in zip(rows, results):
        rid = os.path.splitext(row["file_name"])[0]
        if res["state"] != "done":
            raise RuntimeError("解析状态 %s: %s" % (res["state"], res.get("err_msg", "")))
        if not res.get("full_zip_url"):
            raise RuntimeError("state=done 但缺少 full_zip_url")
        zr = requests.get(res["full_zip_url"], timeout=600)
        if zr.status_code != 200:
            raise RuntimeError("结果 zip 下载失败 HTTP %d" % zr.status_code)
        final_dir = ensure_dir(os.path.join(out_root, rid))
        content, middle, images = extract_zip_outputs(cfg, zr.content, rid, final_dir)
        with open(content, "r", encoding="utf-8") as f:
            page_ids = {b.get("page_idx") for b in json.load(f) if isinstance(b, dict)}
        infos[rid] = {
            "report_id": rid,
            "content_list": content,
            "middle_json": middle,
            "images_dir": images,
            "pages": len([p for p in page_ids if p is not None]),
            "batch_id": batch_id,
        }
    return infos


def merge_parts(cfg, base_id, part_ids):
    """把多个拆卷的解析产物合并成逻辑报告 base_id 的标准输出。

    - content_list：part 序号大的 page_idx 整体偏移前面各卷页数后拼接
    - images：改名 {partN}_原文件名 复制到 base/images/，并重写块内 img_path（防跨卷重名）
    - middle.json：pdf_info 页数组按序拼接（备用数据）
    任一卷产物缺失 / middle 结构未知 -> 抛异常，不兜底。
    """
    out_root = abspath(cfg_get(cfg, "paths.mineru_json"))
    base_dir = os.path.join(out_root, base_id)
    images_dir = ensure_dir(os.path.join(base_dir, "images"))
    merged_blocks = []
    merged_pdf_info = []
    offset = 0
    for pid in sorted(part_ids, key=part_number):
        pdir = os.path.join(out_root, pid)
        content = os.path.join(pdir, "%s_content_list.json" % pid)
        middle = os.path.join(pdir, "%s_middle.json" % pid)
        if not os.path.isfile(content) or not os.path.isfile(middle):
            raise RuntimeError("拆卷 %s 产物不齐（先完成全部拆卷解析）" % pid)
        with open(content, "r", encoding="utf-8") as f:
            blocks = json.load(f)
        pno = part_number(pid)
        for b in blocks:
            if not isinstance(b, dict):
                raise RuntimeError("拆卷 %s content_list 含非字典块" % pid)
            b = dict(b)
            if b.get("page_idx") is not None:
                b["page_idx"] += offset
            ip = b.get("img_path")
            if ip:
                src = os.path.join(pdir, ip.replace("/", os.sep))
                if not os.path.isfile(src):
                    raise RuntimeError("拆卷 %s 图片缺失: %s" % (pid, ip))
                new_rel = "images/%s_%s" % (pno, os.path.basename(ip))
                shutil.copyfile(src, os.path.join(images_dir, "%s_%s" % (pno, os.path.basename(ip))))
                b["img_path"] = new_rel
            merged_blocks.append(b)
        with open(middle, "r", encoding="utf-8") as f:
            mid = json.load(f)
        if not isinstance(mid, dict) or "pdf_info" not in mid:
            raise RuntimeError("拆卷 %s middle.json 无 pdf_info 字段，无法按页合并" % pid)
        merged_pdf_info.extend(mid["pdf_info"])
        offset += max([b.get("page_idx") for b in blocks if b.get("page_idx") is not None],
                      default=-1) + 1

    content_dst = os.path.join(base_dir, "%s_content_list.json" % base_id)
    with open(content_dst, "w", encoding="utf-8") as f:
        json.dump(merged_blocks, f, ensure_ascii=False, indent=2)
    middle_dst = os.path.join(base_dir, "%s_middle.json" % base_id)
    with open(middle_dst, "w", encoding="utf-8") as f:
        json.dump({"pdf_info": merged_pdf_info}, f, ensure_ascii=False, indent=2)
    page_ids = {b.get("page_idx") for b in merged_blocks if isinstance(b, dict)}
    return {"report_id": base_id, "merged_from": sorted(part_ids, key=part_number),
            "content_list": content_dst, "middle_json": middle_dst,
            "images_dir": os.path.join(base_dir, "images"),
            "pages": len([p for p in page_ids if p is not None]), "merged": True}


def run_reports(cfg, rows):
    """处理放行 rows（可一批或多份），返回 (succeeded, failed)。graph 与 CLI 共用。"""
    api = MinerUAPI(cfg)
    succeeded, failed = [], []
    limit = int(cfg_get(cfg, "mineru.batch_limit"))
    for start in range(0, len(rows), limit):
        batch = rows[start:start + limit]
        try:
            infos = process_batch(cfg, api, batch)
            for rid, info in infos.items():
                succeeded.append(info)
                print("[成功] %s（%d 页）" % (rid, info["pages"]))
        except Exception as e:  # 单批失败：记日志后降为逐份重试，仍失败才入 failed
            print("[批次失败] %s: %s，降为逐份重试" % ([r["file_name"] for r in batch], e))
            for row in batch:
                try:
                    infos = process_batch(cfg, api, [row])
                    for rid, info in infos.items():
                        succeeded.append(info)
                        print("[成功] %s（%d 页）" % (rid, info["pages"]))
                except Exception as e2:
                    rid = os.path.splitext(row["file_name"])[0]
                    failed.append({"report_id": rid, "error": str(e2)})
                    print("[失败] %s: %s" % (rid, e2), file=sys.stderr)

    # 拆卷合并：同一逻辑报告的多个 _partN 全部产物齐了 -> 合并为 base_id 标准输出
    groups = {}
    for r in rows:
        base = logical_report_id(os.path.splitext(r["file_name"])[0])
        groups.setdefault(base, []).append(os.path.splitext(r["file_name"])[0])
    for base, parts in groups.items():
        if len(parts) < 2:
            continue
        try:
            info = merge_parts(cfg, base, parts)
            # 合并成功后删除分卷目录（产物已并入 base，分卷无用）
            out_root = abspath(cfg_get(cfg, "paths.mineru_json"))
            for pid in parts:
                shutil.rmtree(os.path.join(out_root, pid))
            succeeded.append(info)
            print("[合并] %s <- %s（共 %d 页），分卷目录已删除"
                  % (base, ", ".join(info["merged_from"]), info["pages"]))
        except Exception as e:
            failed.append({"report_id": base, "error": "拆卷合并失败: %s" % e})
            print("[合并失败] %s: %s" % (base, e), file=sys.stderr)
    return succeeded, failed


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_id", help="只重跑这一份（重跑粒度=单份财报）")
    args = ap.parse_args()

    rows = read_manifest(cfg)
    ok_status = {cfg_get(cfg, "manifest_status.ok"),
                 cfg_get(cfg, "manifest_status.manual_filled"),
                 cfg_get(cfg, "manifest_status.llm_filled")}
    need_manual_status = cfg_get(cfg, "manifest_status.need_manual")

    need_manual = [r for r in rows if r["parse_status"] == need_manual_status]
    if need_manual:
        print("[提醒] 以下 need_manual 未补填，进不了步骤1：")
        for r in need_manual:
            print("  - " + r["pdf_path"])

    if args.report_id:
        rows = [r for r in rows if os.path.splitext(r["file_name"])[0] == args.report_id]
        if not rows:
            fail("对照表中无此 report_id: %s" % args.report_id)

    todo = [r for r in rows if r["parse_status"] in ok_status]
    succeeded, failed = run_reports(cfg, todo)

    record = {"run_scope": args.report_id or "all", "success": succeeded, "failed": failed}
    log = append_log(cfg, "step1_mineru_report", record)
    print("成功 %d / 失败 %d，报告: %s" % (len(succeeded), len(failed), log))
    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
