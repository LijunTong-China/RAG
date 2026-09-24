# -*- coding: utf-8 -*-
"""QA 侧公共工具：复用数据处理侧 common.py 的通用能力 + 提示词加载 + 语料库范围。

严格模式原则与 scripts/common.py 一致：配置缺失即报错，不兜底。
"""
import json
import os
import re
import sys

_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from common import (abspath, append_log, cfg_get, load_config, read_manifest,
                    logical_report_id)  # noqa: E402,F401
from prompts import load_prompt as _load_yaml_prompt  # noqa: E402  统一提示词管理


def get_qa_cfg(cfg):
    """qa_service 整块配置，缺失即报错。"""
    qa = cfg.get("qa_service")
    if not qa:
        raise KeyError("配置缺失: [qa_service]")
    return qa


def load_prompt(cfg, name):
    """从 config/prompts.yaml 加载 QA 提示词（P0~P6 键名），统一维护入口。"""
    return _load_yaml_prompt(name, cfg)


def fill_prompt(template, **vars):
    """{var} 占位注入；未提供的占位符保留原样（P3 的 {company_list} 等由调用方负责）。"""
    out = template
    for k, v in vars.items():
        out = out.replace("{%s}" % k, str(v))
    return out


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_json_strict(text):
    """从模型输出提取纯 JSON（容忍 markdown 围栏），失败抛 ValueError。"""
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError("输出中未找到 JSON 对象: %r" % text[:200])
    return json.loads(m.group(0))


def log_turn(cfg, record):
    """问答全过程日志：data/logs/qa_turns.json（追加式）。"""
    append_log(cfg, "qa_turns", record)


def corpus_scope(cfg):
    """语料库范围：公司清单 / 报告期清单（来自 file_manifest，供 P3/P4 注入对齐枚举）。

    返回 {reports: [...], company_list, period_list}；
    reports 每项 {report_id, company, stock_code, report_type, period}，stock_code 为空则 None。
    """
    rows = read_manifest(cfg)
    seen = {}
    for r in rows:
        rid = logical_report_id(os.path.splitext(r["file_name"])[0])
        if rid in seen:
            continue
        seen[rid] = {
            "report_id": rid,
            "company": r["parsed_company"],
            "stock_code": (r.get("parsed_stock_code") or "").strip() or None,
            "report_type": r["parsed_report_type"],
            "period": r["parsed_period"],
        }
    companies = sorted({v["company"] for v in seen.values() if v["company"]})
    periods = sorted({v["period"] for v in seen.values() if v["period"]})
    return {"reports": list(seen.values()), "company_list": companies, "period_list": periods}
