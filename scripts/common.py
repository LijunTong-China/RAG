# -*- coding: utf-8 -*-
"""公共模块：配置加载（严格模式，无默认值兜底）、路径、manifest 读写、token 计数。

原则：任何配置缺失/为空 = 直接报错终止，代码内不提供默认值。
"""
import csv
import json
import os
import re
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "pipeline_config.json")


class ConfigError(Exception):
    pass


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def cfg_get(cfg, dotted_key):
    """按点号路径取配置项，缺失即报错（不兜底）。"""
    cur = cfg
    walked = []
    for part in dotted_key.split("."):
        walked.append(part)
        if not isinstance(cur, dict) or part not in cur:
            raise ConfigError("配置缺失: [%s] （配置文件: %s）" % (".".join(walked), CONFIG_PATH))
        cur = cur[part]
    if cur is None or (isinstance(cur, str) and cur == ""):
        raise ConfigError("配置项为空: [%s] （禁止兜底，请在 %s 中填写）" % (dotted_key, CONFIG_PATH))
    return cur


def abspath(rel):
    return rel if os.path.isabs(rel) else os.path.join(PROJECT_ROOT, rel)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def require_file(path, what):
    if not os.path.isfile(path):
        raise ConfigError("缺少%s: %s" % (what, path))
    return path


# ---------- manifest ----------

def read_manifest(cfg):
    path = abspath(cfg_get(cfg, "paths.file_manifest"))
    require_file(path, "文件对照表（先运行 step0_scan.py 生成并人工确认）")
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def append_log(cfg, step_name, record):
    """data/logs/{step}.json 为追加式：文件内容是 list，每次运行追加一条。"""
    log_dir = ensure_dir(abspath(cfg_get(cfg, "paths.logs")))
    path = os.path.join(log_dir, step_name + ".json")
    history = []
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)
    history.append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return path


# ---------- tokenizer ----------

_TIKTOKEN = None


def get_token_counter(cfg):
    global _TIKTOKEN
    if _TIKTOKEN is not None:
        return _TIKTOKEN
    engine = cfg_get(cfg, "tokenizer.engine")
    if engine != "tiktoken":
        raise ConfigError("不支持的 tokenizer.engine: %s" % engine)
    import tiktoken
    encoding = cfg_get(cfg, "tokenizer.encoding")
    _TIKTOKEN = tiktoken.get_encoding(encoding)
    return _TIKTOKEN


def count_tokens(cfg, text):
    return len(get_token_counter(cfg).encode(text))


# ---------- misc ----------

def get_llm_client(cfg):
    """OpenAI 兼容客户端（Agicto）。key 只走环境变量，缺失即报错。"""
    from openai import OpenAI
    env_key = cfg_get(cfg, "llm.api_key_env")
    key = os.environ.get(env_key)
    if not key:
        raise ConfigError("环境变量 %s 未设置（LLM API key 不兜底）" % env_key)
    return OpenAI(base_url=cfg_get(cfg, "llm.base_url"), api_key=key)


def llm_chat(cfg, client, system, user, response_format=None):
    """response_format 可选（如 {"type": "json_object"}）；不支持时由 API 报错暴露，不静默降级。"""
    model = cfg_get(cfg, "llm.model")
    temperature = cfg_get(cfg, "llm.temperature")
    max_retries = int(cfg_get(cfg, "llm.max_retries"))
    wait_s = int(cfg_get(cfg, "llm.retry_wait_seconds"))
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            kwargs = {"response_format": response_format} if response_format else {}
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                **kwargs,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(wait_s)
    raise ConfigError("LLM 调用失败（已重试 %d 次）: %s" % (max_retries, last_err))


def fail(msg):
    print("[失败] " + msg, file=sys.stderr)
    sys.exit(1)


_PART_RE = re.compile(r"^(.*)_part(\d+)$")


def logical_report_id(stem):
    """拆卷文件名 -> 逻辑报告 id：xxx_partN -> xxx；无 _partN 后缀原样返回。"""
    m = _PART_RE.match(stem)
    return m.group(1) if m else stem


def part_number(stem):
    """拆卷序号；非拆卷文件返回 0。"""
    m = _PART_RE.match(stem)
    return int(m.group(2)) if m else 0


def parse_filename(cfg, file_name):
    """按配置正则解析文件名（去扩展名）。不可解析返回 None（进 need_manual，不猜值）。"""
    pattern = cfg_get(cfg, "filename_pattern")
    stem = os.path.splitext(file_name)[0]
    m = re.match(pattern, stem)
    if not m:
        return None
    return {
        "parsed_company": m.group("company"),
        "parsed_stock_code": m.group("stock_code"),
        "parsed_report_type": m.group("report_type"),
        "parsed_period": m.group("period"),
        "report_id": stem,
    }
