# -*- coding: utf-8 -*-
"""提示词统一加载器：所有 LLM 提示词集中存于 config/prompts.yaml（一个文件管理），
脚本内零硬编码。键名见 prompts.yaml 顶部登记表。"""
import os

import yaml

from common import abspath, cfg_get, ConfigError, load_config


def prompts_file(cfg=None):
    cfg = cfg or load_config()
    return abspath(cfg_get(cfg, "prompts.file"))


def load_prompt(key, cfg=None):
    path = prompts_file(cfg)
    if not os.path.isfile(path):
        raise ConfigError("提示词文件缺失: %s" % path)
    with open(path, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    if not isinstance(prompts, dict) or key not in prompts or not prompts[key]:
        raise ConfigError("提示词键缺失或为空: [%s] （统一在 %s 中维护）" % (key, path))
    return prompts[key].strip()
