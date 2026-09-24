# -*- coding: utf-8 -*-
"""检索问答阶段（在线 QA 服务）独立包。

与数据处理管线（scripts/step0~step6）严格分离：
- 本包只复用 scripts/common.py 的通用工具（配置/日志/token 计数/LLM 客户端）
  以及 step5/step6 暴露的向量检索与 BM25 检索原语；
- 数据处理侧的任何改动不进本包，本包的逻辑也不写入 scripts/。
"""
