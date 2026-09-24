# -*- coding: utf-8 -*-
"""S1 会话记忆管理：纯内存态，不落盘。

- 容量计量：tiktoken o200k_base（复用数据处理侧 count_tokens）
- 触发压缩：total_tokens > max_memory_tokens；压缩保证保留最近 keep_recent_turns 轮原文
- 进入 LLM 的形态：summary + 最近 4 轮原文
"""
import threading
import time
import uuid
from datetime import datetime, timedelta

from common import cfg_get, count_tokens
from qa_common import get_qa_cfg


class SessionMemory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.last_active_at = datetime.now()
        self.summary = ""
        self.messages = []  # {message_id, turn_no, role, content, tokens}
        self.total_tokens = 0
        self.lock = threading.Lock()  # 同会话串行处理，避免压缩与问答交错

    def next_turn_no(self):
        return (self.messages[-1]["turn_no"] + 1 if self.messages else 1)

    def add_message(self, turn_no, role, content):
        tokens = count_tokens(self.cfg, content)
        msg = {"message_id": str(uuid.uuid4()), "turn_no": turn_no,
               "role": role, "content": content, "tokens": tokens}
        self.messages.append(msg)
        self.total_tokens += tokens
        self.last_active_at = datetime.now()
        return msg

    def recent_messages(self):
        keep = int(cfg_get(self.cfg, "qa_service.memory.keep_recent_turns"))
        if not self.messages:
            return []
        last_turn = self.messages[-1]["turn_no"]
        cutoff = last_turn - keep + 1
        return [m for m in self.messages if m["turn_no"] >= cutoff]

    def context_for_llm(self):
        """summary + 最近 keep_recent_turns 轮原文，供 P1/P3 使用。"""
        lines = []
        if self.summary:
            lines.append("【对话摘要】\n" + self.summary)
        for m in self.recent_messages():
            lines.append("[%s] %s" % (m["role"], m["content"]))
        return "\n".join(lines) if lines else "（无历史对话）"

    def compress(self, compress_fn):
        """滚动压缩：压缩最旧轮次（保留最近 keep_recent_turns 轮），摘要本身超限再压摘要。

        compress_fn(summary, texts) -> new_summary 由调用方注入（LLM P2），便于测试替换。
        """
        mcfg = get_qa_cfg(self.cfg)["memory"]
        max_tokens = int(mcfg["max_memory_tokens"])
        if self.total_tokens <= max_tokens:
            return False
        recent = self.recent_messages()
        recent_ids = {id(m) for m in recent}
        to_compress = [m for m in self.messages if id(m) not in recent_ids]
        if to_compress:
            texts = ["[%s] %s" % (m["role"], m["content"]) for m in to_compress]
            new_summary = compress_fn(self.summary, texts)
            summary_max = int(mcfg["summary_max_chars"])
            if len(new_summary) > summary_max:  # 摘要超限：对摘要再压缩（不调 LLM，硬截）
                new_summary = new_summary[:summary_max]
            removed = sum(m["tokens"] for m in to_compress)
            self.summary = new_summary
            self.messages = [m for m in self.messages if id(m) in recent_ids]
            # 口径：tokens(summary) + sum(messages.tokens)
            self.total_tokens = (count_tokens(self.cfg, self.summary)
                                 + sum(m["tokens"] for m in self.messages))
            if to_compress and self.total_tokens > max_tokens and recent:
                # 最近轮原文仍超限（极少见）：只保留摘要
                self.messages = []
                self.total_tokens = count_tokens(self.cfg, self.summary)
        return True


class SessionStore:
    """会话容器：创建/定位/销毁 + 空闲超时扫描。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self._sessions = {}
        self._sweeper = None

    def new_session(self):
        s = SessionMemory(self.cfg)
        self._sessions[s.session_id] = s
        return s

    def get(self, session_id):
        return self._sessions.get(session_id)

    def close(self, session_id):
        self._sessions.pop(session_id, None)

    def sweep_idle(self):
        timeout = timedelta(seconds=int(get_qa_cfg(self.cfg)["session_idle_timeout_seconds"]))
        now = datetime.now()
        dead = [sid for sid, s in self._sessions.items()
                if now - s.last_active_at > timeout]
        for sid in dead:
            self.close(sid)
        return dead

    def start_sweeper(self, interval_seconds=60):
        """后台空闲扫描线程（daemon，随服务退出）。"""
        def _run():
            while True:
                time.sleep(interval_seconds)
                self.sweep_idle()
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        self._sweeper = t
        return t
