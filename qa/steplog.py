# -*- coding: utf-8 -*-
"""问答过程步骤日志（控制台显眼输出，实时 flush + 耗时统计落盘）。

每一步打印：
╔══════════════════════════════════════════════════════════════
║ ★★★ 步骤[2] S3 任务拆解（P3）
╚══════════════════════════════════════════════════════════════
  【输入】
    ...
  【输出】(耗时 2.3s)
    ...
  ★ 步骤完成 ★

长输出（证据块/整份回答）自动截断到 limit 字符，只保留头部。

耗时统计：step()/done() 记录每步秒数到内存；LLM 调用由 stages 上报；
answer() 结束时 dump(cfg) 追加一条到 data/logs/qa_timing.json（供后续优化分析），
并打印本轮流量的耗时汇总表。多线程并行阶段（Map 子任务、子查询检索）步骤交错，
单步耗时仍准确，但各步之间时间有重叠，总耗时 ≠ 各步之和。
"""
import datetime
import json
import os
import sys
import threading
import time

_WIDTH = 62
_no = 0
_steps = []       # [{no, title, seconds}]
_llm_calls = []   # [{prompt, seconds, attempts, repairs}]
_tls = threading.local()  # 每线程当前步骤标题（Map 子任务并行时互不串号）


def reset():
    """每轮问答开始时清零步骤计数与耗时记录。"""
    global _no
    _no = 0
    _steps.clear()
    _llm_calls.clear()


def _pretty(obj, limit=900):
    if isinstance(obj, str):
        s = obj
    else:
        try:
            s = json.dumps(obj, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            s = str(obj)
    s = s.replace("\r", "")
    if len(s) > limit:
        s = s[:limit] + "\n…（已截断，原文共 %d 字符）" % len(s)
    return "\n".join("  " + l for l in s.split("\n"))


def step(title, inputs=None):
    """标记新步骤开始；返回计时起点 t0（配 done(t0, output) 使用）。"""
    global _no
    _no += 1
    _tls.no, _tls.title = _no, title
    print("\n╔" + "═" * _WIDTH)
    print("║ ★★★ 步骤[%d] %s" % (_no, title))
    print("╚" + "═" * _WIDTH)
    if inputs is not None:
        print("  【输入】")
        print(_pretty(inputs))
    sys.stdout.flush()
    return time.time()


def done(t0, output=None):
    """步骤结束：打印耗时与输出，并把耗时记入本轮汇总。"""
    secs = time.time() - t0
    if getattr(_tls, "title", None):
        _steps.append({"no": _tls.no, "title": _tls.title, "seconds": round(secs, 3)})
        _tls.title = None
    print("  【输出】(耗时 %.1fs)" % secs)
    if output is not None:
        print(_pretty(output))
    print("  ★ 步骤完成 ★")
    sys.stdout.flush()


def note(text):
    """步骤内的行内提示（不占步骤号）。"""
    print("  · " + text)
    sys.stdout.flush()


def llm_call(prompt, seconds, attempts, repairs):
    """上报一次结构化 LLM 调用的耗时（stages._call_json 调用）。"""
    _llm_calls.append({"prompt": prompt, "seconds": round(seconds, 3),
                       "attempts": attempts, "repairs": repairs})
    print("  ⏱ LLM %s 耗时 %.1fs（尝试 %d 次，修复旁路 %d 次）"
          % (prompt, seconds, attempts, repairs))
    sys.stdout.flush()


def summary():
    """本轮各步骤耗时（供 log_turn 一并写入 qa_turns.json）。"""
    return {"steps": list(_steps), "llm_calls": list(_llm_calls)}


def dump(cfg, session_id, turn_no, total_seconds, question=""):
    """本轮结束：追加耗时汇总到 data/logs/qa_timing.json 并打印汇总表。"""
    from common import abspath, cfg_get, ensure_dir
    record = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id, "turn_no": turn_no, "question": question[:200],
        "total_seconds": round(total_seconds, 3),
        "steps": list(_steps), "llm_calls": list(_llm_calls),
    }
    log_dir = ensure_dir(abspath(cfg_get(cfg, "paths.logs")))
    path = os.path.join(log_dir, "qa_timing.json")
    history = []
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)
    history.append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # 汇总表（按耗时降序）
    rows = sorted(_steps, key=lambda s: -s["seconds"])
    print("\n┌" + "─" * _WIDTH)
    print("│ ⏱ 本轮耗时汇总（qa_timing.json 已保存，按耗时降序；并行步骤有重叠）")
    print("├" + "─" * _WIDTH)
    for s in rows:
        print("│ %7.2fs  步骤[%s] %s" % (s["seconds"], s["no"], s["title"]))
    for c in sorted(_llm_calls, key=lambda c: -c["seconds"]):
        print("│ %7.2fs  LLM %s（尝试%d/修复%d）"
              % (c["seconds"], c["prompt"], c["attempts"], c["repairs"]))
    print("│ %7.2fs  —— 总耗时（含并行重叠，不等于各行之和）" % total_seconds)
    print("└" + "─" * _WIDTH)
    sys.stdout.flush()
    return path
