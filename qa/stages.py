# -*- coding: utf-8 -*-
"""LLM 步骤：S2 意图识别（P1）、记忆压缩（P2）、S3 查询改写（P3）、
Map 合并节点（P45 = 充分性判断 + 作答，单次调用）、Reduce 汇总（P6）。

- 模型输出必须是纯 JSON；解析失败重试（复用 llm.max_retries），仍失败抛异常（不静默兜底）
- temperature=0（沿用 llm 配置）
"""
import json
import time

from common import cfg_get, get_llm_client, llm_chat
from qa_common import fill_prompt, get_qa_cfg, load_prompt, parse_json_strict
import steplog


def _repair_json(cfg, original_system, bad_text, err):
    """格式修复旁路（P0）：把原始 system prompt + 坏输出交给第二个 LLM 重新格式化。

    仅在解析失败时触发（虚线旁路），正常路径不经过；修复失败向上抛，计入重试。
    """
    system = load_prompt(cfg, "P0_json_repair")
    user = ("下面是定义JSON对象Schema和示例的系统提示词:\n\"\"\"\n%s\n\"\"\"\n\n---\n\n"
            "下面是需要你格式化为合法JSON的LLM原始输出（解析错误: %s）：\n\"\"\"\n%s\n\"\"\""
            % (original_system[:6000], err, bad_text))
    client = get_llm_client(cfg)
    raw = llm_chat(cfg, client, system, user)
    return parse_json_strict(raw)


def _call_json(cfg, prompt_name, user_content, validator=None, **vars):
    """结构化输出三级链路：API 级 response_format + 解析失败走修复旁路 + 重试。

    validator(out) 为字段级校验，抛 ValueError/KeyError 视为同样不合格。
    全部用尽后抛异常（不静默兜底）。
    """
    template = load_prompt(cfg, prompt_name)
    system = fill_prompt(template, **vars) if vars else template
    max_retries = int(cfg_get(cfg, "llm.max_retries"))
    client = get_llm_client(cfg)
    last_err = None
    t_all = time.time()
    attempts, repairs = 0, 0
    for attempt in range(1, max_retries + 1):
        attempts = attempt
        raw = llm_chat(cfg, client, system, user_content,
                       response_format={"type": "json_object"})
        try:
            out = parse_json_strict(raw)
            if validator:
                validator(out)
            steplog.llm_call(prompt_name, time.time() - t_all, attempts, repairs)
            return out
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            last_err = e
            try:  # 修复旁路：盲重试对"习惯性围栏输出"无效，先格式化再判
                repairs += 1
                out = _repair_json(cfg, system, raw, e)
                if validator:
                    validator(out)
                steplog.llm_call(prompt_name, time.time() - t_all, attempts, repairs)
                return out
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e2:
                last_err = e2
    raise RuntimeError("%s 结构化输出失败（已重试 %d 次，含修复旁路）: %s"
                       % (prompt_name, max_retries, last_err))


# ---------- S2 意图识别（P1） ----------

def detect_intent(cfg, question, memory_context):
    out = _call_json(cfg, "P1_intent", "【对话记忆】\n%s\n\n【最新问题】\n%s"
                     % (memory_context, question),
                     validator=lambda o: bool(o["is_financial_qa"]) in (True, False))
    return {"is_financial_qa": bool(out["is_financial_qa"]),
            "reason": str(out.get("reason", "")),
            "confidence": float(out.get("confidence", 0.0))}


# ---------- P2 记忆压缩 ----------

def compress_memory(cfg, old_summary, texts):
    user = "【已有摘要】\n%s\n\n【待压缩对话】\n%s" % (old_summary or "（无）", "\n".join(texts))
    template = load_prompt(cfg, "P2_memory_compress")
    client = get_llm_client(cfg)
    t0 = time.time()
    out = llm_chat(cfg, client, template, user).strip()
    steplog.llm_call("P2_memory_compress", time.time() - t0, 1, 0)
    return out


# ---------- S3 查询改写（P3） ----------

def rewrite_query(cfg, question, memory_context, scope):
    user = "【对话记忆】\n%s\n\n【最新问题】\n%s" % (memory_context, question)
    out = _call_json(cfg, "P3_rewrite", user,
                     validator=lambda o: bool(o.get("sub_queries")),
                     company_list="、".join(scope["company_list"]) or "（无）",
                     period_list="、".join(scope["period_list"]) or "（无）")
    return out


# ---------- Map 合并节点：P45 充分性判断 + 作答（单次 LLM 调用） ----------

def answer_with_check(cfg, question, sub_queries, evidence_block, loop, max_loops, scope):
    """P45：一次调用同时产出 充分性判断 + 回答。

    返回 {sufficient, missing_points, additional_queries, reason,
          insufficient, missing_info, answer}。
    """
    def _v(o):
        if "sufficient" not in o:
            raise KeyError("sufficient")
        if "insufficient" not in o or "answer" not in o:
            raise KeyError("insufficient/answer")
        if o["sufficient"] and (o.get("additional_queries") or o.get("missing_points")):
            raise ValueError("sufficient=true 时 additional_queries/missing_points 应为空")
    out = _call_json(cfg, "P45_answer", "请给出判断与作答结果 JSON。",
                     validator=_v,
                     company_list="、".join(scope["company_list"]) or "（无）",
                     period_list="、".join(scope["period_list"]) or "（无）",
                     evidence=evidence_block,
                     question=question,
                     sub_queries=json.dumps(sub_queries, ensure_ascii=False),
                     loop=str(loop),
                     max_loops=str(max_loops))
    out.setdefault("missing_points", [])
    out.setdefault("additional_queries", [])
    out.setdefault("missing_info", [])
    out.setdefault("answer", "")
    return out


# ---------- Reduce：P6 汇总各子任务答案 ----------

def aggregate_answers(cfg, question, task_results):
    """Map 结果 → Reduce 汇总。task_results 元素:
    {task_id, task_query, answer, sources, missing_info}"""
    out = _call_json(cfg, "P6_aggregate", "请给出汇总回答 JSON。",
                     validator=lambda o: "insufficient" in o and "answer" in o,
                     task_answers=json.dumps(task_results, ensure_ascii=False, indent=1),
                     question=question)
    return out
