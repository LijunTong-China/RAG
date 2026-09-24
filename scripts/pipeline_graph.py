# -*- coding: utf-8 -*-
"""LangGraph 流水线编排：把步骤1~5 串成状态图，每份财报跑一遍图。

图结构（每份财报一次执行）：

    step1_mineru → step2_render_md → qa_gate（记录抽检状态，不阻塞）→ step3_flatten → step4_chunk → step5_milvus → END

- 每个节点 = 原步骤脚本的共用函数；节点异常直接抛出终止本份（不影响其他份）
- 人工抽检只是记录：qa_records 未填「通过」也会跑完全链，仅汇总提醒补检
- 并行：份与份之间用线程池并发（config.pipeline.max_workers，默认 3）；
  单份内部 step1~6 仍串行。控制台输出加 [report_id] 前缀 + 全局打印锁
- 全量执行：python pipeline_graph.py            （扫描 step0 -> 并发跑图）
- 单份执行：python pipeline_graph.py --report_id xxx
"""
import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from common import (abspath, cfg_get, fail, load_config, logical_report_id,
                    read_manifest)
from step1_mineru import run_reports
from step2_render_md import render_report
from step3_flatten import flatten_report
from step4_chunk import chunk_report
from step5_milvus import index_report
from step6_bm25 import build_bm25


class PipelineState(TypedDict):
    report_id: str
    error: str
    stage: str          # 当前节点名
    qa_passed: bool     # 人工闸门结果


def _node(name, fn):
    def node(state: PipelineState) -> dict:
        fn(load_config(), state["report_id"])   # 异常直接抛出，终止本次图的执行
        return {"stage": name, "error": ""}
    return node


def qa_gate(state: PipelineState) -> dict:
    """人工抽检状态检查：只记录，不做闸门，后续流程不受影响。"""
    from step3_flatten import qa_status
    cfg = load_config()
    status = qa_status(cfg, state["report_id"])
    return {"qa_passed": status == "通过", "stage": "qa_gate", "error": ""}


def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("step1_mineru", _node("step1_mineru", run_one))
    g.add_node("step2_render_md", _node("step2_render_md", render_report))
    g.add_node("qa_gate", qa_gate)
    g.add_node("step3_flatten", _node("step3_flatten", flatten_report))
    g.add_node("step4_chunk", _node("step4_chunk", chunk_report))
    g.add_node("step5_milvus", _node("step5_milvus", index_report))
    g.add_node("step6_bm25", _node("step6_bm25", build_bm25))

    g.set_entry_point("step1_mineru")
    g.add_edge("step1_mineru", "step2_render_md")
    g.add_edge("step2_render_md", "qa_gate")
    g.add_edge("qa_gate", "step3_flatten")   # 抽检状态只记录，始终放行
    g.add_edge("step3_flatten", "step4_chunk")
    g.add_edge("step4_chunk", "step5_milvus")
    g.add_edge("step5_milvus", "step6_bm25")
    g.add_edge("step6_bm25", END)
    return g.compile()


def run_one(cfg, rid):
    """graph 的 step1 节点用：单份（含拆卷组）走 run_reports。"""
    ok_status = {cfg_get(cfg, "manifest_status.ok"),
                 cfg_get(cfg, "manifest_status.manual_filled"),
                 cfg_get(cfg, "manifest_status.llm_filled")}
    rows = read_manifest(cfg)
    rows = [r for r in rows
            if logical_report_id(os.path.splitext(r["file_name"])[0]) == rid]
    if not rows:
        raise RuntimeError("对照表中无此 report_id: %s" % rid)
    bad = [r["file_name"] for r in rows if r["parse_status"] not in ok_status]
    if bad:
        raise RuntimeError("拆卷未放行: %s" % bad)
    succeeded, failed = run_reports(cfg, rows)
    if failed or not succeeded:
        raise RuntimeError("MinerU 解析失败: %s" % (failed or "无产物"))
    # 检查本逻辑报告的合并产物（单卷 = 本身）
    content = os.path.join(abspath(cfg_get(cfg, "paths.mineru_json")), rid,
                           "%s_content_list.json" % rid)
    if not os.path.isfile(content):
        raise RuntimeError("解析产物缺失: %s" % content)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_id", help="只跑这一份；不传 = 扫描后跑全部放行财报")
    args = ap.parse_args()

    # 步骤0：扫描登记（幂等，重跑生成新对照表；单份模式跳过避免误清工作副本）
    if not args.report_id:
        import step0_scan
        step0_scan.main()

    rows = read_manifest(cfg)
    ok_status = {cfg_get(cfg, "manifest_status.ok"),
                 cfg_get(cfg, "manifest_status.manual_filled"),
                 cfg_get(cfg, "manifest_status.llm_filled")}
    if args.report_id:
        ids = [args.report_id]
        if not any(logical_report_id(os.path.splitext(r["file_name"])[0]) == args.report_id
                   for r in rows):
            fail("对照表中无此 report_id: %s" % args.report_id)
    else:
        # 逻辑报告 id 去重（拆卷多行 -> 一个 base id）
        ids = sorted({logical_report_id(os.path.splitext(r["file_name"])[0])
                      for r in rows if r["parse_status"] in ok_status})
    if not ids:
        fail("无放行财报可处理")

    max_workers = int(cfg_get(cfg, "pipeline.max_workers"))  # 严格模式：缺失即报错
    graph = build_graph()
    done, need_qa, errors = [], [], []
    lock = threading.Lock()

    def _print(msg, err=False):
        with lock:
            print(msg, file=sys.stderr if err else sys.stdout, flush=True)

    def _run(rid):
        try:
            state = graph.invoke({"report_id": rid, "error": "", "stage": "", "qa_passed": False})
        except Exception as e:  # 节点异常终止本份，不影响其他份
            _print("[%s] 失败: %s" % (rid, e), err=True)
            return rid, None
        tag = "完成（人工抽检未完成，qa_records 可随时补检）" if not state["qa_passed"] \
            else "完成（全链入库，抽检已通过）"
        _print("[%s] %s" % (rid, tag))
        return rid, state["qa_passed"]

    print("[编排] %d 份财报，并发 %d（config.pipeline.max_workers）" % (len(ids), max_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for rid, qa_passed in pool.map(_run, ids):
            if qa_passed is None:
                errors.append(rid)
            else:
                done.append(rid)
                if not qa_passed:
                    need_qa.append(rid)

    print("\n汇总：完成 %d / 失败 %d" % (len(done), len(errors)))
    for rid in errors:
        print("  失败: %s" % rid)
    if need_qa:
        print("  待人工抽检（不阻塞，建议补检）: " + ", ".join(need_qa))


if __name__ == "__main__":
    main()
