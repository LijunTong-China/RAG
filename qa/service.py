# -*- coding: utf-8 -*-
"""Web 问答服务入口（在线 QA 阶段，与数据处理管线分离）。

启动：python -m uvicorn qa.service:app --host 127.0.0.1 --port 8800
      （或直接 python qa/service.py）
启动自检不通过（Milvus/BM25 索引缺失）直接拒绝启动。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

from common import fail, load_config
from pipeline import QAService  # noqa: E402


def _create_app():
    cfg = load_config()
    svc = QAService(cfg)
    svc.startup_check()  # 语料库为空/服务不可用 = 拒绝启动（不兜底）
    svc.store.start_sweeper()

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from pydantic import BaseModel

    webapp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "webapp", "index.html")

    class ChatIn(BaseModel):
        session_id: str
        message: str

    class CloseIn(BaseModel):
        session_id: str

    app = FastAPI(title="财报 RAG 问答服务")

    @app.get("/")
    def index():
        if not os.path.isfile(webapp):
            raise HTTPException(500, "前端页面缺失: webapp/index.html")
        return FileResponse(webapp)

    @app.get("/favicon.ico")
    def favicon():
        # 内联 SVG 图标，避免浏览器默认请求 404
        from fastapi.responses import Response
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
               '<rect width="32" height="32" rx="6" fill="#1a5fb4"/>'
               '<text x="16" y="23" font-size="18" text-anchor="middle" '
               'fill="#fff" font-family="serif">财</text></svg>')
        return Response(svg, media_type="image/svg+xml")

    @app.post("/api/session/new")
    def session_new():
        s = svc.store.new_session()
        return {"session_id": s.session_id}

    @app.post("/api/chat")
    def chat(body: ChatIn):
        s = svc.store.get(body.session_id)
        if s is None:
            raise HTTPException(404, "会话不存在或已过期，请新建会话")
        with s.lock:  # 同会话串行，避免记忆压缩与问答交错
            return svc.answer(s, body.message)

    @app.post("/api/chat/stream")
    def chat_stream(body: ChatIn):
        """SSE 流式问答：实时推送 流程进度(stage_*) / thinking 增量(delta) /
        行内提示(note) / LLM 耗时(llm)，结束时推 done（完整结果）或 error。"""
        import json as _json
        import queue as _queue
        import threading as _threading
        from fastapi.responses import StreamingResponse
        import steplog

        s = svc.store.get(body.session_id)
        if s is None:
            raise HTTPException(404, "会话不存在或已过期，请新建会话")
        q = _queue.Queue()

        def _emitter(ev):
            q.put(ev)

        def _run():
            try:
                steplog.set_emitter(_emitter)
                with s.lock:  # 同会话串行，与 /api/chat 一致
                    result = svc.answer(s, body.message)
                q.put({"type": "done", "reply": result["reply"],
                       "sources": result["sources"],
                       "insufficient": result["insufficient"],
                       "elapsed_ms": result["elapsed_ms"]})
            except Exception as e:
                q.put({"type": "error", "message": str(e)})
            finally:
                steplog.set_emitter(None)
                q.put(None)  # 结束哨兵

        _threading.Thread(target=_run, daemon=True).start()

        def _gen():
            while True:
                ev = q.get()
                if ev is None:
                    break
                yield "data: " + _json.dumps(ev, ensure_ascii=False) + "\n\n"

        return StreamingResponse(_gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.post("/api/session/close")
    def session_close(body: CloseIn):
        svc.store.close(body.session_id)
        return {"closed": True}

    return app


app = _create_app()

if __name__ == "__main__":
    import uvicorn
    from qa_common import get_qa_cfg
    cfg = load_config()
    qa_cfg = get_qa_cfg(cfg)
    uvicorn.run(app, host=qa_cfg["host"], port=int(qa_cfg["port"]))
