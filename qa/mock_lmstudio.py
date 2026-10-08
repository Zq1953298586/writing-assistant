#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模拟 LM Studio（OpenAI 兼容接口）——专属写作助手 QA 专用。
用法: python3 mock_lmstudio.py [端口，默认 1234]
只做一件事：让页面的"直连模型"链路能跑通，返回固定中文文本。
注意：真实模型测试只能在用户台式机上做，这里只验证链路。"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 1234

CANNED = ("【模拟模型输出】链路通畅。这是一段模拟生成的中文文本："
"夜雨敲着铁皮屋顶，床底下那道蓝光又跳了一下，像在呼吸。")


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send({}, 204)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._send({"data": [{"id": "mock-qwen3-writing", "object": "model"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw.decode("utf-8"))
        except Exception:
            req = {}
        msgs = req.get("messages", []) or []
        tail = str(msgs[-1].get("content", ""))[:60] if msgs else ""
        text = CANNED + (("（收到输入前60字：" + tail + "）") if tail else "")
        if self.path.startswith("/v1/chat/completions"):
            self._send({
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
            })
        else:
            self._send({"error": "not found"}, 404)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"mock LM Studio listening on 127.0.0.1:{PORT}", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
