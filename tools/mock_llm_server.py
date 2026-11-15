"""最小 OpenAI 兼容端点，用于验证策略模型接入而不依赖真实推理服务。

用法：
    python mock_llm_server.py --port 8123 [--reject-json-mode] [--always-fail]

只实现 POST /v1/chat/completions，返回固定 JSON 决策，便于校验：
- 请求体是否带 response_format；
- 端点拒绝 response_format 时是否降级重试；
- 失败是否被包装成 LLMProviderError（不泄露 token）。
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

RESPONSE = {
    "action": "retrain_with_replay",
    "confidence": 0.86,
    "reason": "特征漂移与性能下降同时出现，建议带重放训练",
}


class Handler(BaseHTTPRequestHandler):
    reject_json_mode = False
    always_fail = False
    calls: list[dict] = []

    def log_message(self, *args):  # noqa: D102 - 静默日志
        pass

    def do_POST(self):  # noqa: N802 - 与 BaseHTTPRequestHandler 保持一致
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        Handler.calls.append(body)

        if Handler.always_fail:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": "boom"}}).encode())
            return

        if Handler.reject_json_mode and "response_format" in body:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"error": {"message": "response_format is not supported"}}).encode()
            )
            return

        payload = {
            "id": "mock-1",
            "object": "chat.completion",
            "model": body.get("model", "mock-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(RESPONSE, ensure_ascii=False),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 321,
                "completion_tokens": 48,
                "total_tokens": 369,
            },
        }
        raw = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--reject-json-mode", action="store_true")
    parser.add_argument("--always-fail", action="store_true")
    args = parser.parse_args()
    Handler.reject_json_mode = args.reject_json_mode
    Handler.always_fail = args.always_fail
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"mock endpoint on http://127.0.0.1:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
