#!/usr/bin/env python3
"""阿里云 Qwen3.7 Max Batch API 适配器脚本

用于 FormTSR 评测框架的适配器，通过临时 HTTP 服务器托管图片。

环境变量:
  FORMTSR_IMAGE_PATH: 图片路径
  FORMTSR_PROMPT: 提示词
  FORMTSR_HTTP_PORT: HTTP 服务器端口（默认 8765）
  FORMTSR_HTTP_BASE_URL: 图片基础 URL（默认 http://localhost:8765）
  OPENAI_API_KEY: API Key
  OPENAI_BASE_URL: API URL
"""

from __future__ import annotations

import base64
import http.server
import json
import os
import socket
import socketserver
import sys
import threading
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print(json.dumps({"error": "openai library not installed"}), file=sys.stderr)
    sys.exit(1)


def find_free_port(start_port: int = 8765, max_attempts: int = 100) -> int:
    """查找可用端口"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"无法找到可用端口 ({start_port}-{start_port+max_attempts})")


def start_http_server(data_root: Path, port: int) -> tuple[threading.Thread, socketserver.TCPServer]:
    """启动 HTTP 服务器"""

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(data_root), **kwargs)

        def log_message(self, format, *args):
            pass  # 禁用日志

    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("", port), QuietHandler)

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    return thread, httpd


def main() -> None:
    # 读取环境变量
    image_path_str = os.environ.get("FORMTSR_IMAGE_PATH", "")
    prompt = os.environ.get("FORMTSR_PROMPT", "")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    model = os.environ.get("FORMTSR_MODEL", "qwen3.7-max")
    http_port_str = os.environ.get("FORMTSR_HTTP_PORT", "")
    http_base_url = os.environ.get("FORMTSR_HTTP_BASE_URL", "")

    # 验证必需参数
    if not image_path_str or not prompt or not api_key or not base_url:
        error_msg = "Missing required environment variables"
        print(json.dumps({"error": error_msg}), file=sys.stderr)
        sys.exit(1)

    image_path = Path(image_path_str)

    # 确定图片 URL
    if http_base_url:
        # 使用预设的 base URL
        image_url = f"{http_base_url}/{image_path}"
    else:
        # 启动临时 HTTP 服务器
        # 找到图片根目录（FormTSR/datasets）
        parts = list(image_path.parts)
        try:
            formtsr_index = parts.index("FormTSR")
            data_root = Path(*parts[:formtsr_index+2])  # FormTSR/datasets
            relative_path = Path(*parts[formtsr_index:])
        except (ValueError, IndexError):
            # 回退：使用当前目录
            data_root = Path.cwd()
            relative_path = image_path

        # 查找可用端口
        if http_port_str:
            port = int(http_port_str)
        else:
            port = find_free_port()

        # 启动服务器
        _thread, httpd = start_http_server(data_root, port)
        time.sleep(0.5)  # 等待服务器启动

        image_url = f"http://localhost:{port}/{relative_path}"

    # 调用 API
    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"Picture: {image_url}\n{prompt}"
                }
            ],
            temperature=0.0,
            max_tokens=8192,
        )

        raw_response = response.choices[0].message.content

        # 输出结果到 stdout（FormTSR 框架会捕获）
        print(raw_response)

    except Exception as e:
        error_msg = str(e)
        print(json.dumps({"error": error_msg}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
