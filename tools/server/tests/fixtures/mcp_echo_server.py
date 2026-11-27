#!/usr/bin/env python3
"""
Minimal MCP server for testing.
Implements JSON-RPC 2.0 over stdio (line-delimited JSON).
"""
import json
import sys
import os

# Ensure we use python3 from the current environment
if sys.platform == "win32":
    # On Windows, we need to use the same python interpreter
    pass

TOOLS = [
    {
        "name": "echo",
        "description": "Echo back the input message",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Message to echo"}
            },
            "required": ["message"]
        }
    },
    {
        "name": "add",
        "description": "Add two numbers",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }
    },
    {
        "name": "fail_once",
        "description": "Fails on first call, succeeds on subsequent calls",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
]

_state = {"fail_once_called": False}

def handle_initialize(params, req_id):
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "echo-test", "version": "1.0"}
        }
    }

def handle_tools_list(params, req_id):
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {"tools": TOOLS}
    }

def handle_tools_call(params, req_id):
    tool_name = params.get("name")
    arguments = params.get("arguments", {})

    if tool_name == "echo":
        message = arguments.get("message", "")
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": f"echo: {message}"}]
            }
        }
    elif tool_name == "add":
        a = arguments.get("a", 0)
        b = arguments.get("b", 0)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": str(a + b)}]
            }
        }
    elif tool_name == "fail_once":
        if not _state["fail_once_called"]:
            _state["fail_once_called"] = True
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": "transient error"}
            }
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": "ok"}]
            }
        }
    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32602, "message": f"Unknown tool: {tool_name}"}
        }

def handle_ping(params, req_id):
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {}
    }

HANDLERS = {
    "initialize": handle_initialize,
    "tools/list": handle_tools_list,
    "tools/call": handle_tools_call,
    "ping": handle_ping,
}

def read_lines():
    if sys.platform != "win32" or "--poll-stdin" not in sys.argv:
        yield from sys.stdin
        return

    # A pending blocking ReadFile can let an oversized PIPE_NOWAIT write make
    # progress. Poll instead, so the transport must fit writes in the pipe buffer.
    import ctypes
    import msvcrt
    import time
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    peek = kernel32.PeekNamedPipe
    peek.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                     ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    peek.restype = wintypes.BOOL
    fd = sys.stdin.fileno()
    msvcrt.setmode(fd, os.O_BINARY)
    handle = msvcrt.get_osfhandle(fd)
    pending = b""
    while True:
        available = wintypes.DWORD()
        if not peek(handle, None, 0, None, ctypes.byref(available), None):
            if ctypes.get_last_error() == 109:  # ERROR_BROKEN_PIPE
                return
            raise ctypes.WinError(ctypes.get_last_error())
        if available.value == 0:
            time.sleep(0.001)
            continue
        chunk = os.read(fd, min(available.value, 4096))
        if not chunk:
            return
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            yield line.decode("utf-8")


def main():
    # Use unbuffered output
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

    for line in read_lines():
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = request.get("method")
        req_id = request.get("id")
        params = request.get("params", {})

        # JSON-RPC 2.0: a message without an id is a notification and must not receive a response
        if req_id is None:
            continue

        handler = HANDLERS.get(method)
        if handler:
            response = handler(params, req_id)
        else:
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            }

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
