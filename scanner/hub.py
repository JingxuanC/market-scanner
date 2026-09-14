"""hub MCP 调用层 —— 走「直连服务端口 + X-License-Key」，与 deploy/cron_tasks.sh 同一条路。

为什么不用 mcphub 路由：mcphub 要 bearer key，而配额层当前把 key 禁了；cron 一直
用的是直连端口这条已验证的路，扫描器沿用，避免把选股绑在配额层状态上。

工具返回值是 **JSON 字符串**（在 MCP 的 result.content[0].text 里），这里解析成 dict。
`isError=true` 一律抛 HubError —— 铁律 3：不静默降级。
"""

import json
import os
import urllib.request

SERVICES = {
    "astock-data": ("127.0.0.1", 50052),
    "factor-miner": ("127.0.0.1", 50053),
}


class HubError(RuntimeError):
    """hub 工具返回错误 / 不可达。调用方必须显式处理，不允许吞掉。"""


def call_tool(tool: str, arguments: dict = None, service: str = None,
              timeout: int = 900, base_url: str = None, license_key: str = None) -> dict:
    """调 hub 工具，返回解析后的 dict。service 省略时按工具名前缀猜。"""
    if base_url is None:
        svc = service or _guess_service(tool)
        host, port = SERVICES[svc]
        base_url = "http://%s:%d/mcp" % (host, port)
    key = license_key or os.environ.get("MCP_LICENSE_KEY", "")
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": tool, "arguments": arguments or {}}}
    req = urllib.request.Request(base_url, data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json, text/event-stream")
    if key:
        req.add_header("X-License-Key", key)
    try:
        raw = urllib.request.urlopen(req, timeout=timeout).read().decode()
    except Exception as e:  # noqa: BLE001
        raise HubError("调用 %s 失败: %s: %s" % (tool, type(e).__name__, e)) from e
    for line in raw.splitlines():          # SSE 包装
        if line.startswith("data: "):
            raw = line[6:]
            break
    try:
        resp = json.loads(raw)
    except ValueError as e:
        raise HubError("%s 返回非 JSON: %s" % (tool, raw[:200])) from e
    if "error" in resp:
        raise HubError("%s JSON-RPC error: %s" % (tool, resp["error"]))
    result = resp.get("result") or {}
    text = ""
    for c in (result.get("content") or []):
        if c.get("type") == "text":
            text = c.get("text", "")
            break
    if result.get("isError"):
        raise HubError("%s 返回错误: %s" % (tool, text[:300]))
    try:
        return json.loads(text)
    except ValueError as e:
        raise HubError("%s 的 payload 不是 JSON: %s" % (tool, text[:200])) from e


def _guess_service(tool: str) -> str:
    if tool.startswith("get_a_"):
        return "astock-data"
    return "factor-miner"


# ── 选股用到的几个工具（薄封装，参数名对齐 hub 真实 schema）──

def trade_universe(exclude_new_days: int = 60, min_amount_wan: float = 5000.0,
                   include_bj: bool = False, **kw) -> dict:
    return call_tool("get_a_trade_universe", dict(
        exclude_new_days=exclude_new_days, min_amount_wan=min_amount_wan,
        include_bj=include_bj, **kw), timeout=180)


def ml_predict(klines_list: list, model: str = "lgbm", **kw) -> dict:
    return call_tool("ml_predict", {"klines_list": klines_list, "model": model},
                     timeout=1800, **kw)


def portfolio_optimize(symbols: list, klines: dict, method: str = "hrp",
                       lookback: int = 120, **kw) -> dict:
    return call_tool("portfolio_optimize",
                     {"symbols": symbols, "klines": klines,
                      "method": method, "lookback": lookback}, timeout=600, **kw)
