"""组合权重：α 倾斜优化（B1 的正解）。

    max  αᵀw − λ·wᵀΣw     s.t.  Σw = 1,  0 ≤ w ≤ max_weight

为什么需要它：hub 的 `portfolio_optimize` 只吃 klines（→ 协方差），**没有期望收益入参**，
所以只能做纯风险配置（HRP）。结果是"权重与选择序无关"——实测权重第 1 的票选择序第 20。
本模块把 alpha 真正放进目标函数，让仓位反映"我多看好"，同时保留 §8 的单票上限。

α 的量纲问题：alpha 是 20 日反转（0.2~0.4），而 Σ 是年化协方差（1e-2~1e-1 量级），
直接相加 λ 没有意义。所以先把 alpha **做截面 z-score**，λ 就成了可解释的风险厌恶系数
（λ 越大越靠风险平价，越接近等权/HRP）。这一点写在这里，免得日后调参靠猜。
"""

import logging

import numpy as np

log = logging.getLogger("scanner.optimize")

TRADING_DAYS = 252


def _cov(returns, symbols):
    """年化协方差矩阵（按 symbols 顺序）。样本不足时返回 None。"""
    import pandas as pd  # noqa: PLC0415

    cols = [s for s in symbols if s in returns.columns]
    if len(cols) < 2:
        return None, []
    r = returns[cols].dropna(how="any")
    if len(r) < 20:
        return None, []
    return r.cov().to_numpy(dtype=float) * TRADING_DAYS, cols


def alpha_tilted_weights(alpha: dict, returns, lam: float = 2.0,
                         max_weight: float = 0.12, fallback: str = "equal") -> dict:
    """→ {symbol: weight}，满足 Σw=1、0 ≤ w ≤ max_weight。

    lam 越大越偏风险平价；lam→0 则集中到 alpha 最高的（受 max_weight 封顶）。
    不可行（max_weight × n < 1）或样本不足时走 fallback，并把原因写日志——
    **不静默给一个坏权重**（铁律 3）。
    """
    syms = list(alpha.keys())
    n = len(syms)
    if n == 0:
        return {}
    if n == 1:
        return {syms[0]: 1.0}
    if max_weight * n < 1.0 - 1e-9:
        log.warning("单票上限 %.2f × %d 只 < 100%%，约束不可行；fallback=%s",
                    max_weight, n, fallback)
        return _fallback(syms, fallback)
    cov, cols = _cov(returns, syms)
    if cov is None:
        log.warning("收益样本不足（需 ≥2 标的、≥20 日）→ fallback=%s", fallback)
        return _fallback(syms, fallback)

    use = list(cols)
    a = np.array([float(alpha[s]) for s in use], dtype=float)
    a = (a - a.mean()) / (a.std() + 1e-12)          # 截面 z-score，让 λ 可解释

    def neg_util(w):
        return -(a @ w - lam * float(w @ cov @ w))

    def grad(w):
        return -(a - 2.0 * lam * (cov @ w))

    try:
        from scipy.optimize import minimize  # noqa: PLC0415
    except ImportError:
        log.warning("scipy 不可用 → fallback=%s", fallback)
        return _fallback(syms, fallback)

    w0 = np.full(len(use), 1.0 / len(use))
    res = minimize(neg_util, w0, jac=grad, method="SLSQP",
                   bounds=[(0.0, max_weight)] * len(use),
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                                 "jac": lambda w: np.ones_like(w)}],
                   options={"maxiter": 500, "ftol": 1e-12})
    if not res.success:
        log.warning("α 倾斜优化未收敛（%s）→ fallback=%s", res.message, fallback)
        return _fallback(syms, fallback)
    w = np.clip(res.x, 0.0, max_weight)
    tot = w.sum()
    if tot <= 0:
        return _fallback(syms, fallback)
    w = w / tot                                        # 归一（clip 后可能有 1e-12 级偏差）
    out = {s: float(x) for s, x in zip(use, w)}
    for s in syms:                                     # 未进协方差样本的票，权重记 0 而非丢键
        out.setdefault(s, 0.0)
    return out


def _fallback(syms, kind: str) -> dict:
    if kind == "zero":
        return {s: 0.0 for s in syms}
    return {s: 1.0 / len(syms) for s in syms}          # equal
