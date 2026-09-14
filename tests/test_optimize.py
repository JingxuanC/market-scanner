import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scanner import optimize  # noqa: E402


def _returns(syms, days=140, seed=0):
    rnd = np.random.default_rng(seed)
    return pd.DataFrame(rnd.normal(0.0005, 0.02, size=(days, len(syms))), columns=syms)


def test_weights_sum_to_one_and_respect_cap():
    syms = [f"sh60000{i}" for i in range(1, 13)]      # 12 只 × 12% = 144% 可行
    alpha = {s: 1.0 - i * 0.05 for i, s in enumerate(syms)}
    w = optimize.alpha_tilted_weights(alpha, _returns(syms), lam=2.0, max_weight=0.12)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert all(0.0 <= v <= 0.12 + 1e-9 for v in w.values())


def test_high_alpha_gets_more_than_equal_weight():
    syms = [f"sh60000{i}" for i in range(1, 13)]
    alpha = {s: 1.0 - i * 0.05 for i, s in enumerate(syms)}
    w = optimize.alpha_tilted_weights(alpha, _returns(syms), lam=0.5, max_weight=0.12)
    top = syms[0]                                     # alpha 最高
    assert w[top] >= 1.0 / len(syms) - 1e-9, "α 最高的票权重不该低于等权"
    assert w[top] == pytest.approx(max(w.values()), abs=1e-6)  # 多只并列在 cap 上


def test_large_lambda_moves_toward_equal():
    syms = [f"sh60000{i}" for i in range(1, 9)]
    alpha = {s: 1.0 - i * 0.05 for i, s in enumerate(syms)}
    ret = _returns(syms)
    w_tilt = optimize.alpha_tilted_weights(alpha, ret, lam=0.05, max_weight=0.6)
    w_risk = optimize.alpha_tilted_weights(alpha, ret, lam=500.0, max_weight=0.6)
    spread_tilt = max(w_tilt.values()) - min(w_tilt.values())
    spread_risk = max(w_risk.values()) - min(w_risk.values())
    assert spread_risk < spread_tilt, "λ 很大时应更接近等权"


def test_infeasible_cap_falls_back_and_logs(caplog):
    syms = [f"sh60000{i}" for i in range(1, 7)]       # 6 × 12% = 72% < 100%
    alpha = {s: 1.0 for s in syms}
    with caplog.at_level("WARNING"):
        w = optimize.alpha_tilted_weights(alpha, _returns(syms), max_weight=0.12)
    assert abs(sum(w.values()) - 1.0) < 1e-9          # equal fallback
    assert any("不可行" in r.message for r in caplog.records), "必须显式记账"


def test_insufficient_samples_falls_back(caplog):
    syms = ["sh600001", "sh600002"]
    with caplog.at_level("WARNING"):
        w = optimize.alpha_tilted_weights({s: 1.0 for s in syms},
                                          _returns(syms, days=5), max_weight=0.6)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert any("样本不足" in r.message for r in caplog.records)
