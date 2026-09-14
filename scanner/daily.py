"""20:10 日频轨：可交易域 → h5 截面 → 粗筛 → ML 打分 → 组合优化 → candidate_pool。

设计依据：DESIGN §4.1（可交易域）/§4.4（组合优化出**权重**而不是 TopN 等权）/§4.5（落库）。

三个刻意的工程决定，都写在这里以便复审：
1. **as-of 截面必须完整**：修复期 h5 最新日可能只有 1091 行（全市场约 5200），
   拿它选股等于把 4/5 的票排除在外。所以取「最近一个截面行数 ≥ min_symbols 的交易日」，
   而不是无脑用 `max(datetime)`。
2. **粗筛用 20 日反转，定义与 hub 的 reversal20 完全一致**（`-close.pct_change(20)`）。
   之所以本地算而不是读 Redis `dfactor:`：修复期那批键只覆盖按代码序的前 1091 只
   （有偏子集），拿它当全市场筛子会系统性偏向 000xxx/600xxx。全量落库后可直接切回
   `dfactor:`（本模块留了 `alpha_source` 开关）。
3. **ML 缺失要显式记账**，不静默降级：ml_predict 失败时 ml_score 置空、reason 里写明
   `ml_unavailable`、summary.warnings 里列出。
"""

import json
import logging
import time
from pathlib import Path

from scanner import hub, optimize, store

log = logging.getLogger("scanner.daily")


def apply_weight_cap(weights: dict, max_weight: float) -> "tuple[dict, float]":
    """把单票权重压到 max_weight 以下且总和仍为 1 → (weights, capped_count)。

    **必须用 water-filling**：先封顶超限的，再把余量只分给未封顶的那些。
    写成"截断→整体归一"会永远收敛不了（归一又把被封顶的推过线）——2026-09-14 踩过。
    若 max_weight × n < 1，约束本身不可行，这里**不静默**：返回原权重并把 capped 置 -1，
    由调用方显式记账。
    """
    n = len(weights)
    if n == 0:
        return {}, 0
    if max_weight * n < 1.0 - 1e-9:
        return dict(weights), -1
    raw = {k: max(float(v), 0.0) for k, v in weights.items()}
    total = sum(raw.values()) or 1.0
    raw = {k: v / total for k, v in raw.items()}
    out, active, budget, capped = {}, list(raw), 1.0, 0
    while active:
        s = sum(raw[k] for k in active) or 1.0
        scaled = {k: raw[k] * budget / s for k in active}
        over = [k for k, v in scaled.items() if v > max_weight + 1e-12]
        if not over:
            out.update(scaled)
            break
        for k in over:
            out[k] = max_weight
            budget -= max_weight
            capped += 1
            active.remove(k)
    return out, capped


def read_h5(h5_path) -> "object":
    import pandas as pd

    return pd.read_hdf(str(h5_path), key="data")


def pick_as_of(df, min_symbols: int = 3000):
    """最近一个「截面完整」的交易日（截面行数 ≥ min_symbols）。"""
    sizes = df.groupby(level="datetime").size()
    ok = sizes[sizes >= int(min_symbols)]
    if ok.empty:
        raise ValueError("没有截面行数 ≥ %d 的交易日（h5 可能损坏）" % min_symbols)
    return ok.index[-1]


def close_frame(df, as_of, lookback: int = 120):
    """as_of 往回 lookback 个交易日的收盘价宽表（行=日期，列=instrument 大写）。"""
    import pandas as pd

    dates = df.index.get_level_values("datetime")
    start = as_of - pd.Timedelta(days=int(lookback * 1.9) + 40)
    sub = df[(dates > start) & (dates <= as_of)]
    px = sub["$close"].unstack("instrument")
    return px.dropna(how="all").tail(int(lookback) + 1)


def bars_of(sub, symbol: str) -> list:
    """把 h5 的某只票切成 hub 要的 bar 列表 [{date,open,high,low,close,volume}, ...]。"""
    try:
        one = sub.xs(symbol, level="instrument")
    except KeyError:
        return []
    out = []
    for d, r in one.iterrows():
        c = r.get("$close")
        if c != c or c is None:  # NaN
            continue
        out.append({"date": d.strftime("%Y-%m-%d"), "open": float(r.get("$open", c)),
                    "high": float(r.get("$high", c)), "low": float(r.get("$low", c)),
                    "close": float(c), "volume": float(r.get("$volume", 0) or 0)})
    return out


def run(h5_path=None, db_path=None, as_of=None, shortlist: int = 200, top_n: int = 40,
        lookback: int = 120, min_symbols: int = 3000, min_amount_wan: float = 5000.0,
        use_ml: bool = True, method: str = "hrp", hub_mod=hub, alpha_source: str = "local",
        ml_max_symbols: int = 120, df=None, min_universe: int = 500,
        max_weight: float = 0.12, industry_cap: float = 0.25,
        ranking: str = "alpha", weighting: str = "hrp", lam: float = 2.0) -> dict:
    """跑一次日频轨，返回 summary（并落 candidate_pool）。"""
    t0 = time.time()
    warn = []

    # 1) 可交易域（hub）
    uni = hub_mod.trade_universe(min_amount_wan=min_amount_wan)
    if isinstance(uni, dict) and uni.get("error"):
        raise hub.HubError("可交易域获取失败: %s" % uni)
    uni_rows = uni.get("universe") or []
    if len(uni_rows) < int(min_universe):
        raise hub.HubError("可交易域只有 %d 只 (< %d)，判定数据源退化，拒绝用它选股"
                           % (len(uni_rows), min_universe))
    by_sym = {r["symbol"].upper(): r for r in uni_rows}

    # 2) h5 + as-of 截面（df 可注入：测试不必依赖 pytables，服务器上也能只读一次复用）
    if df is None:
        if h5_path is None:
            raise ValueError("必须给 h5_path 或 df")
        df = read_h5(h5_path)
    as_of = as_of or pick_as_of(df, min_symbols)
    px = close_frame(df, as_of, lookback)
    dates = df.index.get_level_values("datetime")
    sub = df[(dates > (as_of - __import__("pandas").Timedelta(days=int(lookback * 1.9) + 40)))
             & (dates <= as_of)]
    cols = [c for c in px.columns if c in by_sym]
    if len(cols) < 2:
        raise ValueError("交集为空：可交易域与 h5 instrument 名对不上")
    px = px[cols]
    last = px.iloc[-1]
    prev = px.iloc[-21] if len(px) > 21 else px.iloc[0]
    alpha = -(last / prev - 1.0)                        # 20 日反转，定义同 hub reversal20
    alpha = alpha.dropna().sort_values(ascending=False)

    # 3) 粗筛
    short = list(alpha.index[:int(shortlist)])

    # 4) ML 打分（缺失显式记账）
    ml_scores = {}
    if use_ml and short:
        cand = short[:int(ml_max_symbols)]
        kl = [{"symbol": s, "klines": bars_of(sub, s)} for s in cand]
        kl = [x for x in kl if len(x["klines"]) >= 62]
        if kl:
            try:
                pred = hub_mod.ml_predict(kl)
                if isinstance(pred, dict):
                    # hub 的错误形状是 {"status":"error","message":...}，不是 {"error":...}；
                    # 而且它曾返回过 {"status":"ok","n_predicted":0}——**一只都没算却报成功**。
                    # 两种都必须显式报出来，绝不静默当"ML 没分"。
                    if pred.get("status") == "error":
                        raise hub.HubError(pred.get("message") or str(pred))
                    if pred.get("error"):
                        raise hub.HubError(str(pred["error"]))
                    if pred.get("n_predicted") == 0:
                        raise hub.HubError("ml_predict 返回 ok 但 n_predicted=0（等于没打分）")
                for s, v in (pred.get("predictions") or pred.get("scores") or {}).items():
                    ml_scores[str(s).upper()] = float(v)
                if not ml_scores:
                    warn.append("ml_predict 返回体里没有 predictions/scores 字段")
            except Exception as e:  # noqa: BLE001 — 显式记账，不静默
                warn.append("ml_predict 不可用: %s" % e)
        else:
            warn.append("ML 输入不足：没有 >=62 根 bar 的标的")
    elif not use_ml:
        warn.append("use_ml=False（本轮未接 ML 打分）")

    # 5) 排序取 top_n，再交给 hub 做组合优化（出权重，不是等权）
    # 选择序的**依据**必须是有证据的那个：
    #   alpha  —— 反转20，有实测 IC（factor_recent_ic +0.053）→ 默认
    #   ml     —— 仅当 ml_metrics 里真有 OOS 指标（ic/rank_ic）时才该用；
    #             2026-09-15 实测 ml_metrics 返回 {model_exists: true, metrics: {}}（空），
    #             那等于用未验证信号定序，所以不设为默认。
    if ranking == "ml":
        ranked = sorted(short, key=lambda x: (-ml_scores.get(x, float("-inf")), -float(alpha[x])))
        basis = "ml→alpha（需 ml_metrics 有 OOS 指标才成立）"
        if not ml_scores:
            warn.append("ranking=ml 但本轮没有任何 ml 分，实际退化为 alpha 排序")
    else:
        ranked = sorted(short, key=lambda x: -float(alpha[x]))
        basis = "alpha(reversal20，实测 IC +0.053)"
    sel_order = {sym: i for i, sym in enumerate(ranked, start=1)}   # 选择序：ml 优先、无 ml 回落 alpha
    pick = ranked[:int(top_n)]
    klines = {sym: bars_of(sub, sym) for sym in pick}
    klines = {sym: v for sym, v in klines.items() if len(v) >= 21}
    pick = [sym for sym in pick if sym in klines]
    if len(pick) < 2:
        raise ValueError("组合优化至少需要 2 只有效标的，实际 %d" % len(pick))
    if weighting == "alpha_tilted":
        # B1 的正解：本地做 max αᵀw − λwᵀΣw（含 §8 单票上限），让权重反映"多看好"，
        # 而不只是相关性。hub 的 portfolio_optimize 没有期望收益入参，做不了这件事。
        rets = px[[c for c in pick if c in px.columns]].pct_change().dropna(how="all")
        weights = optimize.alpha_tilted_weights(
            {s: float(alpha.get(s, 0.0)) for s in pick}, rets,
            lam=float(lam), max_weight=float(max_weight))
        method = "alpha_tilted"
    else:
        opt = hub_mod.portfolio_optimize(pick, klines, method=method, lookback=lookback)
        if isinstance(opt, dict) and opt.get("error"):
            raise hub.HubError("组合优化失败: %s" % opt["error"])
        weights = opt.get("weights") or opt.get("target_weights") or opt
        weights = {str(k).upper(): float(v) for k, v in weights.items()
                   if isinstance(v, (int, float)) and v >= 0}

    # 6) §8 风控约束：**能执行的执行，不能执行的显式记账**（不许静默跳过）
    raw_weights = dict(weights)
    capped_w, capped_n = apply_weight_cap(raw_weights, max_weight)
    if capped_n < 0:
        warn.append("§8 单票上限 %.0f%% × %d 只 = %.0f%% < 100%%，约束**不可行**："
                    "本轮不做单票截断（需增加持仓数或放宽上限）"
                    % (100 * max_weight, len(raw_weights), 100 * max_weight * len(raw_weights)))
        weights = raw_weights
    else:
        weights = capped_w

    has_industry = any((by_sym.get(sym, {}).get("industry") or "").strip() for sym in pick)
    ind_agg = {}
    for sym in pick:
        ind = (by_sym.get(sym, {}).get("industry") or "").strip() or "UNKNOWN"
        ind_agg[ind] = ind_agg.get(ind, 0.0) + weights.get(sym, 0.0)
    worst = max(ind_agg.values()) if ind_agg else 0.0
    if not has_industry:
        warn.append("单板块 %.0f%% 约束**未执行**：本轮 universe 的 industry 全为空"
                    "（腾讯兜底不提供行业），集中度无法校验" % (100 * industry_cap))
    elif worst > industry_cap + 1e-9:
        warn.append("单板块暴露 %.1f%% 超过 §8 上限 %.0f%%（标的选择未做行业中性，"
                    "这是 §4.2 指出的未中性化风险）" % (100 * worst, 100 * industry_cap))

    # 7) 落库：rank = **选择序**（ml→alpha），reason 里的每个数字都要是真的
    date = as_of.strftime("%Y-%m-%d")
    ml_rank = {sym: i for i, sym in enumerate(
        sorted([x for x in pick if x in ml_scores], key=lambda x: -ml_scores[x]), start=1)}
    trimmed = max(capped_n, 0)
    rows = []
    for sym in sorted(pick, key=lambda x: -weights.get(x, 0.0)):
        info = by_sym.get(sym, {})
        mlv = ml_scores.get(sym)
        reason = "选择序 %d/%d（依据 %s）" % (sel_order.get(sym, 0), len(ranked), basis)
        reason += " | alpha %.4f" % float(alpha.get(sym, 0))
        reason += " | ml " + ("%.5f（第%d/%d）" % (mlv, ml_rank.get(sym, 0), len(ml_scores))
                             if mlv is not None else "unavailable")
        reason += " | %s %.2f%%" % (method.upper(), 100 * weights.get(sym, 0.0))
        if raw_weights.get(sym, 0) > max_weight + 1e-12:
            reason += "（原 %.2f%% 已按 §8 单票上限 %.0f%% 截断）" % (
                100 * raw_weights[sym], 100 * max_weight)
        rows.append({"symbol": sym.lower(), "name": info.get("name"),
                     "alpha": float(alpha.get(sym, 0)),
                     "target_weight": weights.get(sym, 0.0), "ml_score": mlv,
                     "rank": sel_order.get(sym, 0),
                     "reason": reason,
                     "factor_snapshot": {"reversal20": float(alpha.get(sym, 0)),
                                         "amount_wan": info.get("amount_wan"),
                                         "industry": info.get("industry")}})
    conn = store.connect(db_path)
    n = store.write_candidates(conn, date, rows)
    conn.close()
    summary = {"date": date, "as_of": date, "universe": len(uni_rows),
               "universe_source": uni.get("source"), "shortlist": len(short),
               "picked": len(pick), "written": n,
               "weights_sum": round(sum(weights.values()), 6),
               "weights_sum_before_cap": round(sum(raw_weights.values()), 6),
               "capped_symbols": trimmed,
               "industry_max_exposure": round(worst, 4),
               "industry_constraint_applied": bool(has_industry),
               "ranking_basis": basis, "ranking_mode": ranking,
               "weighting": weighting, "lambda": lam,
               "ml_scored": len(ml_scores), "warnings": warn,
               "elapsed_sec": round(time.time() - t0, 1)}
    return summary
