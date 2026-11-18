"""τ 与四路权重敏感性扫描（离线、纯标准库，复用 step2/step3 落盘产物）。

输入：outputs/sim_components.csv（7,140 对 code/signal/position/return 相似度）
      outputs/metrics_is.csv（各候选 IS Sharpe）
      outputs/returns_wide.csv.gz（1,899 日 × 120 候选 净收益）
输出：outputs/sensitivity_tau.csv、outputs/sensitivity_weight.csv
口径与主实验一致：equal 权重、IS<=2022-12-31、OOS>2022-12-31、每簇取 1 取满 10、簇内按 IS Sharpe。
"""
import csv, gzip, math, statistics as st
from collections import defaultdict

comp = list(csv.DictReader(open("outputs/sim_components.csv")))
met = {r["candidate_id"]: float(r["sharpe"]) for r in csv.DictReader(open("outputs/metrics_is.csv"))}

# 收益宽表
with gzip.open("outputs/returns_wide.csv.gz", "rt") as f:
    rows = list(csv.reader(f))
ids = rows[0][1:]
dates = [r[0] for r in rows[1:]]
rets = {c: [] for c in ids}
for r in rows[1:]:
    for c, v in zip(ids, r[1:]):
        try: rets[c].append(float(v))
        except ValueError: rets[c].append(0.0)

IS_CUT = "2022-12-31"
is_idx = [i for i, d in enumerate(dates) if d <= IS_CUT]
oos_idx = [i for i, d in enumerate(dates) if d > IS_CUT]

def stat(series):
    """与 step3 完全一致：ann_ret = 末净值^(252/n)-1，ann_vol = std*sqrt(252)，sharpe = ann_ret/ann_vol。"""
    if len(series) < 30: return (None, None, None)
    eq = [1.0]
    for x in series: eq.append(eq[-1] * (1 + x))
    ar = eq[-1] ** (252 / len(series)) - 1
    vl = st.stdev(series) * math.sqrt(252)
    sh = ar / vl if vl else float("nan")
    peak, mdd = 1.0, 0.0
    for v in eq:
        peak = max(peak, v); mdd = min(mdd, v / peak - 1)
    return (sh, ar, mdd)

def arm(sel):
    p = [st.mean([rets[c][i] for c in sel]) for i in range(len(dates))]
    isr = [p[i] for i in is_idx]; oosr = [p[i] for i in oos_idx[1:]]  # 与 step3 一致：OOS 去掉切分当日
    return dict(IS=stat(isr), OOS=stat(oosr))

def sim(weights):
    w = dict(zip(["code", "signal", "position", "return"], weights))
    S = defaultdict(dict)
    for r in comp:
        a, b = r["pair"].split("-")
        v = sum(float(r[k]) * w[k] for k in w)
        S[a][b] = v; S[b][a] = v
    return S

def greedy(S, tau, order, k=None):
    """与 step2 完全一致的贪心去重：按给定顺序扫描，若与已选全部相似度<tau 则入选。"""
    sel = []
    for cid in order:
        if k is not None and len(sel) >= k:
            break
        if all(S[cid].get(s, 0.0) < tau for s in sel):
            sel.append(cid)
    return sel

A = sorted(ids, key=lambda x: -met[x])[:10]
BASE = (0.30, 0.30, 0.20, 0.20)

out_tau, out_w = [], []
a_met = arm(A); a_met["n_eff"] = len(greedy(sim(BASE), 0.70, A))
out_tau.append(["A_fixed", 0.70, "30/30/20/20", a_met["n_eff"], *a_met["IS"], *a_met["OOS"]])

order = sorted(ids, key=lambda x: -met[x])
for tau in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    selB = greedy(sim(BASE), tau, order, k=10); m = arm(selB)
    out_tau.append([f"B_tau{tau:.2f}", tau, "30/30/20/20", len(selB), *m["IS"], *m["OOS"]])

for name, w in [("equal_25", (0.25, 0.25, 0.25, 0.25)), ("code_heavy", (0.50, 0.20, 0.15, 0.15)),
                ("behavior_heavy", (0.10, 0.35, 0.25, 0.30)), ("return_heavy", (0.10, 0.20, 0.20, 0.50)),
                ("code_only", (1, 0, 0, 0)), ("behavior_only", (0, 0.40, 0.30, 0.30))]:
    selB = greedy(sim(w), 0.70, order, k=10); m = arm(selB)
    out_w.append([f"B_{name}", 0.70, name, len(selB), *m["IS"], *m["OOS"]])

HDR = ["arm", "tau", "weights", "n_eff", "IS_sharpe", "IS_ann_ret", "IS_mdd", "OOS_sharpe", "OOS_ann_ret", "OOS_mdd"]
for path, rows in [("outputs/sensitivity_tau.csv", out_tau), ("outputs/sensitivity_weight.csv", out_w)]:
    with open(path, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(HDR); wr.writerows(rows)

print("HDR:", *HDR)
for r in out_tau + out_w:
    print(*[f"{x:.4f}" if isinstance(x, float) else x for x in r])
