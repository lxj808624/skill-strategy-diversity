"""Step 3: 两臂组合指标（IS/OOS/ALL）+ 有效策略数 + 相关性上限 + 逐年滚动 + IS 第一名 OOS 分位.

平台：PandaAI EVO（Python 3.11，项目根 /home/coder/project/projects/default）
输出：portfolio_metrics.csv, rolling_by_year.csv, summary.json
"""
import os, json
import numpy as np
import pandas as pd

OUT = "/home/coder/project/projects/default/experiments/diversity"
IS_END = "2022-12-31"
TAU = 0.70
R = pd.read_csv(os.path.join(OUT, "returns_wide.csv.gz"), index_col=0, parse_dates=[0])
sel = pd.read_csv(os.path.join(OUT, "selection.csv"))
S = pd.read_csv(os.path.join(OUT, "sim_joint.csv.gz"), index_col=0)
met = pd.read_csv(os.path.join(OUT, "metrics_is.csv")).set_index("candidate_id")

def stats(r):
    r = r.dropna()
    if len(r) < 20:
        return (np.nan, np.nan, np.nan, np.nan)
    eq = (1 + r).cumprod()
    ar = eq.iloc[-1] ** (252 / len(r)) - 1
    vl = r.std() * np.sqrt(252)
    mdd = float((eq / eq.cummax() - 1).min())
    return (ar, vl, ar / vl if vl else np.nan, mdd)

res = []
for arm, g in sel.groupby("arm"):
    ids = list(g.sort_values("rank").candidate_id)
    pr = R[ids].mean(axis=1)
    for win, rr in [("IS", pr.loc[:IS_END]), ("OOS", pr.loc[IS_END:].iloc[1:]), ("ALL", pr)]:
        ar, vl, sh, mdd = stats(rr)
        res.append({"arm": arm, "window": win, "n": len(ids), "ann_ret": ar, "ann_vol": vl,
                    "sharpe": sh, "mdd": mdd, "days": len(rr.dropna())})
    cor = S.loc[ids, ids].values.copy()
    np.fill_diagonal(cor, np.nan)
    comp = []
    for cid in ids:
        if all(S.loc[cid, c] < TAU for c in comp):
            comp.append(cid)
    res.append({"arm": arm, "window": "structure", "n": len(ids), "eff_n": len(comp),
                "max_pair_corr": float(np.nanmax(cor)), "mean_pair_corr": float(np.nanmean(cor))})
PM = pd.DataFrame(res)
PM.to_csv(os.path.join(OUT, "portfolio_metrics.csv"), index=False)
print("corr mats ok")

# rolling: select by prior-years IS sharpe (A) vs diversity-constrained (B), test next year
years = sorted(set(R.index.year))
rows = []
for y in years[1:]:
    tr = R[R.index.year < y]
    te = R[R.index.year == y]
    if len(tr) < 200 or len(te) < 30:
        continue
    sh = tr.apply(lambda s: stats(s)[2]).dropna().sort_values(ascending=False)
    A = list(sh.index[:10])
    B = []
    for cid in sh.index:
        if len(B) >= 10:
            break
        if all(S.loc[cid, c] < TAU for c in B):
            B.append(cid)
    for arm, ids in [("A_no_constraint", A), ("B_diversity", B)]:
        pr = te[ids].mean(axis=1)
        ar, vl, s2, mdd = stats(pr)
        rows.append({"year": y, "arm": arm, "n": len(ids), "test_days": len(pr.dropna()),
                     "test_ann_ret": ar, "test_sharpe": s2, "test_mdd": mdd,
                     "train_top_sharpe": float(sh.iloc[0])})
RB = pd.DataFrame(rows)
RB.to_csv(os.path.join(OUT, "rolling_by_year.csv"), index=False)

# IS top-1 candidate OOS percentile
allsh = R.apply(lambda s: stats(s)[2]).dropna().sort_values(ascending=False)
topc = list(allsh.index[:1])[0]
is_sh = met.loc[topc, "sharpe"]
top_r = R[topc]
oos_r = top_r.loc[IS_END:].iloc[1:]
top_oos = stats(oos_r)[2]
pool_oos = R.apply(lambda s: stats(s.loc[IS_END:].iloc[1:])[2]).dropna()
pct = float((pool_oos < top_oos).mean()) if len(pool_oos) else np.nan
summary = {
  "top_is_candidate": topc, "top_is_sharpe": float(is_sh), "top_oos_sharpe": float(top_oos),
  "top_oos_pctile_in_pool": pct, "pool_oos_sharpe_median": float(pool_oos.median()),
  "A_is": float(PM.query("arm=='A_no_constraint' and window=='IS'").sharpe.iloc[0]),
  "B_is": float(PM.query("arm=='B_diversity' and window=='IS'").sharpe.iloc[0]),
  "A_oos": float(PM.query("arm=='A_no_constraint' and window=='OOS'").sharpe.iloc[0]),
  "B_oos": float(PM.query("arm=='B_diversity' and window=='OOS'").sharpe.iloc[0]),
  "A_maxcorr": float(PM.query("arm=='A_no_constraint' and window=='structure'").max_pair_corr.iloc[0]),
  "B_maxcorr": float(PM.query("arm=='B_diversity' and window=='structure'").max_pair_corr.iloc[0]),
  "A_effn": int(PM.query("arm=='A_no_constraint' and window=='structure'").eff_n.iloc[0]),
  "B_effn": int(PM.query("arm=='B_diversity' and window=='structure'").eff_n.iloc[0]),
  "n_pool": int(R.shape[1]), "n_dates": int(R.shape[0]),
  "rolling_n": int(len(RB)),
  "cost": "2bp per side on turnover, signal t -> ret t+1"
}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2, ensure_ascii=False))
print(PM.to_string(index=False))
print("files", sorted(os.listdir(OUT)))
print(RB.round(3).to_string(index=False))
