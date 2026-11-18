"""Step 1: 固定搜索预算下的候选策略池 + 样本内(IS)指标.

运行平台：PandaAI EVO / QuantStudio（项目根 /home/coder/project/projects/default）
数据：reports/<run>/data/futures_daily.csv.gz（主力连续后复权、价差法，get_future_daily_post）
产物：experiments/diversity/{candidates.csv, metrics_is.csv, returns_wide.csv.gz,
      positions_wide.csv.gz, futures_daily.csv.gz}
"""
import os, glob, shutil
import numpy as np
import pandas as pd

ROOT = "/home/coder/project/projects/default"
OUT = os.path.join(ROOT, "experiments", "diversity")
os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(OUT, "futures_daily.csv.gz")
if not os.path.exists(DATA):
    hits = glob.glob(os.path.join(ROOT, "reports", "*", "data", "futures_daily.csv.gz"))
    shutil.copy2(sorted(hits)[-1], DATA)
print("data size", os.path.getsize(DATA))

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"].astype(str))
S = ["RB", "I", "CU", "AU", "AG", "M", "TA", "MA", "SC", "P", "Y"]
df = df[df["symbol"].isin(S)]
close = df.pivot_table(index="date", columns="symbol", values="close")[S]
ret = close.pct_change()
v20 = ret.rolling(20).std()
vmed = v20.rolling(252, min_periods=120).median()
ma20 = close.rolling(20).mean()
ma60 = close.rolling(60).mean()
rows = [(mo, L, h, f) for mo in ["trend", "reversal"] for L in [5,10,20,40,60] for h in [1,3,5] for f in ["none","ma20","ma60","volcap"]]
cand = pd.DataFrame(rows, columns=["mode","L","hold","filter"])
cand["weight"] = "equal"
cand["candidate_id"] = ["C%03d" % i for i in range(len(cand))]
print("pool", len(cand), "symbols", close.shape[1], "days", close.shape[0])
cand.to_csv(os.path.join(OUT, "candidates.csv"), index=False)

R, P = {}, {}
for _, x in cand.iterrows():
    sg = np.sign(close / close.shift(int(x["L"])) - 1) * (1 if x["mode"] == "trend" else -1)
    if x["filter"] == "ma20":
        sg = sg.where(((sg > 0) & (close > ma20)) | ((sg < 0) & (close < ma20)), 0)
    elif x["filter"] == "ma60":
        sg = sg.where(((sg > 0) & (close > ma60)) | ((sg < 0) & (close < ma60)), 0)
    elif x["filter"] == "volcap":
        sg = sg.where(v20 > vmed, 0)
    ps = sg.rolling(int(x["hold"])).mean()
    w = ps.div(ps.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    R[x["candidate_id"]] = (w * ret.shift(-1)).sum(axis=1) - 0.0002 * w.diff().abs().sum(axis=1)
    P[x["candidate_id"]] = w.sum(axis=1)
Rw = pd.DataFrame(R)
Pw = pd.DataFrame(P)
Rw.to_csv(os.path.join(OUT, "returns_wide.csv.gz"), compression="gzip")
Pw.to_csv(os.path.join(OUT, "positions_wide.csv.gz"), compression="gzip")
IS = Rw.loc[:"2022-12-31"]
rec = []
for k in Rw.columns:
    s = IS[k].dropna()
    eq = (1 + s).cumprod()
    ar = eq.iloc[-1] ** (252 / len(s)) - 1
    vl = s.std() * np.sqrt(252)
    dd = float((eq / eq.cummax() - 1).min())
    sh = ar / vl if vl else np.nan
    rec.append((k, ar, vl, sh, dd))
M = pd.DataFrame(rec, columns=["candidate_id","ann_ret","ann_vol","sharpe","mdd"]).merge(cand, on="candidate_id")
M.to_csv(os.path.join(OUT, "metrics_is.csv"), index=False)
print("IS days", len(IS), "shape", Rw.shape)
print(M.sort_values("sharpe", ascending=False).head(5)[["candidate_id","mode","L","hold","filter","sharpe","ann_ret","mdd"]].round(3).to_string(index=False))
print(M.sort_values("sharpe").head(3)[["candidate_id","sharpe"]].round(3).to_string(index=False))
print(sorted(os.listdir(OUT)))
