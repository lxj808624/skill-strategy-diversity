"""Step 2: 四路（代码/信号/持仓/收益）联合相似度 + tau 连通聚类 + 两臂选择.

平台：PandaAI EVO（Python 3.11，项目根 /home/coder/project/projects/default）
输入：experiments/diversity/{candidates.csv, metrics_is.csv, returns_wide.csv.gz, positions_wide.csv.gz}
输出：sim_joint.csv.gz, sim_components.csv, selection.csv, selection_summary.json
"""
import os, json
import numpy as np
import pandas as pd

OUT = "/home/coder/project/projects/default/experiments/diversity"
IS_END = "2022-12-31"
TAU = 0.70

cand = pd.read_csv(os.path.join(OUT, "candidates.csv")).set_index("candidate_id")
met = pd.read_csv(os.path.join(OUT, "metrics_is.csv")).set_index("candidate_id")
R = pd.read_csv(os.path.join(OUT, "returns_wide.csv.gz"), index_col=0, parse_dates=[0])
P = pd.read_csv(os.path.join(OUT, "positions_wide.csv.gz"), index_col=0, parse_dates=[0])
ids = list(cand.index)
R_is = R.loc[:IS_END][ids]
P_is = P.loc[:IS_END][ids]
C_ret = R_is.corr().abs().fillna(0).values
C_sig = P_is.corr().abs().fillna(0).values
nz = (P_is.abs() > 1e-9).values
pv = P_is.values
n = len(ids)
J = np.zeros((n, n))
for i in range(n):
    a = nz[:, i]
    for j in range(i, n):
        b = nz[:, j]
        u = (a | b).sum()
        if u == 0:
            J[i, j] = J[j, i] = 0.0
            continue
        both = a & b
        agree = float((np.sign(pv[both, i]) == np.sign(pv[both, j])).mean()) if both.any() else 0.0
        J[i, j] = J[j, i] = (a & b).sum() / u * agree
print("corr mats done", C_ret.shape)

mode_map = {"trend": 0.0, "reversal": 1.0}
flt_map = {"none": 0.0, "ma20": 0.5, "ma60": 0.5, "volcap": 1.0}
V = np.array([[mode_map[r["mode"]], (r["L"] - 5) / 55.0, (r["hold"] - 1) / 4.0, flt_map[r["filter"]]] for _, r in cand.iterrows()])
Wv = np.array([0.25, 0.30, 0.20, 0.25])
D = np.zeros((n, n))
for i in range(n):
    D[i] = (np.abs(V - V[i]) * Wv).sum(axis=1)
C_code = 1.0 - D / D.max()
S = 0.30 * C_code + 0.30 * C_sig + 0.20 * J + 0.20 * C_ret
np.fill_diagonal(S, 1.0)
Sdf = pd.DataFrame(S, index=ids, columns=ids)
Sdf.to_csv(os.path.join(OUT, "sim_joint.csv.gz"), compression="gzip")
comp = pd.DataFrame({"pair": [ids[i] + "-" + ids[j] for i in range(n) for j in range(i+1, n)],
                     "code": [C_code[i, j] for i in range(n) for j in range(i+1, n)],
                     "signal": [C_sig[i, j] for i in range(n) for j in range(i+1, n)],
                     "position": [J[i, j] for i in range(n) for j in range(i+1, n)],
                     "return": [C_ret[i, j] for i in range(n) for j in range(i+1, n)],
                     "joint": [S[i, j] for i in range(n) for j in range(i+1, n)]})
comp.to_csv(os.path.join(OUT, "sim_components.csv"), index=False)
order = met["sharpe"].dropna().sort_values(ascending=False)
selA = list(order.index[:10])
selB = []
for cid in order.index:
    if len(selB) >= 10:
        break
    if all(Sdf.loc[cid, s] < TAU for s in selB):
        selB.append(cid)

sel = pd.concat([pd.DataFrame({"arm": "A_no_constraint", "candidate_id": selA, "rank": range(1, len(selA)+1)}),
                 pd.DataFrame({"arm": "B_diversity", "candidate_id": selB, "rank": range(1, len(selB)+1)})])
sel.to_csv(os.path.join(OUT, "selection.csv"), index=False)
sm = {"n_pool": n, "tau": TAU, "n_selected_A": len(selA), "n_selected_B": len(selB),
      "n_eff_A": n, "n_eff_B": int(len(selB)),
      "mean_pair_joint_A": float(np.mean([S[i, j] for i in range(n) for j in range(i+1, n)])),
      "max_pair_joint_A": float(np.max([S[i, j] for i in range(n) for j in range(i+1, n)]))}
with open(os.path.join(OUT, "selection_summary.json"), "w") as f:
    json.dump(sm, f, indent=2)
print("selected A", selA)
print("selected B", selB)
print("mean pairwise joint %.3f  max %.3f" % (sm["mean_pair_joint_A"], sm["max_pair_joint_A"]))
print("files", sorted(os.listdir(OUT)))
