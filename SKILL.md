---
name: strategy-diversity-constraint
description: 在固定任务/模型/搜索预算下，用代码-信号-持仓-收益四路联合相似度识别重复策略，对比"无多样性约束"与"多样性约束"两臂的有效独立策略数与组合风险。适用于策略批量生成后的去重、选型与组合构建前置约束检查。
---

# 策略多样性约束（strategy-diversity-constraint）

## 何时用

- 你已经批量生成了一批候选策略（固定池、固定预算），需要判断"名义 K 个策略里真正独立的有几个"
- 你准备按 IS 指标排 top-K 建组合，想先确认这个组合是不是重复下注
- 你要验收"多样性约束"到底有没有用（包括**可能没用**的情形）

## 不用它的情形

- 单策略调参、因子挖掘本身（这一步不做信号研发）
- 需要源码 AST/语义级代码去重（本 skill 的 code 路是参数化离散编码，见下）
- 需要执行撮合级成本核验（本 skill 用声明式 bpx 成本）

## 输入约定

| 输入 | 格式 | 必需 |
|---|---|---|
| 候选清单 | `candidates.csv`，含 `candidate_id` 与策略参数列 | 是 |
| 收益宽表 | `returns_wide.csv.gz`，index=日期，columns=candidate_id | 是 |
| 持仓宽表 | `positions_wide.csv.gz`，同形状（权重/净值敞口） | 是 |
| IS 指标 | `metrics_is.csv`，含 `sharpe` | 是 |
| 代码/参数编码 | 由 `candidates.csv` 列构造 4 维向量 | 是 |

## 四路联合相似度（冻结口径）

```
S = 0.30 * code + 0.30 * signal + 0.20 * position + 0.20 * return
```

- `code`：参数向量加权曼哈顿距离归一（mode 0.25 / lookback 0.30 / hold 0.20 / filter 0.25）。**不是源码相似度**。
- `signal`：IS 区间信号（持仓方向）相关绝对值。
- `position`：`Jaccard(|持仓|非零集合) × 方向一致率`。
- `return`：IS 区间日收益相关绝对值。

`S ≥ 0.70` 视为同簇；**有效策略数 = 连通分量簇数**。

## 两臂对照

- A 无约束：按 IS Sharpe 降序取 top-K
- B 有约束：按 IS Sharpe 降序遍历，与已选集合 `max S < 0.70` 才入选，取满 K
- 两臂只差"是否加多样性过滤"，其余（池、模型、预算、费用、切分）全部相同

## 判据（先写死，再看结果）

1. 有效策略数：B > A 才算"结构改善"
2. 最大配对相关：B < A 才算"风险集中度下降"
3. 样本外：B 的 OOS Sharpe 与逐年滚动胜率**必须单独报告**，不得用 IS 改善替代
4. IS 冠军的 OOS 分位：若接近 50%，则 IS 排名无预测力，需在结论中显式写出

第 3、4 条是防止"结构改善被误读成收益改善"的闸门。

## 输出

`sim_joint.csv.gz`、`sim_components.csv`（逐对四路分量）、`selection.csv`（两臂明细）、
`selection_summary.json`、`portfolio_metrics.csv`、`rolling_by_year.csv`、`summary.json`。

## 执行（EVO / 本地均可）

```bash
python scripts/step1_pool.py     # 无池时先造池；有池可跳过
python scripts/step2_select.py   # 四路相似度 + 聚类 + 两臂
python scripts/step3_eval.py     # 两臂指标 + 有效数 + 逐年滚动
```

脚本除 `ROOT` 常量外无环境依赖（numpy / pandas 即可），全流程无随机性。

## 已知边界

- τ=0.70 与四路权重为冻结单点，未做敏感性扫描；换市场/频率需重新校准
- code 路对自由格式代码库无效，必须替换为 AST 或嵌入
- 结论依赖池的同质程度：本实验池内平均配对相似度 0.477，属"高度同质池"，同质池上多样性约束的结构收益最明显
- 有效数上升不等于收益上升：本实验为负结果（见 README 第 5 节）

## 复用检查清单

- [ ] 两臂除多样性过滤外无其它差异
- [ ] OOS 与滚动结果单独报告，且给出抽样误差量级
- [ ] 有效策略数用簇数而非"去重后条数"
- [ ] 成本口径写明（本次为 2bp/单边假设，非撮合实测）
- [ ] 失败边界与等价回放范围写在前面的文档里，而不是附录
