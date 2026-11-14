# 论文实验问题与冻结证据审计

## 使用规则

本文件把师兄提出的任务拆成 **6 个大问题**。每个大问题下都分别列出：

- 要回答的子问题；
- 现有冻结证据；
- 当前可以写进论文的结论；
- 尚缺失的证据与不可写的结论。

实验已经冻结。本审计只读取现有 Markdown、JSON、轨迹、截图和 HTML；不修改
harness、不调用模型、不重放浏览器，也不重新计算官方分数。

主要 benchmark 为 Interaction2Code。候选级综合分定义为：

```text
B = mean(full_page_clip, full_page_ssim, full_page_text,
         interaction_clip, interaction_ssim, interaction_text,
         position, implemented)
```

`position_after` 仅作诊断，不计入 `B`。

---

## 问题 1：GRADE 到底有没有用？

### 子问题 1.1：Actor harness evolution 是否提升前端生成质量？

**需要的比较**：Minimal Harness 与 Actor-only Evolution。

**冻结证据**：在 64 case、Qwen3.5-9B、8 个冻结 Actor 候选池上：

| Actor | 机制 | Mean B | IR | 成功产出 |
| --- | --- | ---: | ---: | ---: |
| A0 | XML minimal | 0.2811 | 0.3750 | 47/64 |
| A6 | `edit-before-stop` 进化 | 0.3938 | 0.5313 | 59/64 |
| A1 | 历史 visual-only 变体 | 0.4430 | 0.6563 | 50/64 |

**可写结论**：相较 A0，A6 提升 `+0.1127 B` 和 `+15.63` 个 IR 点，支持
“Actor harness evolution 提升 Interaction2Code 候选质量”。A1 更强，但它是历史
泛化变体，不应写成与 A0 完全隔离的自动进化接受实验。

### 子问题 1.2：Full GRADE 是否优于 Actor-only？

**需要的比较**：相同 Actor draft，在引入 Judge feedback 和 repair 后是否高于
Actor-only。

**历史过程证据（非最终 CCEF G3）**：独立 64-case holdout 使用的是旧 scalar Judge：

| 条件 | Mean B | IR |
| --- | ---: | ---: |
| A1 Actor-only 草稿 | 0.5682 | 0.8438 |
| A1 -> 历史 scalar Judge -> A1 repair | 0.5116 | 0.7813 |

结果为 `delta B = -0.0567`（相对 `-9.97%`），胜/平/负为 22/3/39。

**可写结论**：该结果只能说明历史 scalar feedback + repair 协议没有提升，不能用于
评价最终 CCEF G3。现有冻结材料仍没有最终 CCEF G3 驱动 repair 后与 Actor-only 的
同协议对比，因此主结果中的 Full GRADE vs Actor-only 尚不能由这条历史结果回答。

### 子问题 1.3：能否完成三 benchmark、Heavy Harness、弱 4B 的主表？

**需要的比较**：WebGen、Vision2Web、Interaction2Code 上的 Minimal、Heavy、
Actor-only、Full GRADE；主模型为 9B，并用额外 4B 复现四行；所有方法匹配
官方 Judge 调用预算。

**冻结证据状态**：只有 Interaction2Code 有当前 GRADE 兼容产物。没有同预算
Heavy Harness 对比，没有 WebGen/Vision2Web 的完整四方法协议，也没有 4B 的四行
结果。历史 WebGen/Vision2Web 目录是独立生成实验，不能拼入此主表。

**结论**：无法用现有冻结数据生成师兄要求的完整主结果表。

---

## 问题 2：Judge evolution、重复聚合标签和 Challenge 是否必要？

### 子问题 2.1：进化 Judge 是否优于 Raw VLM Judge？

**需要的比较**：Raw VLM、单次官方标签训练的 G1、重复聚合标签训练的 G2、完整
GRADE Judge G3；指标为 pairwise accuracy、`R_2`、`HJ/LB@2`。

**论文最终结果**：64 case x 8 Actor 候选 = 512 个候选，Qwen3.5-9B Actor、
Mimo-v2.5 Judge。以下数据以最终论文表为唯一权威口径：

| Judge variant | Pairwise accuracy@0 | R2 | HJ/LB@2 |
| --- | ---: | ---: | ---: |
| G0 Raw VLM Judge | 0.6583 | 0.3672 | 0.0938 |
| G1 Evolved Judge, single label | 0.7492 | 0.5313 | 0.0547 |
| G2 Evolved Judge, aggregated label | 0.7492 | 0.5313 | 0.0547 |
| G3 GRADE Judge, CCEF | 0.7594 | 0.5703 | 0.0313 |

**可写结论**：G1/G2 相对 Raw VLM 显著提高 pairwise accuracy 和 shortlist recall，
并降低 HJ/LB，支持 Judge evolution 有效。最终 G3 CCEF 在三项指标上均优于 G1/G2；
其中 `HJ/LB@2` 从 `0.0547` 降至 `0.0313`，支持 CCEF 降低 Judge 高分但 benchmark
低分的错误选择。

### 子问题 2.2：重复聚合官方标签是否有收益？

**需要的比较**：G1（单次标签）对 G2（重复并聚合标签）。

**冻结证据**：这 512 个候选的官方 evaluator 三次重复在浮点精度下完全一致：
分数标准差均为 0，pairwise flip rate 为 `0/4689`。因此 G1 与 G2 的训练标签相同，
三项指标也相同。

**可写结论**：本次官方 evaluator 在该候选池上稳定；现有数据**不能证明**聚合标签
有额外收益，也不能证明“官方 Judge 存在显著波动”。

### 子问题 2.3：Challenge 是否减少 Judge exploit / Goodhart？

**需要的比较**：最终 G3 CCEF 与无 Challenge 的 G2 使用相同评测候选池和官方 audit
预算，比较 `HJ/LB@2` 以及其他排序指标。

**论文最终证据**：G2 为 `0.7492 / 0.5313 / 0.0547`，G3 CCEF 为
`0.7594 / 0.5703 / 0.0313`。G3 的 pairwise accuracy 提升 `0.0102`，R2 提升
`0.0390`，HJ/LB@2 下降 `0.0234`（越低越好）。

**可写结论**：在匹配的正式消融中，CCEF 提高排序和 shortlist 质量，并减少
High-Judge / Low-Benchmark 候选，支持 Challenge/CCEF 缓解 Judge exploit 与
Goodhart 风险。

### 子问题 2.4：正式 G3 是否能指导有效 repair？

**冻结证据**：最终 CCEF 表评估的是 Judge 排序与筛选能力，并未直接报告 repair 后
的 `delta B`。历史 scalar Judge 的 repair 结果属于旧协议，不能代表最终 CCEF。

**可写结论**：可以宣称 CCEF 改善 Judge 筛选并减少 exploit；不能从该表进一步宣称
CCEF 一定提高 Actor repair 后的页面分数。

### 子问题 2.5：官方评测器本身是否有噪声？

**需要的统计**：同一冻结候选重复官方评测的分数标准差与 pairwise flip rate。

**冻结证据**：64-case scalar 池的 3 次官方重复显示均值/中位数/最大标准差为 0，
pairwise flip rate 为 `0/4689`。

**注意**：师兄原文写“35 次”，此前讨论过 3 次或 3--5 次；现有结果只有 3 次。
论文必须写真实重复次数 3，不能写 35 次。

---

## 问题 3：相同官方预算下，GRADE screening 能否找到更好的 Harness？

### 子问题 3.1：三种 search/screening 方法在相同官方调用预算下的曲线如何？

**需要的比较**：Official-only search、Raw VLM screening、GRADE screening；
横轴为累计官方 Judge 调用，纵轴为当前 best harness 的聚合官方分数。三者必须共享
mutation operator 与总官方预算。

**冻结证据**：仅有 smoke run：3 case、1 轮、每轮 2 个候选、`K=1`、每组计费官方
调用数均为 3：

| 方法 | 初始 B | 一轮后 B | 计费官方调用数 |
| --- | ---: | ---: | ---: |
| Official-only | 0.4398 | 0.4398 | 3 |
| Raw VLM screening | 0.4398 | 0.4545 | 3 |
| 历史 GRADE screening（非最终 CCEF） | 0.4398 | 0.6803 | 3 |

### 子问题 3.2：是否能报告 N、K、Actor rollouts 与成本？

**冻结证据状态**：该 smoke 记录了 `N=2`、`K=1` 与官方调用数；没有多轮、多 seed、
完整 actor rollout 总数或可用于论文的 wall-clock / cost 汇总。

**结论**：该结果仅验证 pipeline 与一次匹配预算记账，不能绘制 Phase II 曲线或支持
“相同官方预算下最终找到更强 harness”的中心论点。

---

## 问题 4：Test-time Best-of-N 是否稳定有效？

### 子问题 4.1：N 增大时，GRADE Judge selection 是否优于随机和 Raw VLM？

**需要的比较**：`N=1,2,4,8` 下的 Random selection、Raw VLM selection、
GRADE Judge selection 与 Oracle selection。

### 子问题 4.2：提升是否达到进入正文的阈值？

**需要的判据**：稳定提升约 3--5 分才进入正文，否则仅放附录。

**冻结证据状态**：没有受控的 Best-of-N 结果。

**结论**：本节没有可报告结果，不应放入正文。

---

## 问题 5：能否给出一个完整的 Judge exploit 定性案例？

### 子问题 5.1：是否存在“Judge 高分、官方低分”的候选？

### 子问题 5.2：Failure Signature 是否定位到 Judge 盲点？

### 子问题 5.3：更新 Judge 后是否拒绝同类错误候选？

**师兄要求的完整链条**：高 Judge 分但低官方分 -> 重复官方确认 -> Failure Signature
定位 -> 更新 Judge Harness -> 更新后拒绝同类错误。

**冻结证据**：

- `cases/ccef_exploit_68_2/`：最终选定的论文交付案例。`68_2__A2` 被 G2 以 0.72
  排在 case 第 1，但三次官方评价均为 B=0.1116、implemented=0；CCEF 根据“参考
  有状态变化但候选 action 未执行”的 Failure Signature 将其降至 0.37、case 第 3，
  并对两个同类候选产生相同纠正。

**结论**：`ccef_exploit_68_2` 已覆盖“高 Judge/低官方 -> Failure Signature -> 更新
Judge Harness -> 拒绝同类错误”的完整链条，可作为正文定性案例。需要准确表述为
CCEF 修正 Judge 排序，不应扩写成 Actor repair 成功案例。

详见 [案例索引](cases/CASE_INDEX.md)。

---

## 问题 6：最终论文需要哪些结果，现有冻结材料能交付哪些？

### 子问题 6.1：主结果表是否齐全？

**目标**：三个 benchmark、9B 与弱 4B、Minimal/Heavy/Actor-only/Full GRADE。

**现状**：不齐全。只有 Interaction2Code 上的 Actor-only 正证据，且 Full repair
为负；没有三 benchmark、Heavy 与 4B 的匹配比较。

### 子问题 6.2：Judge 消融表和官方噪声表是否齐全？

**目标**：G0--G3 在同一正式划分上的三项指标，加官方重复噪声。

**现状**：最终 64-case G0--G3 表完整，正式 G3 为 CCEF；官方噪声的 3 次重复结果
存在，但不是 35 次。G1 与 G2 相同，因此该数据不能证明聚合标签的独立收益。

### 子问题 6.3：Score vs Official Calls 曲线是否齐全？

**现状**：仅一个单轮 smoke 点，不具备曲线、多 seed、成本或总 rollout 报告条件。

### 子问题 6.4：Best-of-N 和 exploit case 是否齐全？

**现状**：Best-of-N 没有数据；CCEF exploit 定性案例已经选定并完成证据归档。

### 可交付结论

当前最完整、最可复核的材料是：

1. Interaction2Code 上 Actor harness evolution 相对 minimal baseline 的正提升；
2. 旧 scalar Judge 64-case 消融中，evolved Judge 相对 Raw VLM 的排序提升；
3. 官方评测器在这批候选上的 3 次重复稳定性；
4. 最终 G3 CCEF 相对 G1/G2 的三项提升，尤其 `HJ/LB@2` 从 0.0547 降至 0.0313；
5. Full repair loop 为负的透明反证；
6. 一个可检查的 CCEF exploit 完整案例。

不能用现有冻结材料宣称：三 benchmark 泛化、4B 泛化、Heavy Harness 比较、Full
GRADE 端到端优于 Actor-only、聚合标签收益、CCEF 改善 repair、Phase II
预算曲线优势、或完整 exploit-repair 成功案例。

---

## 原始来源

- 64-case Judge 消融：`runs/judge_ablation_main/main_20260725_203200_r1/main_summary.json`
- 512 候选语料：`runs/judge_ablation_main/main_20260725_203200_r1/candidate_corpus.json`
- Actor/Judge 候选池来源：`docs/MAIN_TABLE_HARNESS_POOL.md`
- 历史 scalar Judge 与早期 Challenge 记录：`docs/JUDGE_GROUP_EVOLUTION_RECORD.md`
- 完整 repair holdout：`runs/final_harnessgan_aja_holdout64_20260727_001/aja_uplift_summary.json`
- Phase II smoke：`runs/phase2_online_smoke_20260727_007/run/summary.json`
- 最终 G3 CCEF 正式数值：本文件“问题 2”中的用户确认论文最终表
- CCEF 前的 pairwise challenge 开发记录：`runs/judge_g3_pairwise_challenge_heldout16_20260727_001/summary.json`
- Pairwise challenge 开发案例：`runs/judge_g3_pairwise_challenge_heldout16_20260727_001/pairs/56_1/`
- 后续 listwise 原型：`runs/judge_g4_listwise_board_dev8_20260728_002/summary.json`
