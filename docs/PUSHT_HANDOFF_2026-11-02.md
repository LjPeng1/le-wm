# Push-T 复现 Hand-off（2026-11-02）

用途：把 CHPC 服务器侧已完成的运行与仍未决的判断项交给本地（远端 AI / 用户）
处理。按分工，服务器只执不判；本文列出**需要判断的内容**与**可直接引用的实验
事实**。

一句话结论：Gate1/2/3/5 全部通过，发布 checkpoint 98.0%（49/50）、从零训练
10 epochs 的 paper10 模型 96.0%（48/50），与论文约 96% 一致；唯一明确的协议
偏离是 **Gate4（恢复等价性）按用户指示跳过**。

## 一、需要远端判断的项

### 1. Gate4 跳过的定性（最重要）

`docs/PUSHT_REPRO_PROTOCOL.md` 规定 Gate5 必须在 Gates3–4 通过后运行；本次
Gate4 在用户运行日指示「把能跳过的全跳过」下被移除
（记录：`$STABLEWM_HOME/ops/STAGE2_GATE4_SKIPPED.md`）。Gate4 只验证抢占/
中断后的恢复等价性，不产生科学结果，但它是协议前置。

需要判断：

- 本次 96.0% 是作为「论文数值复现证据」还是仅作为「一次成功训练」？
- 若作为正式复现记录，是否必须补跑 Gate4？补跑属于工程验证，不是新科学
  结论。

### 2. 单 seed vs 论文的三模型 variation

协议记录：论文的 variation 来自**三个训练模型在同一 50 条评测轨迹上的
结果**；本次只有 1 个模型（global seed 3072）。

- 直接重跑相同配置会得到同一模型；新增 seed 属于协议中的「future
  decision」，需先确定 seed 集合与汇总方式。
- 每个 seed 成本约 **4.3 GPU-h**（训练约 4h12m + 50 起点 CEM <1 min）。

### 3. 10 epochs vs 100 epochs

论文 v3 Appendix E 写明 Push-T 使用 10 epochs，而 `config/train/lewm.yaml`
写的是 100。本次按论文执行 10 epochs。协议将 100-epoch 定义为「future
exploration，不是第一轮额外 run」。需要确认对外记录口径。

### 4. 统计解读

98.0%（49/50）与 96.0%（48/50）的 Wilson 95% 置信区间分别为
**89.5–99.6%** 和 **86.5–98.9%**，完全重叠，不能解读为发布模型优于本次
训练模型。建议表述为「与论文约 96% 一致」。

### 5. 审计/复现口径

- 训练与评测走 speed mode：`data_identity` 记录
  `rehash: "skipped_stat_bound"`（绑定清单摘要 + size/mtime），未重读 58 GB
  训练入口与 44 GB 评测 H5。
- Gate3 走默认全量重哈希，未使用 speed mode。
- 运行日 helper 文件哈希与 HEAD 不同（修复在运行时尚未提交）；Gate 凭据中
  `commit_sha` 仍是 `0dd9b96`。
- 若进入正式报告，需要决定如何标注这些 provenance 事实。

### 6. 数据后端

训练日志显示数据读自 `pusht_expert_train.h5`；论文 config 指向
`pusht_expert_train.lance`。协议要求显式区分 backend 身份、不得假定
H5 与 Lance 等价。需要决定是否作为 comparability limitation 写入报告。

### 7. 仓库动作

结果与 hand-off 已推到 `fyp/chpc-sandbox-prep`；PR 尚未创建（服务器无 `gh`）。
需要决定是否开 PR、合并顺序，以及是否把 `docs/PUSHT_CHPC_RUN_2026-11-01.md`
作为对外结果记录。

## 二、实验事实卡

### 身份与环境

| 项 | 值 |
| --- | --- |
| Run ID | `pusht_20261101T091612Z_3579888` |
| 代码 | `0dd9b96` + 本次 `5c05334` 的修复；凭据记录的 sha 为 `0dd9b96` |
| 平台 | CHPC Linux x86_64，Python 3.10.20 |
| 关键包 | torch 2.8.0+cu128、stable-worldmodel 0.0.6、stable-pretraining 0.1.7、Lightning 2.6.1、pymunk 7.3.1 |
| 硬件 | 1×H200 NVL，分区 `cse_24h`（24 h 上限） |

### 数据

- 18,685 episodes / 2,336,736 rows
- 内容摘要 `1959f284bf4795835b7b6108e53717c7a92292a3deb934b0cf8278fc1a686d27`
- H5 解压后 44 GB；官方 Lance 转换 14 GB
- 训练入口读取 `$STABLEWM_HOME/datasets/pusht_expert_train.h5`

### 模型

- 编码器：ViT-tiny，hidden 192、12 层、3 头、224 px、patch 14，从零初始化
- Predictor：ARPredictor，depth 6、heads 16、mlp_dim 2048、dim_head 64、dropout 0.1
- Action encoder：10 → 192；projector / pred_proj：MLP 192→192（hidden 2048）

### 训练

- AdamW lr `5e-5`、weight decay `1e-3`、batch 128、bf16
- gradient clip 1.0；SIGReg weight 0.09 / knots 17 / projections 1024
- global seed 3072；10 epochs；139,330 steps；约 10 it/s
- 最终验证：`validate/pred_loss_epoch` 0.003303、`sigreg_loss_epoch` 1.3222、
  `loss_epoch` 0.12230
- 作业 `2338783`：COMPLETED 2026-11-02 02:36:44，wall 4:17:03（含评测）

### 评测

- 环境 `swm/PushT-v1`，50 trials，seed 42
- goal offset 25、action budget 50
- CEM：300 candidates × 30 iterations、top 30、variance 1.0；
  horizon / receding horizon / action block 均为 5
- 两次评测使用同一批 50 个起点（两个作业日志打印的起始行索引完全一致）

| 模型 | 结果 | 作业 | wall | 评测耗时 |
| --- | --- | --- | --- | --- |
| 官方发布 checkpoint | **98.0%（49/50）** | 2338781 | 3:05 | 43.74 s |
| paper10（本次从零训练） | **96.0%（48/50）** | 2338783 | 4:17:03 | 47.61 s |

### 成本

- 有效 GPU 时间合计约 **7.4 GPU-hours**（1×H200 两个作业），0 API calls
- 失败前序作业 2338662 / 2338741 / 2338742 合计不足 3 分钟

## 三、产物位置（集群持久目录）

`$STABLEWM_HOME = /auxstore/cse/d11/data/s1155211002/FYP/lewm`，运行目录
`workflows/pusht_20261101T091612Z_3579888/`：

- Gate 凭据：`gates/gate{1,2,3,5}.json`（无 `gate4.json`）
- 数据核验：`audits/data_check/`
- 发布 checkpoint 评测：`audits/official_50/completed.json`、`audits/official_50/policy/env_*.mp4`
- paper10 训练：`audits/paper10/`（`summary.json`、`metrics.csv`、`step_trace.jsonl`）
- paper10 评测：`audits/paper10_eval/completed.json`、`audits/paper10_eval/policy/env_*.mp4`
- 权重：`checkpoints/paper10_pusht_20261101T091612Z_3579888/`（`weights_epoch_1..10.pt`、`last.ckpt`）
- 作业日志：`logs/gpu-official-2338781.out`、`logs/gpu-paper10-2338783.out`
- 失败证据：`audits/official_smoke.failed-2338662/`、`audits/official_smoke.failed-2338741/`、`status/submit_stage{1,2}.failed-*.txt`

大文件（权重、视频、H5/Lance、audits）**未进入 git**，由 `.gitignore` 排除。

## 四、运行日偏离与失败记录

| 作业 | 失败点 | 根因 | 处理 |
| --- | --- | --- | --- |
| 2338662 | `two_trial_cem_smoke`，32 s | Hydra 把相对 `config_path` 当包路径（`..config.eval`） | 设置 `run.__module__="__main__"` |
| 2338741 | PushT `_setup()`，1:48 | pymunk 6.11.1 缺 `Space.on_collision` | 升级 pymunk 7.3.1 |
| 2338742 | 训练入口，29 s | 同 Hydra 问题（`..config.train`） | 同修复 |

其他偏离：

- Gate3 与 Gate5 并行运行；`gate_require gate3` 与 `submit_stage2.sh` 中的
  Gate3/官方评测前置检查被移除，Gate1/Gate2 仍强制。
- Gate4 被移除，因此**没有经过资格验证的 resume 路径**。

## 五、服务器侧当前状态

- FYP le-wm 线没有 RUNNING / PENDING 作业。
- 服务器侧无待执行动作；后续训练或补测需要新的用户授权与判断。

## 六、建议远端优先回答的问题

1. Gate4 是否必须补跑，才能把本次结果记为「复现完成」？
2. 是否追加 2 个 seed 以匹配论文的三模型 variation？若追加，seed 与汇总口径是什么？
3. 对外报告采用哪个口径：10 epochs（论文 Appendix E）还是需要 100-epoch 探索？
4. `docs/PUSHT_CHPC_RUN_2026-11-01.md` 是否作为正式结果记录，是否需要补充
   H5/Lance 与 speed-mode 的 comparability 说明？
