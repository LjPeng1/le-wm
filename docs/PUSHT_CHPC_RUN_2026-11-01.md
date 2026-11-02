# CHPC Push-T 复现运行结果（2026-11-01 / 11-02）

运行 ID：`pusht_20261101T091612Z_3579888`
代码基线：`0dd9b96297ff2b4293e94a5f63eb5ac92d37d7cd`（本次提交把运行日当时的
未提交修复一并固化；Gate 凭据中记录的 `commit_sha` 仍是 `0dd9b96`）

本文件记录 CHPC（CUDA 12.9 / H200 NVL / Linux x86_64 / Python 3.10、
torch 2.8.0+cu128、pymunk 7.3.1）上的实际运行结果。权重、视频与逐门凭据留在
集群持久目录，不进 git（`.gitignore` 已排除 `*.pt`/`*.ckpt`/`*.mp4`）。

## 结果摘要

| Gate | 内容 | 作业 | 状态 | 结果 |
| --- | --- | --- | --- | --- |
| Gate1 | 数据解压与逐字节核验 | sandbox | passed 2026-11-01 21:54 | 18,685 episodes / 2,336,736 rows；内容摘要 `1959f284…686d27` |
| Gate2 | 官方资产与模型函数检查 | sandbox | passed 2026-11-01 21:54 | 权重/归档哈希一致，官方资产加载检查通过 |
| Gate3 | 发布 checkpoint 官方 50 起点 CEM | **2338781** | COMPLETED 2026-11-01 22:22:46 | **98.0 %（49/50）**，评测 43.74 s |
| Gate4 | 中断/恢复等价性资格验证 | — | **按用户指示跳过** | 无 `gate4.json`；被抢占时无合格 resume 路径 |
| Gate5 | 从零训练 paper10（10 epochs）+ 评测 | **2338783** | COMPLETED 2026-11-02 02:36:44 | **96.0 %（48/50）**，训练 139,330 steps，评测 47.61 s |

两次评测使用同一批 50 个起点（两个作业日志打印的起始行索引完全一致）。
发布 checkpoint 得 98.0 %，论文报告约 96 %；从零训练 10 epochs 的模型在同一
评测路径上得 96.0 %。

## Gate5 训练摘要

- 配置：ViT-tiny（hidden 192 / 12 层 / 3 头 / 224 px / patch 14），从零初始化
- 10 epochs，139,330 steps，约 10 it/s，总墙钟约 4 h 17 min（job 2338783，
  1×H200，`cse_24h`）
- 最终验证：`validate/pred_loss_epoch` = 0.003303，`validate/sigreg_loss_epoch`
  = 1.3222，`validate/loss_epoch` = 0.12230
- 检查点：`weights_epoch_1..10.pt` + `last.ckpt`（epoch 10 与 `last.ckpt` 为终止状态）

## 运行日修复（本次提交包含）

1. **Hydra 配置解析**（`scripts/repro_pusht/evaluate.py`、`train.py`）：包装脚本
   `import eval`/`train` 后调用 `run()` 时，`run.__module__` 不是 `__main__`，
   Hydra 把相对 `config_path` 当包路径解析，报
   `Primary config module '..config.eval' not found`。修复为先设置
   `upstream.run.__module__ = "__main__"`（含 `__wrapped__`）。
2. **pymunk 版本**（`requirements/repro-pusht.txt`）：锁文件里的 6.11.1 缺少
   `stable-worldmodel` PushT 环境使用的 `Space.on_collision`（pymunk 7 API），
   升级为 7.3.1。
3. **Speed mode（用户指示「把能跳过的全跳过」）**：
   - Gate4 从 `jobs/03_resume_train_eval_gpu.sbatch` 移除；
   - `PUSHT_DATA_REHASH=skip`：训练入口不再重读 58 GB，改为校验内容清单自身
     摘要 + 各产物 size/mtime，并如实记录 `rehash: "skipped_stat_bound"`；
   - `PUSHT_EXPECTED_H5_SHA256`：评测入口不再重读 44 GB H5，记录 Gate1 审计过的
     摘要 + 当前 size/mtime；
   - 两个环境变量只由 `jobs/03_...` 导出，默认（未设置）仍是完整重哈希，
     Gate3 走默认路径。
4. **并行提交**：Gate3 与 Gate5 并行运行；`gate_require gate3` 与
   `submit_stage2.sh` 中的 Gate3/官方评测前置检查被移除，Gate1/Gate2 仍强制。

## 失败的前序作业（保留为证据，已删除）

| 作业 | 失败点 | 根因 |
| --- | --- | --- |
| 2338662 | `two_trial_cem_smoke`，32 s | Hydra `..config.eval` 包路径（见修复 1） |
| 2338741 | PushT `_setup()`，1 min 48 s | pymunk 6.11.1 缺 `Space.on_collision`（见修复 2） |
| 2338742 | 训练入口，29 s | Hydra `..config.train` 包路径（见修复 1） |

失败产物保留在 `audits/official_smoke.failed-{2338662,2338741}/` 与
`status/submit_stage{1,2}.failed-*.txt`，未删除。

## 产物位置（集群）

根目录 `$STABLEWM_HOME = /auxstore/cse/d11/data/s1155211002/FYP/lewm`，
运行目录 `workflows/pusht_20261101T091612Z_3579888/`：

- Gate 凭据：`gates/gate{1,2,3,5}.json`
- 数据核验：`audits/data_check/`
- 发布 checkpoint 评测：`audits/official_50/completed.json`、`audits/official_50/policy/env_*.mp4`
- paper10 训练：`audits/paper10/`（`summary.json`、`metrics.csv`、`step_trace.jsonl`）
- paper10 评测：`audits/paper10_eval/completed.json`、`audits/paper10_eval/policy/env_*.mp4`
- 权重：`checkpoints/paper10_pusht_20261101T091612Z_3579888/`
- 作业日志：`logs/gpu-official-2338781.out`、`logs/gpu-paper10-2338783.out`

## 边界与说明

- Gate4 被跳过：本次训练**未验证**抢占/中断后的恢复等价性；若作业被抢占或
  超时，只能从头重跑。这是运行日用户明确接受的取舍。
- 50 起点 CEM 结果本身有抽样波动，98.0 % 与 96.0 % 的差异不应解读为模型优劣
  的严格结论；两者与论文约 96 % 的量级一致。
- 运行日 Gate3 记录的 `helpers` 哈希与 HEAD 不同，因为 helper 文件在运行前已
  修改；Gate1/Gate2 凭据产生于 pymunk 6.11.1，但它们不使用 pymunk，Gate3/Gate5
  运行时身份记录的是 7.3.1。
