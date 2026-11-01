# PushT 依赖与数据接口审计

状态：2026-11-01，仅完成官方源码、发布包与 Hub 元数据核查，以及 Linux/Python 3.10 依赖解析。本文件给出**可复查的候选环境**，不是作者原始环境，也不是已经通过真实 PushT 训练/评估的环境。实际运行验收见[复现协议](PUSHT_REPRO_PROTOCOL.md)与[本地验证记录](PUSHT_LOCAL_VALIDATION.md)。

- [固定来源](#固定来源)
- [依赖环境与验证边界](#依赖环境与验证边界)
- [调度与恢复](#调度与恢复)
- [数据格式与路径](#数据格式与路径)
- [官方权重与配置](#官方权重与配置)
- [服务器准入条件](#服务器准入条件)

## 固定来源

机器可读身份、文件大小和哈希见 [PUSHT_SOURCE_MANIFEST.json](PUSHT_SOURCE_MANIFEST.json)。Hub 大文件的 SHA256 来自实时官方 LFS 元数据；本次没有下载或逐字节核验权重及 13.14 GB 数据压缩文件。

| 对象 | 固定修订 | 本次依据 |
| --- | --- | --- |
| LeWM | `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` | [官方训练、评估与 YAML](https://github.com/lucas-maes/le-wm/tree/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac)，2026-05-22 提交 |
| stable-worldmodel | `abdced49809d5eae38e24b27dc7b635c502c4812` | [官方源码](https://github.com/galilai-group/stable-worldmodel/tree/abdced49809d5eae38e24b27dc7b635c502c4812)，2026-05-20；选为 LeWM 固定提交之前的同期候选 |
| stable-pretraining | `bce7c8b35a62399d0529068ed7fea5dd2ce9021e` / `v0.1.7` | [官方源码](https://github.com/galilai-group/stable-pretraining/tree/bce7c8b35a62399d0529068ed7fea5dd2ce9021e)；0.1.7 官方 wheel 的 Manager、Module、scheduler、vit_hf 与此源码按 LF 归一后逐字相同 |
| 官方 PushT 权重 | `22b330c28c27ead4bfd1888615af1340e3fe9052` | [固定 Hub 模型树](https://huggingface.co/quentinll/lewm-pusht/tree/22b330c28c27ead4bfd1888615af1340e3fe9052) |
| 官方 PushT 数据 | `655cd446b9929369d7d406001da85c15d1457850` | [固定 Hub 数据树](https://huggingface.co/datasets/quentinll/lewm-pusht/tree/655cd446b9929369d7d406001da85c15d1457850) |

时间相邻只能提供选择候选的依据，不能证明作者使用这两个依赖提交。上游 LeWM 没有发布环境锁；作者训练时的 Python、torch、CUDA、依赖修订与实际 scheduler 参数仍未知。

## 依赖环境与验证边界

[repro-pusht.in](../requirements/repro-pusht.in) 显式固定计算、模型、数据与 PushT 仿真所需的直接依赖。[repro-pusht.txt](../requirements/repro-pusht.txt) 是 `uv 0.9.27` 为 **Linux x86_64-manylinux_2_28 / CPython 3.10** 解析的 137 包完整运行时版本锁。核心候选是 torch 2.8.0、torchvision 0.23.0、Lightning 2.6.1、transformers 4.51.3、NumPy 2.2.6、pyarrow 20.0.0、lancedb 0.30.0、pylance 4.0.0。所有显式候选版本已从官方 PyPI 元数据核实存在并接受 Python 3.10。

这次解析未安装完整环境，未下载 GPU 训练栈，未验证服务器驱动或 Linux 原生扩展。运行时版本锁也没有固定所有 wheel 字节或隔离构建工具。安装时仍须保存安装报告、`pip freeze`、平台/驱动信息与所用 wheel 哈希。锁中 Linux 的 torch 会带 CUDA 12.8 系列运行时依赖；能否使用必须以服务器驱动和实际 import/smoke 检查为准。

在仓库根目录重新解析的命令如下；新解析可能改变间接依赖，不能覆盖已执行实验使用的锁：

```bash
uv pip compile requirements/repro-pusht.in --python-version 3.10 \
  --python-platform x86_64-manylinux_2_28 --no-strip-extras --no-header \
  --output-file requirements/repro-pusht.txt
```

解析成功，有一条非致命警告：Minari 的旧元数据请求 `typer[all]`，解析出的 typer 0.27.3 不再声明该 extra。完整环境尚未安装，不能据解析成功宣称 CLI 全部可用。无需安装 `stable-worldmodel[env]` 带来的 Craftax、OGBench、MuJoCo 等无关环境；PushT 使用明确列出的 pygame/pymunk/OpenCV 依赖。

**不能用 `stable-worldmodel==0.0.6` 替代 Git SHA。** 2026-03-16 发布的同名 PyPI wheel 没有当前所需的 Lance/`load_dataset` 接口，而 5 月候选源码的 pyproject 仍写 0.0.6。同一版本字符串不能证明文件相同。此旧 wheel 已下载核验，SHA256 为 `90601307b3430436d617a5134579089414a1f98eab1dfb6aeabf36ea7f46ff4d`，作为排除证据保留在清单中。候选安装必须核验 `direct_url.json` 中的实际 Git commit。

| 接口 | 已读到的源码事实 | 尚未完成的检查 |
| --- | --- | --- |
| `swm.data.load_dataset` / `HDF5Dataset` | 支持 LeWM train 的 Lance 入口；HDF5 类接受 eval 的 name/cache_dir/keys_to_cache | 全量真实数据与 Linux worker 读取 |
| `swm.World.evaluate` | 接受 dataset、episodes_idx、start_steps、goal_offset、eval_budget、callables、video | 真实 PushT 50 条轨迹与 GPU CEM |
| `swm.PlanConfig` / `WorldModelPolicy` / `CEMSolver` | 固定源码存在对应类及上游调用参数 | 上游完整评估调用链运行 |
| `spt.Module` / scheduler | 0.1.7 wheel 与固定源码的有关文件相同；真实步进见下节 | 完整 LeWM/数据/GPU 训练 |
| `spt.Manager` | 支持完整 checkpoint 传入与 `weights_only=False` | 真实数据多 worker、SIGReg 随机投影与 CUDA 中断恢复 |
| `spt.backbone.utils.vit_hf` | 存在 tiny/patch14/image224 接口；tiny 为 192 维、12 层、3 个注意力头 | 官方权重 strict-load 与完整 planning |

## 调度与恢复

依据 [spt module.py](https://github.com/galilai-group/stable-pretraining/blob/bce7c8b35a62399d0529068ed7fea5dd2ce9021e/stable_pretraining/module.py#L280) 和 [lr_scheduler.py](https://github.com/galilai-group/stable-pretraining/blob/bce7c8b35a62399d0529068ed7fea5dd2ce9021e/stable_pretraining/optim/lr_scheduler.py#L73)，Module 使用 manual optimization，每次 `opt.step()` 后直接 `scheduler.step()`。它不按 LeWM 的 `interval: epoch` 字段改为每轮一步。默认 `LinearWarmupCosineAnnealingLR` 使用：

```text
max_steps = trainer.estimated_stepping_batches
warmup_steps = max(1, int(0.01 * max_steps))
warmup_start_lr = 0
eta_min = 0
```

因此 10 轮与 100 轮目标会从 warmup 开始采用不同日程；用 `max_epochs=10` 跑完再改为 100 不是原 100 轮连续训练。`max_steps`、`limit_train_batches`、batch 数、设备数、累积频率也可能改变 Lightning 推算的日程长度。任何截断验收必须保存实际 `estimated_stepping_batches`、scheduler 的 max/warmup、每步 LR 和 global_step，不能只核对 YAML。spt 的官方 Lightning patch 会转存 gradient clip 参数以供手动优化使用，此 patch 是否成功也须在实际环境核验。

[Manager 源码](https://github.com/galilai-group/stable-pretraining/blob/bce7c8b35a62399d0529068ed7fea5dd2ce9021e/stable_pretraining/manager.py#L469) 要求手工传入的 `ckpt_path` 为存在的绝对路径。其 `weights_only=True` 默认值控制 `Trainer.fit` 的反序列化加载策略，不能仅从这个名字推断是否恢复了优化器；旧 Lightning 不接受此参数时还会回落并警告。审计 wrapper 应显式指定 `weights_only=False`，核查完整 checkpoint 的模型、optimizer、scheduler、loops/global_step 与 callbacks，并以中断/连续对照证明恢复。官方 `SaveCkptCallback` 的 `weights_epoch_N.pt` 只是模型 state_dict，不含优化器、scheduler 或所有随机状态，不能当训练恢复文件。

另外，完整 Lightning checkpoint 并不自动证明外部 `torch.Generator`、随机数据顺序、worker RNG、SIGReg 投影和 CUDA RNG 都按所需顺序恢复。CPU 小模型的恢复检查与真实多 worker/GPU 恢复是不同验收层次。

## 数据格式与路径

固定 LeWM `config/train/data/pusht.yaml` 写 `pusht_expert_train.lance`，但 `eval.py` 直接构造 `HDF5Dataset('pusht_expert_train')`。固定 Hub 数据只列 `pusht_expert_train.h5.zst`，不是 `.tar.zst`。应使用 `unzstd`/`zstd -d` 解压，而不是按 README 的 tar 示例处理这个文件。

候选 swm 的 `get_cache_dir(..., sub_folder='datasets')` 和 `wm.utils` 规定：

```text
$STABLEWM_HOME/
├── datasets/pusht_expert_train.h5       原始评估数据
├── datasets/pusht_expert_train.lance/   显式转换的训练数据
└── checkpoints/<run>/                  config.json、weights_epoch_N.pt
```

README 的 `~/.stable-wm`、根目录 HDF5 与 `_object.ckpt` 路由均不能覆盖当前源码的实际路径。`LOCAL_DATASET_DIR` 传给 loader 的是缓存根目录，loader 还会附加 `datasets/`。

固定 swm 已提供[官方转换函数](https://github.com/galilai-group/stable-worldmodel/blob/abdced49809d5eae38e24b27dc7b635c502c4812/stable_worldmodel/data/utils.py#L235)。以下是**待在服务器执行并验收**的调用，不表示本次已经转换数据：

```python
import os
from pathlib import Path
from stable_worldmodel.data import convert
root = Path(os.environ["STABLEWM_HOME"]) / "datasets"
convert(str(root / "pusht_expert_train.h5"),
        str(root / "pusht_expert_train.lance"),
        source_format="hdf5", dest_format="lance", mode="error", jpeg_quality=95)
```

使用 `mode='error'` 防止默认 append 重复加入轨迹。此转换逐 episode 读取、重建 episode/step 索引，LanceWriter 将非图像数值转 float32，并以 JPEG quality 95 编码图像。**有损 JPEG 不保证像素相同；数值 dtype 变化也须记录。** 服务器需核对原始压缩文件哈希、解压 HDF5 哈希、转换参数、episode 长度/offset、全行顺序、action/state/proprio 数值误差和图像差异，再建立 split/评估身份。不能把训练 `.lance` 与评估 `.h5` 视为同一字节资产，不能声称转换已还原作者实际训练的 Lance 文件。

## 官方权重与配置

固定 Hub 配置已实际下载并保存其 SHA256。ViT tiny/patch14/image224、history 3、latent 192、6 层 predictor、16 heads、MLP 2048、action input 10 等尺寸与上游 YAML 的 resolved 预期一致；action 的 10 仍须由真实二维 action × frameskip 5 验证。

实际 `config.json` 含有嵌套 Hydra `_target_`，顶层是 `stable_worldmodel.wm.lewm.LeWM`，predictor 是 `stable_worldmodel.wm.lewm.module.Predictor`。它不是能直接传给本仓库构造函数的纯参数字典。`norm_fn` 也是带 `_partial_` 的 BatchNorm1d 配置。要用固定上游的 `jepa.JEPA` / `module.ARPredictor`，必须显式映射这些 target，保留所有维度，再 `load_state_dict(strict=True)`；不得直接 `ARPredictor(**cfg['predictor'])` 或把 norm_fn 字典当 callable。

候选 swm 的 HF 加载函数会访问 `/resolve/main/`，没有修订参数；**不要直接使用浮动远程 policy 作为固定模型**。应按清单中的 revision 下载配置和权重、验证大小/SHA，再用本地明确 `.pt` 路径与相邻 `config.json` 加载。源码 `load_pretrained` 支持本地 `.pt`，目录中有多个 `.pt` 时会因歧义报错，所以每轮评估应指定具体 epoch 文件。swm 的 LeWM 和本仓库 JEPA 部分推理实现存在差异；尺寸一致不能代替 strict-load、模型实现身份和原入口 planning 验证。

## 服务器准入条件

环境解析、源码审计与 CPU 夹具检查不足以通过运行 Gate1–5。服务器首先应在隔离 Linux/Python 3.10 环境按锁安装，核验 swm Git SHA、spt wheel 身份、import、真实 dataset batch、模型 strict-load、scheduler 实参及 PushT reset/evaluate 接口。现已补齐数据/函数一致性与恢复比较入口，命令见 [服务器交接](PUSHT_SERVER_STEPS.md)。保存实际安装报告，不能把本机 Windows CPU 测试冒称 Linux GPU 验收。

随后分别完成官方权重的原始 50 起点评估和同一完整日程下的 resume 对照；仍需用户给定服务器资源与执行预算。此准备阶段不提交集群作业、不分配 GPU、不运行新训练，也不把作者 96.0% 当本项目已得到的结果。
