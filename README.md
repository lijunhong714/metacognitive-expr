# metacognitive-expr

基于 [pyKT-toolkit](https://github.com/pykt-team/pykt-toolkit) 构建的知识追踪实验仓库，集成 **AKT_LPKT** 模型（AKT 主干 + LPKT 循环调控器，模型定义见 `pykt/models/akt_lpkt.py`）。本文交代环境要求与 AKT_LPKT 训练/评估的复现方法。

## 环境要求

- Python 3.10（已在 3.10.18 验证）
- PyTorch 2.5.1（含对应 CUDA 运行时，需自行安装；`requirements.txt` 不包含 torch）
- 其余依赖安装：

```bash
pip install -r requirements.txt
pip install -e .   # 以可编辑方式安装本仓库，确保运行的是本地代码而非 PyPI 旧版包
```

## 复现方法

以 `algebra2005` 数据集为例。AKT_LPKT 要求数据集**同时含 questions、concepts、timestamps** 三个字段（题目 ID 用于难度嵌入与 q_matrix 查表，timestamps 用于 LPKT 式间隔时间索引）；纯概念数据集（如 `assist2015`，`num_q=0`）不适用。

### 1. 数据准备

将原始数据文件 `algebra_2005_2006_train.txt` 放入 `data/algebra2005/`，在 `examples` 目录下执行预处理：

```bash
cd examples
python data_preprocess.py --dataset_name algebra2005
```

该脚本完成原始数据解析与序列切分，产出 `train_valid_sequences.csv` / `test_sequences.csv` / `test_window_sequences.csv` 等序列文件，并自动更新 `configs/data_config.json` 中该数据集的条目（`dpath`、`num_q`、`num_c`、`maxlen`、`folds` 等）。

> 首次训练/评估时程序会自动完成 akt_lpkt 专属预处理（时间词表构建、`*_lpkt.pkl` 序列缓存、`qmatrix.npz` 题-概念关联矩阵生成），耗时较长属正常；之后复用缓存，速度恢复正常。

### 2. 训练

```bash
cd examples
python wandb_akt_lpkt_train.py --dataset_name algebra2005 --use_wandb 0 --add_uuid 0
```

- `--use_wandb 0`：关闭在线日志上报，仅本地运行，无需 wandb 登录或 key；`--add_uuid 0`：checkpoint 目录名不追加随机 uuid，便于按固定超参组合复现与复用。
- 其余超参不传时取脚本默认值：`seed=42`、`fold=0`、`learning_rate=1e-4`、`emb_type=qid`，以及 `d_model=256`、`d_ff=256`、`num_attn_heads=8`、`n_blocks=1`、`dropout=0.1`、`kq_same=1`、`final_fc_dim=512`、`separate_qa=0`、`n_causes=3`、`d_k=-1`（表示跟随 `d_model`）、`n_phi=0`。
- `batch_size` 默认强制为 64（模型含逐步循环调控器，开销高于 AKT），可用 `--batch_size` 覆盖；`num_epochs` 上限 200，可用 `--num_epochs` 覆盖。
- checkpoint 与训练配置保存在 `examples/saved_model/<数据集>_akt_lpkt_qid_saved_model_<超参组合>/` 下（内含 `config.json` 与 `qid_model.ckpt`，不入库）。
- 早停逻辑：验证集 AUC 连续 10 个 epoch 不再提升即自动停止。

### 3. 测试集评估

```bash
cd examples
python wandb_predict.py --save_dir "saved_model/<训练产出的 checkpoint 目录>" --use_wandb 0
```

- 脚本根据训练保存的 `config.json` 自动定位数据集、模型结构与 checkpoint（含 akt_lpkt 专属的 `num_at`/`num_it` 时间词表规模恢复），依次执行 chunk 分块协议（`testauc`/`testacc`）和滑动窗口协议（`window_testauc`/`window_testacc`）。
- **AKT_LPKT 为概念级模型，最终报告口径看 `window_testauc` / `window_testacc`**（以脚本打印的 "Final reported metric" 提示为准），不要混用不同协议指标。
- 大数据集的 `test_window_sequences.csv` 体积很大（algebra2005 约 810MB），多个模型评估需串行执行，并行反序列化会内存不足。

### 4. 冒烟验证

小配置 1 epoch（`--d_model 64 --d_ff 64 --num_attn_heads 4 --final_fc_dim 128 --batch_size 32 --num_epochs 1 --learning_rate 1e-3`）参考结果：validauc 0.8998，testauc 0.8992，window_testauc 0.8993（algebra2005, fold 0）。
