import argparse
from wandb_train import main

# akt_lpkt 训练入口：AKT 主干 + LPKT 循环调控器（pykt/models/akt_lpkt.py）
# 本入口只运行题目级 AKT_LPKT；数据为 *_quelevel，每题的 KC 是 [K]。
# 用时字段缺失时模型不读取用时嵌入；原生 AKT/LPKT 的入口不变。
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2012")
    parser.add_argument("--model_name", type=str, default="akt_lpkt")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    # 可选训练覆盖项（不传则使用 configs/kt_config.json 的 train_config）
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=200)

    # ---- AKT 主干参数（与 wandb_akt_train.py 保持一致）----
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--kq_same", type=int, default=1)          # k/q 是否共享投影
    parser.add_argument("--final_fc_dim", type=int, default=512)
    parser.add_argument("--separate_qa", type=int, default=0)      # 是否用独立 qa 嵌入

    # ---- LPKT 调控器/模型特有参数 ----
    parser.add_argument("--n_causes", type=int, default=3)         # 成因后验 pi 的通道数 K
    parser.add_argument("--d_k", type=int, default=64)             # 概念状态维度，控制 4 GB 显存
    parser.add_argument("--n_phi", type=int, default=0)           # 过程特征维数，0 表示数据集不提供
    parser.add_argument("--input_level", type=str, default="question")

    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--add_uuid", type=int, default=1)

    args = parser.parse_args()
    params = vars(args)
    # d_k=-1 表示使用模型默认值（d_k=d_model），避免向构造函数传 -1
    if params["d_k"] == -1:
        params["d_k"] = None
    main(params)
