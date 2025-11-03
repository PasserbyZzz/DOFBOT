import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from torch.distributions import Normal, Categorical
from flexible_networks import FlexibleMLP, FlexibleMDN, FlexibleLSTM, mdn_loss_fn

from flexible_train_utils import train_model

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    data = df.values
    return data

# ⚡ 数据分割
def split_data(data, mode='fk'):
    if mode == 'fk':
        x = data[:, 0:2]      # q1..q5
        y = data[:, 2:4]      # x, y, z
        # y = dataset[:, 5:12]     # x, y, z, a, b, c, d
    elif mode == 'ik':
        x = data[:, 2:4]      # x, y, z
        # X = dataset[:, 5:12]     # x, y, z, a, b, c, d
        y = data[:, 0:2]      # x, y, z, a, b, c, d
    else:
        raise ValueError("Mode should be 'fk' or 'ik'")
    return train_test_split(x, y, test_size=0.2, random_state=42)

def get_model(model_type, input_dim, output_dim, hidden_layers):
    if model_type == 'mlp':
        return FlexibleMLP(input_dim=input_dim, output_dim=output_dim, hidden_layers=hidden_layers)
    elif model_type == 'mdn':
        num_mixtures = 5  # 你可以改成参数
        return FlexibleMDN(input_dim=input_dim, output_dim=output_dim, hidden_layers=hidden_layers, num_mixtures=num_mixtures)
    elif model_type == 'lstm':
        hidden_size = 128  # 例子参数，可自定义
        num_layers = len(hidden_layers)
        return FlexibleLSTM(input_dim=input_dim, output_dim=output_dim, hidden_size=hidden_size, num_layers=num_layers)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Flexible Models for FK or IK")
    parser.add_argument('--csv_path', type=str, default='dataset/two_link_arm_data.csv')
    parser.add_argument('--mode', type=str, choices=['fk', 'ik'], default='ik')
    parser.add_argument('--model_type', type=str, choices=['mlp', 'mdn', 'lstm'], default='mlp')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[512, 512, 256, 128])
    args = parser.parse_args()

    data = load_dataset(args.csv_path)
    X_train_np, X_test_np, y_train_np, y_test_np = split_data(data, mode=args.mode)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train_np, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test_np, dtype=torch.float32).to(device)

    model = get_model(args.model_type, input_dim=X_train.shape[1], output_dim=y_train.shape[1], hidden_layers=args.hidden_layers)

    # 训练，传入模型结构信息
    model_info = {
        'hidden_layers': args.hidden_layers,
    }
    train_model(
        model,
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        model_type=args.model_type,
        mode=args.mode,
        model_info=model_info
    )