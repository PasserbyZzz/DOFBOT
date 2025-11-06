# ⚡ FK, IK, Sequence Learning 训练流程
import time, numpy as np, torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
import pandas as pd
from utils_learn.flexible_networks import FlexibleMLP
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# ---------- 工具函数 ----------
def select_cols(data_df, names):
    # 从 DataFrame 选取列并返回 ndarray
    return data_df[names].values


def split_data(data_df, in_cols, out_cols):
    # 从 DataFrame 拆出输入/输出，并按 8:2 划分 train/test
    X = select_cols(data_df, in_cols)
    Y = select_cols(data_df, out_cols)
    
    return train_test_split(X, Y, test_size=0.2, random_state=42)

def split_data_analytic(data_df, in_cols, out_cols, dofbot):
    # 从 DataFrame 拆出输入/输出，并按 8:2 划分 train/test
    X = select_cols(data_df, in_cols)
    Y = select_cols(data_df, out_cols)
    # 解析解
    Y_analytic = analytic_fk(X, dofbot=dofbot)
    # 残差
    Y_residual = Y - Y_analytic

    return train_test_split(X, Y_residual, Y_analytic, test_size=0.2, random_state=42)


def compute_fk_loss(y_pred, y_true, w_pos=0.9, w_ori=0.1):
    """
    FK 损失
    参数
    ----
    y_pred : [B, 12]  预测位姿（归一化）
    y_true : [B, 12]  真值位姿（仅监控）
    w_pos  : 位置误差权重
    w_ori  : 姿态误差权重
    返回
    ----
    loss : 标量张量
    info : dict
    """
    # 1. 位置
    loss_pos = F.mse_loss(y_pred[:, :3], y_true[:, :3])

    # 2. 旋转矩阵误差（Frobenius 距离）
    R_pred = y_pred[:, 3:]  # [B, 9]
    R_true = y_true[:, 3:]  # [B, 9]
    loss_ori = F.mse_loss(R_pred, R_true)
    # 3. 加权
    # loss = w_pos * loss_pos + w_ori * loss_ori # 加权总损失
    # 4. 监控
    with torch.no_grad():
        mae = torch.mean(torch.abs(y_pred - y_true)).item() # 整体 MAE
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item() # 整体 RMSE
        pos_error = torch.norm(y_pred[:, :3] - y_true[:, :3], dim=1).mean().item() # 位置误差
        ori_error = torch.norm(R_pred - R_true, dim=1).mean().item() # 姿态误差

    loss = F.mse_loss(y_pred, y_true) # 整体 MSE 作为损失
    return loss, {'mae': mae, 'rmse': rmse, 'position_error': pos_error, 'orientation_error': ori_error}


def compute_ik_loss(q_pred, q_true,
                    pose_true=None, fk_ref=None,
                    w_pos=0.9, w_ori=0.1):
    """
    IK 损失
    参数
    ----
    q_pred : [B, 5]  预测关节角（归一化）
    q_true : [B, 5]  真值关节角（仅监控）
    pose_true:[B, 12] 真值末端矩阵 [x,y,z | 9-elements-of-R] （通过训练好的 FK 网络计算）
    fk_ref : 冻结的 FK 网络，输入 q 输出 [B,12]
    w_pos  : 位置误差权重
    w_ori  : 姿态误差权重

    返回
    ----
    loss : 标量张量
    info : dict
    """
    # ---- 1. 关节角监控（无梯度） ----
    with torch.no_grad():
        joint_mae = torch.mean(torch.abs(q_pred - q_true)).item() # 关节 MAE
        joint_rmse = torch.sqrt(torch.mean((q_pred - q_true) ** 2)).item() # 关节 RMSE

    # ---- 2. 无矩阵监督 → 退化为关节 MSE ----
    if pose_true is None or fk_ref is None:
        loss = F.mse_loss(q_pred, q_true) # 无 FK 监督：关节 MSE 作为损失
        info = {'joint_mae': joint_mae, 'joint_rmse': joint_rmse}
        return loss, info

    # ---- 3. FK 监督 → 末端矩阵损失 ----
    pred_mat = fk_ref(q_pred)  # [B,12]

    # 3.1 位置损失
    loss_pos = F.mse_loss(pred_mat[:, :3], pose_true[:, :3])

    # 3.2 旋转矩阵损失（Frobenius）
    loss_ori = F.mse_loss(pred_mat[:, 3:], pose_true[:, 3:])
    # # 3.3 加权总损失
    # loss = w_pos * loss_pos + w_ori * loss_ori

    loss = F.mse_loss(pred_mat, pose_true) # FK 监督：末端矩阵 MSE 作为损失

    # ---- 4. 监控指标 ----
    with torch.no_grad():
        pos_err = torch.norm(pred_mat[:, :3] - pose_true[:, :3], dim=1).mean().item() # 位置误差
        ori_err = torch.norm(pred_mat[:, 3:] - pose_true[:, 3:], dim=1).mean().item() # 姿态误差

    info = {'joint_mae': joint_mae,
            'joint_rmse': joint_rmse,
            'position_error': pos_err,
            'orientation_error': ori_err}

    return loss, info


def plot_training_curves(history: dict, save_path: str):
    # 保存训练/测试曲线
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(history['train'], label=f"Train {history['metric']}")
    plt.plot(history['test'], label=f"Test  {history['metric']}")
    plt.xlabel('Epoch');
    plt.ylabel(history['metric']);
    plt.title(f"{history['metric']} Curve")
    plt.legend();
    plt.tight_layout();
    plt.savefig(save_path, dpi=300);
    plt.close()
    print(f'📈 曲线已保存 → {save_path}')

def analytic_fk(q, dofbot):
    # 正运动学解析
    q = np.array(q)
    if q.shape[1] == 10:  # sin/cos 展开
        q_angles = np.arctan2(q[:, ::2], q[:, 1::2])  # [B,5]
    elif q.shape[1] == 5:  # 直接角度
        q_angles = q
    else:
        raise ValueError(f"输入维度错误: q.shape={q.shape}, 期望为 [B,5] 或 [B,10]")

    # 计算 FK
    B = q_angles.shape[0]
    pose_list = []

    for i in range(B):
        T = dofbot.fkine(q_angles[i])  # 正运动学计算
        Tm = np.array(T.A)  # 取出4x4矩阵
        xyz = Tm[:3, 3]  # 末端位置
        rot = Tm[:3, :3].ravel()  # 展平旋转矩阵 (nx,ny,nz, ox,oy,oz, ax,ay,az)
        pose = np.hstack([xyz, rot])
        pose_list.append(pose)

    # 返回 12 维：xyz + I9
    pose = np.vstack(pose_list)  # [B, 12]
    return pose

# ---------- 唯一入口 ----------
def train_dofbot_model(data_path,
                       model_type='mlp',  # 'mlp' | 'mdn' | 'lstm'
                       mode='fk',  # 'fk'  | 'ik'
                       in_cols=None,  # list[str] 仅 fk 有效
                       out_cols=None,  # list[str] 仅 ik 有效
                       epochs=1000,
                       lr=1e-3,
                       min_lr=1e-3,
                       num_mixtures=5,
                       hidden_layers=[100, 30],
                       seq_len=10,
                       fk_path=None,
                       fk_hidden_layers=None,
                       w_pos=0.9,
                       w_ori=0.1,
                       use_analytic_fk=False,
                       dofbot=None
                       ):
    # 数据加载、模型构造、训练、评估、保存
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    # 0. 确定输入/输出列名
    data_df = pd.read_csv(data_path)
    if mode == 'fk':
        in_cols = in_cols or ['q1_sin', 'q1_cos', 'q2_sin', 'q2_cos', 'q3_sin', 'q3_cos', 'q4_sin', 'q4_cos', 'q5_sin', 'q5_cos']
        out_cols = out_cols or ['x', 'y', 'z', 'nx', 'ny', 'nz', 'ox', 'oy', 'oz', 'ax', 'ay', 'az']  # 默认 xyz+orn
    else:  # ik
        in_cols = in_cols or ['x', 'y', 'z', 'nx', 'ny', 'nz', 'ox', 'oy', 'oz', 'ax', 'ay', 'az']
        out_cols = out_cols or ['q1_sin', 'q1_cos', 'q2_sin', 'q2_cos', 'q3_sin', 'q3_cos', 'q4_sin', 'q4_cos', 'q5_sin', 'q5_cos']
        
        # 加载冻结 FK （基于已训练的FK模型监督训练）
        fk_ref = FlexibleMLP(len(out_cols), len(in_cols), hidden_layers=fk_hidden_layers, dropout=0.0,
                             activation='ReLU', block_type='res',
                             num_blocks=1).to(device)
        fk_ref.load_state_dict(torch.load(fk_path, map_location=device, weights_only=True))
        fk_ref.eval()
        for p in fk_ref.parameters():
            p.requires_grad = False

    # 1. 准备数据
    if use_analytic_fk and mode == 'fk':
        x_train, x_test, y_train, y_test, y_analytic_train, y_analytic_test = split_data_analytic(data_df, in_cols, out_cols, dofbot)   
        x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
        y_analytic_train = torch.tensor(y_analytic_train, dtype=torch.float32, device=device)
        x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
        y_test = torch.tensor(y_test, dtype=torch.float32, device=device)
        y_analytic_test = torch.tensor(y_analytic_test, dtype=torch.float32, device=device)
    else:
        x_train, x_test, y_train, y_test = split_data(data_df, in_cols, out_cols)
        x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
        x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
        y_test = torch.tensor(y_test, dtype=torch.float32, device=device)
    
    # 2. 输出目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results/learn_model") / f"{model_type}_{mode}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. 构造模型
    if model_type == 'mlp':
        model = FlexibleMLP(x_train.shape[1], y_train.shape[1], 
                            hidden_layers,
                            dropout=0.0,  # 0.0% dropout
                            activation='ReLU',
                            block_type='res',
                            num_blocks=1).to(device)  # 换激活函数
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) # Adam 优化器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( # 学习率调度器
            opt, T_max=epochs, eta_min=min_lr)  # eta_min 最低学习率
        history = {'train': [], 'test': [], 'metric': 'MSE'}
        best_test = np.inf
        patience = 0

        for epoch in range(epochs):
            model.train()
            opt.zero_grad()
            y_pred = model(x_train)
            
            # 计算损失
            if not use_analytic_fk:
                if mode == 'fk':
                    loss, info = compute_fk_loss(y_pred, y_train, w_pos=w_pos, w_ori=w_ori)
                else:  # ik 
                    loss, info = compute_ik_loss(y_pred, y_train, pose_true=x_train, fk_ref=fk_ref, w_pos=w_pos, w_ori=w_ori)
            else:
                if mode == 'fk':
                    # 加上解析解
                    y_pred = y_pred + y_analytic_train
                    loss, info = compute_fk_loss(y_pred, y_train + y_analytic_train, w_pos=w_pos, w_ori=w_ori)
                else:  # ik
                    loss, info = compute_ik_loss(y_pred, y_train, pose_true=x_train, fk_ref=fk_ref, w_pos=w_pos, w_ori=w_ori)
            
            # 反向传播
            loss.backward()
            opt.step()
            scheduler.step()  # 退火
            history['train'].append(loss.item())

            # 每 epoch 记录测试
            with torch.no_grad():
                model.eval()
                if not use_analytic_fk:
                    if mode == 'fk':
                        test_loss, test_info = compute_fk_loss(model(x_test), y_test, w_pos=w_pos, w_ori=w_ori)
                    else:  # ik
                        test_loss, test_info = compute_ik_loss(model(x_test), y_test, pose_true=x_test, fk_ref=fk_ref, w_pos=w_pos, w_ori=w_ori)
                else:
                    if mode == 'fk':
                        # 加上解析解
                        y_test_pred = model(x_test) + y_analytic_test
                        test_loss, test_info = compute_fk_loss(y_test_pred, y_test + y_analytic_test, w_pos=w_pos, w_ori=w_ori)
                    else:  # ik
                        test_loss, test_info = compute_ik_loss(model(x_test), y_test, pose_true=x_test, fk_ref=fk_ref, w_pos=w_pos, w_ori=w_ori)
            
            history['test'].append(test_loss.item())
            if (epoch + 1) % max(1, epochs // 100) == 0 or epoch == epochs - 1:
                print(
                    f"[{model_type.upper()} {mode.upper()}] Epoch {epoch + 1}/{epochs} | Train: {loss.item():.6f} | Test: {test_loss.item():.6f} | Test info: {test_info}")

            # 提前停止
            if test_loss < best_test:
                best_test = test_loss
                patience = 0
                torch.save(model.state_dict(), out_dir / 'best_model.pt')
            else:
                patience += 1
                if patience >= 50:
                    print(f"Early stop at epoch {epoch + 1}")
                    break

        torch.save(model.state_dict(), out_dir / 'model.pt')

    # 3. 画图
    curve_png = out_dir / f"training_curve_{history['metric'].replace(' ', '_')}.png"
    # 去掉 nan 再画
    clean = {}
    for k in ['train', 'test']:
        ser = pd.Series(history[k])  # 含 nan 的序列
        ser = ser.ffill()  # 前一有效帧填充
        clean[k] = ser.values  # 转回 ndarray
    clean['metric'] = history['metric']
    plot_training_curves(clean, str(curve_png))

    print(f"✅ 训练完成！模型与曲线已保存到 → {out_dir}")
    return model, out_dir, str(out_dir / 'best_model.pt')  # ① 返回 FK 模型路径


if __name__ == "__main__":
    # 训练正逆运动学模型
    fk_model, fk_dir, fk_path = train_dofbot_model(data_path='../dataset/60000/dofbot_fk_60000_norm.csv',
                                                   model_type='mlp', mode='fk',
                                                   fk_out_cols=['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
                                                   epochs=2000, lr=1e-3, hidden_layers=[128, 128, 64])
    ik_model, ik_dir, ik_path = train_dofbot_model(data_path='../dataset/60000/dofbot_fk_60000_norm.csv',
                                                   model_type='mlp', mode='ik',
                                                   ik_in_cols=['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
                                                   epochs=2000, lr=1e-3, hidden_layers=[128, 128, 64], fk_path=fk_path,
                                                   fk_hidden_layers=[128, 128, 64])
    # # 训练正逆运动学模型
    # fk_model, fk_dir, fk_path = train_dofbot_model(data_path='dataset/60000/dofbot_fk_60000_norm.csv',
    #                                                model_type='mlp', mode='fk',
    #                                                fk_out_cols=['x', 'y', 'z'],
    #                                                epochs=2000, lr=1e-3, hidden_layers=[128, 128, 64])
    # ik_model, ik_dir, ik_path = train_dofbot_model(data_path='dataset/60000/dofbot_fk_60000_norm.csv',
    #                                                model_type='mlp', mode='ik',
    #                                                ik_in_cols=['x', 'y', 'z'],
    #                                                epochs=2000, lr=1e-3, hidden_layers=[128, 128, 64], fk_path=fk_path)
