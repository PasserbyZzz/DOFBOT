# ==========================================
#  Dofbot 正逆运动学模型学习作业任务
# ==========================================

# 模块化工具导入
from utils_learn.utils_model_train import train_dofbot_model
from utils_learn.utils_collect_visual import collect_dofbot_dataset, visualize_workspace
from utils_learn.utils_model_test import ModelValidator
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

pi = 3.1415926          # 自己指定 π，方便后续打印保留 7 位小数

if __name__ == "__main__":
    # ==============================================
    # 仿真任务4、  数据采集与工作空间可视化
    # ==============================================
    # 可参考demo
    # 10 个并行环境 × 50 k 条样本 ≈ 15 min（RTX-3060）
    # 若显存 < 8 G，建议 num_envs ≤ 6
    raw_csv, norm_csv, stats_json = collect_dofbot_dataset(num_envs=2, num_samples=12000, show_gui=False)
    # 可视化仅用于快速验证可达空间是否异常（空洞、断层）
    visualize_workspace(raw_csv = raw_csv)

    # ==============================================
    # 仿真任务5、  正/逆运动学模型训练
    # ==============================================
    # 可选择提供的现有数据集60000(组) or 自己采集的数据集
    # 5.1 正运动学（FK）
    #     输入：6 关节角 → 输出：6 Dof 位姿（x,y,z,roll,pitch,yaw）
    #     隐藏层 [128,128,64] 经网格搜索，在 60 k 数据上验证误差
    # 可参考demo
    # data_path = norm_csv
    data_path = "dataset/120000/dofbot_fk_120000_norm.csv"
    stats_path = "dataset/120000/dofbot_fk_120000_norm_stats.json"
    fk_hidden_layers = [256,256,128]  # [100, 30]
    fk_lr = 1e-3  # 0.1
    fk_epochs = 500
    fk_model, fk_dir, fk_path = train_dofbot_model(
        data_path=data_path,
        model_type='mlp', mode='fk',
        in_cols=['q1_sin', 'q1_cos', 'q2_sin', 'q2_cos', 'q3_sin', 'q3_cos', 'q4_sin', 'q4_cos', 'q5_sin', 'q5_cos'],
        # out_cols=['x', 'y', 'z', 'nx', 'ny', 'nz', 'ox', 'oy', 'oz', 'ax', 'ay', 'az'],
        # out_cols=['x', 'y', 'z', 'roll_sin', 'roll_cos', 'pitch_sin', 'pitch_cos', 'yaw_sin', 'yaw_cos'],
        out_cols=['x', 'y', 'z', 'a', 'b', 'c', 'd'],
        epochs=fk_epochs, lr=fk_lr, min_lr=1e-5, hidden_layers=fk_hidden_layers
    )
    
    # 5.2 逆运动学（IK）
    # 可参考demo, 依赖 FK 用于监督训练
    ik_hidden_layers = [256,256,128]  # [100, 30]
    ik_lr = 1e-3  # 0.1
    ik_epochs = 500
    ik_model, ik_dir, ik_path = train_dofbot_model(
        data_path=data_path,
        model_type='mlp', mode='ik',
        # in_cols=['x', 'y', 'z', 'nx', 'ny', 'nz', 'ox', 'oy', 'oz', 'ax', 'ay', 'az'],
        # in_cols=['x', 'y', 'z', 'roll_sin', 'roll_cos', 'pitch_sin', 'pitch_cos', 'yaw_sin', 'yaw_cos'],
        in_cols=['x', 'y', 'z', 'a', 'b', 'c', 'd'],
        out_cols = ['q1_sin', 'q1_cos', 'q2_sin', 'q2_cos', 'q3_sin', 'q3_cos', 'q4_sin', 'q4_cos', 'q5_sin', 'q5_cos'],
        epochs=ik_epochs, lr=ik_lr, min_lr=1e-5, hidden_layers=ik_hidden_layers,
        fk_path=fk_path, fk_hidden_layers=fk_hidden_layers
    )
    
    # ==============================================
    # 仿真任务6、  验证训练得到的正逆运动学模型预测结果，分析误差原因
    # ==============================================
    # 可参考demo
    validator = ModelValidator(
        fk_model_path=fk_path,
        ik_model_path=ik_path,
        stats_path=stats_path,
        input_keys_fk=['q1_sin','q1_cos','q2_sin','q2_cos','q3_sin','q3_cos','q4_sin','q4_cos','q5_sin','q5_cos'],
        # output_keys_fk=['x', 'y', 'z', 'nx', 'ny', 'nz', 'ox', 'oy', 'oz', 'ax', 'ay', 'az'],
        # output_keys_fk=['x', 'y', 'z', 'roll_sin', 'roll_cos', 'pitch_sin', 'pitch_cos', 'yaw_sin', 'yaw_cos'],
        output_keys_fk=['x', 'y', 'z', 'a', 'b', 'c', 'd'],
        # input_keys_ik=['x', 'y', 'z', 'nx', 'ny', 'nz', 'ox', 'oy', 'oz', 'ax', 'ay', 'az'],
        # input_keys_ik=['x', 'y', 'z', 'roll_sin', 'roll_cos', 'pitch_sin', 'pitch_cos', 'yaw_sin', 'yaw_cos'],
        input_keys_ik=['x', 'y', 'z', 'a', 'b', 'c', 'd'],
        output_keys_ik=['q1_sin','q1_cos','q2_sin','q2_cos','q3_sin','q3_cos','q4_sin','q4_cos','q5_sin','q5_cos'],
        hidden_layers_fk=fk_hidden_layers,
        hidden_layers_ik=ik_hidden_layers,
    )
    
    # 6.1 生成 100 组随机关节角做正运动学模型验证
    rand_q = np.random.uniform(
        low=[0] * 5, high=[np.pi] * 5, size=(100, 5)
    )
    fk_res = validator.validate_fk(rand_q)
    print("fk 平均位置误差: %.2f mm" % fk_res["err_dict"]["mean_err_mm"])
    print("fk 最大位置误差: %.2f mm" % fk_res["err_dict"]["max_err_mm"])
    validator.plot(fk_res["err_dict"], save_path="results/model_results_xyzabcd/error_analysis_fk.png")

    # 6.2 生成 100 组随机末端位姿做逆运动学模型验证（12 维：xyz + I9）
    tgt_pose = np.random.uniform([-0.2, -0.3, 0.1], [0.3, 0.3, 0.4], (100, 3))
    I9 = np.tile(np.eye(3).ravel(), (100, 1))
    tgt_pose = np.hstack([tgt_pose, I9])

    # 6.2 生成 100 组随机末端位姿做逆运动学模型验证（9 维：xyz + rpy 的 sin/cos）
    pos = np.random.uniform(low=[-0.2, -0.3, 0.1], high=[0.3, 0.3, 0.4], size=(100, 3))
    # 欧拉角采样范围：roll,yaw∈[-pi,pi]，pitch∈[-pi/2,pi/2]
    rpy = np.random.uniform(low=[-np.pi, -np.pi/2, -np.pi], high=[np.pi, np.pi/2, np.pi], size=(100, 3))
    roll, pitch, yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    tgt_pose = np.column_stack([
        pos,
        np.sin(roll), np.cos(roll),
        np.sin(pitch), np.cos(pitch),
        np.sin(yaw), np.cos(yaw),
    ])

    # 6.2 生成 100 组随机末端位姿做逆运动学模型验证（7 维：xyz + abcd）
    pos = np.random.uniform(low=[-0.2, -0.3, 0.1], high=[0.3, 0.3, 0.4], size=(100, 3))
    rpy = np.random.uniform(low=[-np.pi, -np.pi/2, -np.pi], high=[np.pi, np.pi/2, np.pi], size=(100, 3))
    quat = R.from_euler('xyz', rpy, degrees=False).as_quat() # 形状 (100,4)，顺序 (x,y,z,w)
    tgt_pose = np.hstack([pos, quat]) # 形状 (100,7)
    
    ik_res = validator.validate_ik(tgt_pose)
    print("iK 平均位置误差: %.2f mm" % ik_res["err_dict"]["mean_err_mm"])
    print("ik 最大位置误差: %.2f mm" % ik_res["err_dict"]["max_err_mm"])
    validator.plot(ik_res["err_dict"], save_path="results/model_results_xyzabcd/error_analysis_ik.png")
    
    validator.close()