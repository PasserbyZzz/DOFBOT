# model_validator.py
import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import torch
import pybullet as p
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from dofbot import DofbotEnv
from utils_learn.flexible_networks import FlexibleMLP
from utils_learn.utils_collect_visual import Normalizer


class ModelValidator:
    """
    一次性加载模型与统计量，之后可反复对任意关节角序列做验证。
    内部自动维护 PyBullet 连接，支持多次 validate 而不断开仿真。
    """

    def __init__(
        self,
        fk_model_path: str | Path, # 正解模型
        ik_model_path: str | Path, # 逆解模型（新增）
        stats_path: str | Path, # 归一化统计量
        input_keys_fk: List[str], 
        output_keys_fk: List[str],
        input_keys_ik: List[str],
        output_keys_ik: List[str],
        hidden_layers_fk: list[int]=[100, 30],
        hidden_layers_ik: list[int]=[100, 30],
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalizer = Normalizer(stats_path)
        self.input_keys_fk = input_keys_fk
        self.output_keys_fk = output_keys_fk
        self.input_keys_ik = input_keys_ik
        self.output_keys_ik = output_keys_ik

        # ---------- 加载 FK 模型 ----------
        self.fk_model = FlexibleMLP(
            input_dim=len(input_keys_fk),
            output_dim=len(output_keys_fk),
            hidden_layers=hidden_layers_fk,
            dropout=0.0,
            activation="ReLU",
            block_type="res",
            num_blocks=1,
        ).to(self.device)
        self.fk_model.load_state_dict(
            torch.load(fk_model_path, map_location=self.device)
        )
        self.fk_model.eval()

        # ---------- 加载 IK 模型（可选） ----------
        self.ik_model = None
        if ik_model_path is not None:
            self.ik_model = FlexibleMLP(
                input_dim=len(input_keys_ik),
                output_dim=len(output_keys_ik),
                hidden_layers=hidden_layers_ik,
                dropout=0.0,
                activation="ReLU",
                block_type="res",
                num_blocks=1,
            ).to(self.device)
            self.ik_model.load_state_dict(
                torch.load(ik_model_path, map_location=self.device)
            )
            self.ik_model.eval()

        # ---------- 初始化 PyBullet ----------
        if p.isConnected():
            p.disconnect()
        self.env = DofbotEnv()
        self.env.reset()

    # ---------------- internal helpers ----------------
    def _compose_pose_by_keys(self, pos: np.ndarray, orn_quat: np.ndarray, keys: list[str]) -> np.ndarray:
        """Given pos (3,) and quaternion (x,y,z,w), compose pose vector in the order of keys."""
        R = Rotation.from_quat(orn_quat).as_matrix()  # 3x3
        rpy = Rotation.from_quat(orn_quat).as_euler('xyz', degrees=False)  # roll, pitch, yaw
        vals: list[float] = []
        for k in keys:
            if k == 'x': vals.append(float(pos[0]))
            elif k == 'y': vals.append(float(pos[1]))
            elif k == 'z': vals.append(float(pos[2]))
            elif k == 'nx': vals.append(float(R[0, 0]))
            elif k == 'ny': vals.append(float(R[1, 0]))
            elif k == 'nz': vals.append(float(R[2, 0]))
            elif k == 'ox': vals.append(float(R[0, 1]))
            elif k == 'oy': vals.append(float(R[1, 1]))
            elif k == 'oz': vals.append(float(R[2, 1]))
            elif k == 'ax': vals.append(float(R[0, 2]))
            elif k == 'ay': vals.append(float(R[1, 2]))
            elif k == 'az': vals.append(float(R[2, 2]))
            elif k == 'roll': vals.append(float(rpy[0]))
            elif k == 'pitch': vals.append(float(rpy[1]))
            elif k == 'yaw': vals.append(float(rpy[2]))
            elif k == 'roll_sin': vals.append(float(np.sin(rpy[0])))
            elif k == 'roll_cos': vals.append(float(np.cos(rpy[0])))
            elif k == 'pitch_sin': vals.append(float(np.sin(rpy[1])))
            elif k == 'pitch_cos': vals.append(float(np.cos(rpy[1])))
            elif k == 'yaw_sin': vals.append(float(np.sin(rpy[2])))
            elif k == 'yaw_cos': vals.append(float(np.cos(rpy[2])))
            elif k in ('a','b','c','d'):
                # Quaternion assumed (x,y,z,w) from env
                qx, qy, qz, qw = orn_quat
                mapping = {'a': qx, 'b': qy, 'c': qz, 'd': qw}
                vals.append(float(mapping[k]))
            else:
                raise ValueError(f"Unsupported key in pose composition: {k}")
        return np.asarray(vals, dtype=np.float32)

    # ------------------------- FK 精度验证--------------------------
    def validate_fk(self, q_list: List[List[float]] | np.ndarray) -> Dict:
        """
        主接口：输入 (N,5) 关节角，返回误差统计与原始数据
        """
        q_arr = np.asarray(q_list, dtype=np.float32)
        if q_arr.ndim == 1 and q_arr.size == 5:
            q_arr = q_arr[None, :]
        assert q_arr.shape[1] == 5, "关节角必须是 5 DoF"

        N = q_arr.shape[0]

        # ---- 1. 网络预测 ----
        # 网络输入
        if len(self.input_keys_fk) == 10: # sin/cos
            q_sc = np.concatenate([np.sin(q_arr), np.cos(q_arr)], axis=1)
        elif len(self.input_keys_fk) == 5: # 直接角度
            # direct angles as inputs
            q_sc = q_arr
        # 网络输出
        with torch.no_grad():
            pred_norm = (
                self.fk_model(torch.tensor(q_sc, device=self.device))
                .cpu()
                .numpy()
            )  # (N, len(output_keys_fk))
        # 反归一化
        nn_list = [
            self.normalizer.denormalize_cols(pred_norm[i], self.output_keys_fk)
            for i in range(N)
        ]
        nn_pose = np.array(nn_list, dtype=np.float32)  # (N, len(output_keys_fk))

        # ---- 2. PyBullet 真值 ----
        pb_pose = np.zeros((N, len(self.output_keys_fk)), dtype=np.float32)
        for i, q in enumerate(q_arr):
            self.env.dofbot_forward_control(q, 0.0)  # 直接控制到位
            time.sleep(0.02)  # 稳定一帧
            pos, orn, _ = self.env.get_dofbot_pose()  # 获取末端位姿
            pb_pose[i] = self._compose_pose_by_keys(pos, orn, self.output_keys_fk)  # (N, len(output_keys_fk))

        # ---- 3. 误差 ----
        err = np.abs(nn_pose - pb_pose)  # (N, len(output_keys_fk))
        err_dict = {
            "pos_err_mm": err[:, :3] * 1000,  # 转 mm
            "mean_err_mm": np.mean(err[:, :3]) * 1000,
            "max_err_mm": np.max(err[:, :3]) * 1000,
        }

        return {
            "err_dict": err_dict,
            "q_arr": q_arr,
            "pb_pose": pb_pose,
            "nn_pose": nn_pose,
            "err": err,
        }

    # ------------------------- IK 精度验证--------------------------
    def validate_ik(
        self, target_pose: np.ndarray, angle_deg: bool = True
    ) -> Dict:
        """
        输入：target_pose (N, len(input_keys_ik))
              angle_deg 返回关节角是否转成 degree（默认 True）
        输出：Dict 含
              - q_pred     : 网络输出关节角 (N, 5)
              - pb_pos_mm  : PyBullet 实际末端位置 (N, 3) 单位 mm
              - err_pos_mm : 位置误差 (N, 3) 单位 mm
              - mean_err_mm: 平均位置误差
              - max_err_mm : 最大位置误差
        """
        tgt = np.asarray(target_pose, dtype=np.float32)
        if tgt.ndim == 1 and tgt.size == len(self.input_keys_ik):
            tgt = tgt[None, :]
        assert tgt.shape[1] == len(self.input_keys_ik), f"target_pose 必须是 (N,{len(self.input_keys_ik)})"

        N = tgt.shape[0]

        # ---- 1. 归一化输入 ----
        tgt_norm = np.array(
            [self.normalizer.normalize_cols(tgt[i], self.input_keys_ik) for i in range(N)]
        )  # (N, len(input_keys_ik))

        # ---- 2. 网络逆解 ----
        with torch.no_grad():
            q_pred = (
                self.ik_model(torch.tensor(tgt_norm, device=self.device))
                .cpu()
                .numpy()
            )  # (N, 5)  输出在 [0,π]
        s = q_pred[:, 0::2]  # (N, 5) sin
        c = q_pred[:, 1::2]  # (N, 5) cos
        q_pred_rad = np.arctan2(s, c)  # (N, 5) 范围 [-π,π]

        # ---- 3. PyBullet 真值 ----
        pb_pose = np.zeros((N, len(self.input_keys_ik)), dtype=np.float32)
        for i, q in enumerate(q_pred_rad):
            self.env.dofbot_forward_control(q, 0.0)
            time.sleep(0.02)
            pos, orn, _ = self.env.get_dofbot_pose()
            pb_pose[i] = self._compose_pose_by_keys(pos, orn, self.input_keys_ik)

        # ---- 4. 误差 ----
        err = np.abs(tgt - pb_pose)
        err_dict = {
            "pos_err_mm": err[:, :3] * 1000,  # 转 mm
            "mean_err_mm": np.mean(err[:, :3]) * 1000,
            "max_err_mm": np.max(err[:, :3]) * 1000,
        }

        return {
            "err_dict": err_dict,
            "tgt_arr": tgt,
            "pb_pose": pb_pose,
            "err": err,
        }


    # ------------------------------------------------------------------
    def plot(self, err_dict: Dict, save_path: str | Path = None):
        """
        可视化：箱线图 + 均值曲线
        """
        pos_mm = err_dict["pos_err_mm"]  # (N,3)

        fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=120)
        labels_pos = ["x", "y", "z"]
        labels_axis = [f"{tok}{i}" for tok in "nox" for i in "xyz"]

        ax.boxplot(pos_mm, labels=labels_pos)
        ax.set_title("Position error (mm)")
        ax.set_ylabel("mm")
        plt.tight_layout()

        if save_path:
            save_dir = Path(save_path).parent
            if save_dir and not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
        plt.show()

    # ------------------------------------------------------------------
    def close(self):
        if p.isConnected():
            p.disconnect()


# ----------------------------------------------------------------------
# 使用示例（可直接当脚本跑）
if __name__ == "__main__":
    validator = ModelValidator(
        model_path="../results/learn_model/mlp_fk_20251017_105721/model.pt",
        stats_path="../dataset/1200000/dofbot_fk_1200000_norm_stats.json",
        output_keys=['q1_sin','q1_cos','q2_sin','q2_cos','q3_sin','q3_cos','q4_sin','q4_cos','q5_sin','q5_cos'],
        hidden_layers=[100, 30]
    )

    # 生成 100 组随机关节角做验证
    rand_q = np.random.uniform(
        low=[0] * 5, high=[np.pi] * 5, size=(100, 5)
    )
    res = validator.validate(rand_q)
    print("平均位置误差: %.2f mm" % res["err_dict"]["mean_pos_mm"])
    print("最大位置误差: %.2f mm" % res["err_dict"]["max_pos_mm"])

    validator.plot(res["err_dict"], save_path="error_analysis.png")
    validator.close()


    # ik_res = validator.validate_ik(tgt_pose, angle_deg=True)
    # print("IK 平均位置误差: %.2f mm" % ik_res["mean_err_mm"])