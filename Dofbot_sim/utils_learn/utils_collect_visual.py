import json, csv, os, time, numpy as np, pybullet as p
from datetime import datetime
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from dofbot import DofbotEnv, any_self_collision
from scipy.spatial.transform import Rotation
import torch
import shutil

# ---------------- 默认量纲边界 ----------------
KEYS = ['q1', 'q2', 'q3', 'q4', 'q5',
        'x', 'y', 'z', 'a', 'b', 'c', 'd',
        'roll', 'pitch', 'yaw',
        'nx', 'ny', 'nz',
        'ox', 'oy', 'oz',
        'ax', 'ay', 'az']

KEYS_NORM = (
    ['q1_sin','q1_cos','q2_sin','q2_cos','q3_sin','q3_cos','q4_sin','q4_cos','q5_sin','q5_cos'] +
    ['x','y','z','a','b','c','d']+
    ['roll_sin','roll_cos','pitch_sin','pitch_cos','yaw_sin','yaw_cos'] +
    ['nx', 'ny', 'nz', 'ox', 'oy', 'oz', 'ax', 'ay', 'az']
)

# sin/cos 列范围固定 [-1,1]；位置/四元数用原 MIN/MAX
MIN_NORM = np.array(
    [-1,-1]*5 +                            # 10 个角度
    [-1,-1,-1] +                           # xyz
    [-1,-1,-1,-1] +                        # quat
    [-1,-1]*3 +                             # 6 个角度
    [-1] * 9
)
MAX_NORM = np.array(
    [1,1]*5 + [1,1,1] + [1,1,1,1] + [1,1]*3 + [1] * 9
)

MIN_VALS = np.array([-np.pi, 0, 0, 0, 0,
                     -0.2444, -0.3170, -0.1273, -1.0, -1.0, -1.0, -1.0,
                     -np.pi, -np.pi/2, -np.pi,
                     -1.0, -1.0, -1.0,   -1.0, -1.0, -1.0,   -1.0, -1.0,-1.0])
MAX_VALS = np.array([np.pi, np.pi, np.pi, np.pi, np.pi,
                     0.3909, 0.3171, 0.4255, 1.0, 1.0, 1.0, 1.0,
                     np.pi, np.pi/2, np.pi,
                     1.0, 1.0, 1.0,   1.0, 1.0, 1.0,   1.0, 1.0, 1.0])

LOWER_POSE = np.array(MIN_VALS[5:15])
UPPER_POSE = np.array(MAX_VALS[5:15])

# ---------- 可调参数 ----------
CHUNK_SIZE   = 1000          # 每多少条刷一次盘
WORK_DIR     = Path('dataset') # 顶层目录

# ---------- 数据集采集仿真函数 ----------
def worker(rank: int, samples_per_worker: int, flush_every: int, run_tag: str):
    """
    每个 worker 独享一个分片 csv，边采边写。
    返回 (rank, 实际写入条数, 分片文件绝对路径)
    """
    chunk_file = WORK_DIR / run_tag / f'chunk_{rank:03d}.csv'
    chunk_file.parent.mkdir(parents=True, exist_ok=True)

    # 如果分片已存在，直接统计条数后返回（续采）
    if chunk_file.exists():
        with open(chunk_file, 'r', newline='') as f:
            exist_rows = sum(1 for _ in f) - 1  # 去掉表头
        print(f'[Worker {rank}] 发现已有分片，跳过采集，已存在 {exist_rows} 条')
        return rank, exist_rows, str(chunk_file.resolve())

    # 否则重新采集
    gui_options = f"--window_port={6660 + rank} --width=640 --height=480"
    conn = p.connect(p.DIRECT, options=gui_options)
    env = DofbotEnv(physicsClientId=conn)
    env.reset()

    ll = [-np.pi, 0, 0, 0, -np.pi]
    ul = [np.pi, np.pi, np.pi, np.pi, np.pi]

    buf, written = [], 0
    max_attempts = 100
    with open(chunk_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(KEYS)          # 表头
        while written < samples_per_worker:
            joint_angles = [np.random.uniform(lo, hi) for lo, hi in zip(ll, ul)]
            joint_angles_now, _ = env.get_dofbot_jointPoses()
            pos_last, _, euler_last = env.get_dofbot_pose()
            err_joint = np.linalg.norm(np.array(joint_angles) - np.array(joint_angles_now))
            err_pos   = 10.0
            attempts  = 0
            while (err_joint > 0.001 or err_pos > 0.001) and attempts < max_attempts:
                env.dofbot_forward_control(joint_angles, 0.0)
                pos_now, orn_now, euler_now = env.get_dofbot_pose()

                # 把 pos + orn + euler 拼成 9 维向量
                pose_vec = np.concatenate([pos_now, orn_now, euler_now])

                # # 只要有一个维度越界或者|q|≠1就触发
                # if ((pose_vec < LOWER_POSE).any() or (pose_vec > UPPER_POSE).any() or abs(np.linalg.norm(orn_now) - 1.0) > 1e-3):
                #     attempts = max_attempts
                #     # print(f"[Worker {rank}][Local {i + 1}] unreachable. Retrying...")
                #     break
                #
                # if any_self_collision(env._dofbot.dofbotUid, safety_margin=0.001):
                #     attempts = max_attempts
                #     # print(f"[Worker {rank}][Local {i + 1}] Collision. Retrying...")
                #     break

                joint_angles_now, _ = env.get_dofbot_jointPoses()
                err_joint = np.linalg.norm(np.array(joint_angles) - np.array(joint_angles_now))
                err_pos   = np.linalg.norm(np.array(pos_now) - np.array(pos_last))
                pos_last  = pos_now
                attempts += 1
            if attempts >= max_attempts:
                # print(f"[Worker {rank}][Local {i + 1}] Collision or unreachable. Retrying...")
                env.reset()
                continue   # 碰撞或不可达，重采
            pos_real, orn_real, euler_real = env.get_dofbot_pose()
            R_mat = Rotation.from_quat(orn_real).as_matrix()
            nx, ny, nz = R_mat[:, 0]
            ox, oy, oz = R_mat[:, 1]
            ax, ay, az = R_mat[:, 2]
            row = list(joint_angles) + list(pos_real) + list(orn_real) + list(euler_real)+[nx, ny, nz, ox, oy, oz, ax, ay, az]
            buf.append(row)
            written += 1
            if len(buf) >= flush_every:
                writer.writerows(buf)
                f.flush()
                buf.clear()
                print(f'[Worker {rank}] 已写入 {written + flush_every}/{samples_per_worker}')
            if written % 1000 == 0:
                print(f"✅ [Worker {rank}] Collected {written}/{samples_per_worker}")
        # 尾部不足一 chunk
        if buf:
            writer.writerows(buf)
            f.flush()
    # 退出时记得断开
    p.disconnect(conn)
    return rank, written, str(chunk_file.resolve())


def angle_encode(theta):
    return [np.sin(theta), np.cos(theta)]

# ---------- 采集Dofbot正运动学数据集 ----------
def collect_dofbot_dataset(num_envs: int = 2,
                           num_samples: int = 4000,
                           show_gui: bool = False,
                           sleep: float = 0.01,
                           flush_every: int = CHUNK_SIZE):
    """
    采集 Dofbot 正运动学数据集
    返回 (N,15) 原始量纲 ndarray，并自动落盘
    流式采集 120 万条，内存占用 < num_envs×flush_every×单行字节数
    """
    mp.set_start_method('spawn', force=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_per_worker = (num_samples + num_envs - 1) // num_envs

    # 1. 启动所有 worker（同步返回分片信息）
    with mp.Pool(num_envs) as pool:
        results = [
            pool.apply_async(worker, (r, samples_per_worker, flush_every, run_tag))
            for r in range(num_envs)
        ]
        chunk_info = [r.get() for r in results]  # [(rank, cnt, path), ...]

    # 2. 主进程：归并分片、计算真实 min/max、写最终三件套
    print('\n[Main] 所有分片采集完成，开始合并与归一化...')
    raw_csv = WORK_DIR / run_tag / f'dofbot_fk_{num_samples}_raw.csv'
    norm_csv = WORK_DIR / run_tag / f'dofbot_fk_{num_samples}_norm.csv'
    stats_json = WORK_DIR / run_tag / f'dofbot_fk_{num_samples}_norm_stats.json'

    # 2.1 先扫描一遍拿到全局 min/max（流式，不加载全量）
    mins = +np.inf * np.ones(24)
    maxs = -np.inf * np.ones(24)
    total_rows = 0
    for _, cnt, path in chunk_info:
        total_rows += cnt
        with open(path, 'r', newline='') as f:
            rdr = csv.reader(f)
            next(rdr)  # 跳过表头
            for row in rdr:
                vals = np.array(row, dtype=np.float64)
                mins = np.minimum(mins, vals)
                maxs = np.maximum(maxs, vals)

    # 2.2 写合并后的 raw 文件（流式复制，不额外占内存）
    with open(raw_csv, 'w', newline='') as dst:
        writer = csv.writer(dst)
        writer.writerow(KEYS)
        for _, _, path in chunk_info:
            with open(path, 'r', newline='') as src:
                next(src)  # 丢表头
                shutil.copyfileobj(src, dst)  # 按块复制，内存友好

    # 2.3 写 stats
    stats = {k: {'min': float(mins[i]), 'max': float(maxs[i])}
             for i, k in enumerate(KEYS)}
    with open(stats_json, 'w') as f:
        json.dump(stats, f, indent=4)

    # 2.4 第二遍流式生成 norm 文件（24 维）
    with open(norm_csv, 'w', newline='') as dst:
        writer = csv.writer(dst)
        writer.writerow(KEYS_NORM)
        for _, _, path in chunk_info:
            with open(path, 'r', newline='') as src:
                rdr = csv.reader(src)
                next(rdr)
                for row in rdr:
                    vec = np.array(row, dtype=np.float32)
                    # 归一化逻辑同原代码
                    q_raw = vec[:5]
                    xyz = vec[5:8]
                    quat = vec[8:12]
                    euler = vec[12:15]
                    dir_vec = vec[15:24]
                    sc_joint = np.concatenate([np.sin(q_raw), np.cos(q_raw)])
                    xyz_n = 2 * (xyz - MIN_VALS[5:8]) / (MAX_VALS[5:8] - MIN_VALS[5:8]) - 1
                    sc_euler = np.concatenate([np.sin(euler), np.cos(euler)])
                    dir_n = (dir_vec - MIN_VALS[15:24]) / (MAX_VALS[15:24] - MIN_VALS[15:24]) - 1
                    norm_row = np.hstack([sc_joint, xyz_n, quat, sc_euler, dir_n])
                    writer.writerow(norm_row)

    # 2.5 可选：删除分片文件以节省磁盘
    for _, _, path in chunk_info:
        os.remove(path)

    print(f'\n✅ 流式采集完成，最终 {total_rows} 条 → {raw_csv.parent}')
    return raw_csv, norm_csv, stats_json

    # # 分布诊断
    # print('\n📊 Distribution (same as original):')
    # for i, k in enumerate(KEYS):
    #     col = dataset_np[:, i]
    #     mean, std, cover = float(col.mean()), float(col.std()), \
    #         (col.max() - col.min()) / (MAX_VALS[i] - MIN_VALS[i] + 1e-8) * 100
    #     print(f' - {k}: mean={mean:.4f} std={std:.4f}  coverage={cover:.2f}%')
    #
    # print('\n✅ Parallel FK dataset collection done.')
    # return dataset_np

    # mp.set_start_method('spawn', force=True)
    # samples_per_worker = (num_samples + num_envs - 1) // num_envs
    # queue = mp.SimpleQueue()
    # processes = [mp.Process(target=worker, args=(r, queue, samples_per_worker, show_gui, sleep))
    #              for r in range(num_envs)]
    # for p in processes:
    #     p.start()
    # # 收集
    # dataset_chunks = []
    # for _ in range(num_envs):
    #     rank, chunk = queue.get()
    #     print(f'[Main] received {chunk.shape[0]} samples from Worker {rank}')
    #     dataset_chunks.append(chunk)
    # for p in processes:
    #     p.join()
    # dataset_np = np.concatenate(dataset_chunks, axis=0)[:num_samples]  # 精确截断
    #
    # ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # save_dir = os.path.join('dataset', ts)  # 例如 dataset/20250919_095516
    # os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    #
    # prefix = os.path.join(save_dir, f'dofbot_fk_{num_samples}')
    #
    # raw_csv   = prefix + '_raw.csv'
    # norm_csv  = prefix + '_norm.csv'
    # stats_json= prefix + '_norm_stats.json'
    #
    # # raw
    # with open(raw_csv, 'w', newline='') as f:
    #     csv.writer(f).writerow(KEYS)
    #     csv.writer(f).writerows(dataset_np.tolist())
    # print('✅ Raw saved →', raw_csv)
    #
    # # ---------- 重新计算真实 min/max ----------
    # real_min = dataset_np.min(axis=0)
    # real_max = dataset_np.max(axis=0)
    #
    # # 更新 MIN_VALS 和 MAX_VALS（用于后续归一化）
    # MIN_VALS[:] = real_min
    # MAX_VALS[:] = real_max
    #
    # # norm
    # # ---------- 1. 先拆列 ----------
    # q_raw = dataset_np[:, :5]  # 5 joint
    # xyz = dataset_np[:, 5:8]  # 3 pos
    # quat = dataset_np[:, 8:12]  # 4 quat
    # euler = dataset_np[:, 12:15]  # 3 euler
    # # ---------- 2. 所有角度统一 sin/cos ----------
    # sin_cos_joint = np.concatenate([np.sin(q_raw), np.cos(q_raw)], axis=1)
    # sin_cos_euler = np.concatenate([np.sin(euler), np.cos(euler)], axis=1)
    # # 位置 & 四元数 → MinMax [-1,1]（四元数已天然在内，可不再缩放）
    # xyz_norm = 2 * (xyz - MIN_VALS[5:8]) / (MAX_VALS[5:8] - MIN_VALS[5:8]) - 1
    # quat_norm = quat  # 已在 [-1,1]
    #
    # # ---------- 3. 拼最终归一化矩阵 ----------
    # norm = np.hstack([sin_cos_joint,  # 10 维  (q1~q5)
    #                   xyz_norm,  # 3 维
    #                   quat_norm,  # 4 维
    #                   sin_cos_euler])  # 6 维  (roll,pitch,yaw)
    #
    # # norm = 2 * (dataset_np - MIN_VALS) / (MAX_VALS - MIN_VALS) - 1
    # with open(norm_csv, 'w', newline='') as f:
    #     csv.writer(f).writerow(KEYS_NORM)
    #     csv.writer(f).writerows(norm.tolist())
    # print('✅ Norm saved →', norm_csv)
    #
    # # stats
    # stats = {k: {'min': float(MIN_VALS[i]), 'max': float(MAX_VALS[i])}
    #          for i, k in enumerate(KEYS)}
    # with open(stats_json, 'w') as f:
    #     json.dump(stats, f, indent=4)
    # print('✅ Stats saved →', stats_json)
    #
    # # 分布诊断
    # print('\n📊 Distribution (same as original):')
    # for i, k in enumerate(KEYS):
    #     col = dataset_np[:, i]
    #     mean, std, cover = float(col.mean()), float(col.std()), \
    #                        (col.max()-col.min())/(MAX_VALS[i]-MIN_VALS[i]+1e-8)*100
    #     print(f' - {k}: mean={mean:.4f} std={std:.4f}  coverage={cover:.2f}%')
    #
    # print('\n✅ Parallel FK dataset collection done.')
    # return dataset_np

# ---------- 可视化Dofbot数据集的工作空间 ----------
def visualize_workspace(raw_csv: str,
                        save_dir: str = "results/workspace_figs",
                        show: bool = True,
                        save: bool = True,
                        point_size: float = 1.0,
                        alpha: float = 0.3):
    """
    可视化 Dofbot 工作空间并保存图片

    参数
    ----
    raw_csv : str
        采集生成的 *_raw.csv 路径
    save_dir : str
        图片保存目录，不存在会自动创建
    show : bool
        是否弹出窗口
    save : bool
        是否保存 png
    point_size / alpha : float
        散点大小与透明度
    """
    # ---------- 数据入口 ----------
    # ---- 读取 csv ----
    data = np.loadtxt(raw_csv, delimiter=',', skiprows=1)
    with open(raw_csv, 'r', newline='') as f:
        header = f.readline().strip().split(',')
    print(f"\n📊 文件：{raw_csv}")
    for col_idx, name in enumerate(header):
        print(f"  {name:>8s}:  min={data[:, col_idx].min():+.6f}  "
              f"max={data[:, col_idx].max():+.6f}")

    xyz = data[:, 5:8]  # x y z 列

    # 3. 绘图 ----------------------------------------------------------
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')
    # 按 Z 高度着色
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                    s=point_size, c=xyz[:, 2], cmap='viridis', alpha=alpha)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Dofbot Reachable Workspace')
    ax.set_box_aspect([1, 1, 1])

    # 颜色条（显示 Z 值）
    cb = fig.colorbar(sc, ax=ax, pad=0.1)
    cb.set_label('Z (m)')

    # 4. 保存 ----------------------------------------------------------
    if save:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = Path(raw_csv).stem.replace('_norm', '').replace('_raw', '')
        png_path = os.path.join(save_dir, f"{name}_workspace_{ts}.png")
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"✅ 图片已保存 → {png_path}")

    # 5. 显示 ----------------------------------------------------------
    # 5. 显示 + q 退出 --------------------------------------------------
    if show:
        plt.show(block=False)  # 非阻塞，才能接事件

        def _quit(event):
            if event.key.lower() == 'q':
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', _quit)

        # 50 ms 轮询，直到窗口被关闭
        while plt.fignum_exists(fig.number):
            plt.pause(0.05)
    else:
        plt.close(fig)

    return fig, ax


# normalizer.py


class Normalizer:
    """
    与 train_dofbot_model 完全对齐的归一化工具。
    - 优先使用数据集中生成的统计量（stats.json）
    - 对于 sin/cos 特征（如 roll_sin、yaw_cos、q1_sin、q5_cos 等），若 stats.json 未提供，默认使用区间 [-1, 1]
    只处理 COLS 中出现的字段，其余忽略。
    """
    COLS = [
        # 关节原始角
        'q1', 'q2', 'q3', 'q4', 'q5',
        # 末端位姿常见表示
        'x', 'y', 'z', 'a', 'b', 'c', 'd',
        'roll', 'pitch', 'yaw',
        'nx', 'ny', 'nz', 'ox', 'oy', 'oz', 'ax', 'ay', 'az',
        # 欧拉角的正余弦展开（可选）
        'roll_sin', 'roll_cos', 'pitch_sin', 'pitch_cos', 'yaw_sin', 'yaw_cos',
        # 关节角的正余弦展开（可选，常用于输入/输出）
        'q1_sin', 'q1_cos', 'q2_sin', 'q2_cos', 'q3_sin', 'q3_cos', 'q4_sin', 'q4_cos', 'q5_sin', 'q5_cos',
    ]

    def __init__(self, stats_path: str):
        """
        stats_path: 由数据集预处理阶段生成的 *.json
        """
        with open(stats_path, 'r', encoding='utf-8') as f:
            full_stats = json.load(f)

        # 构造 keys/mins/maxs：
        # - 若 stats.json 提供则使用
        # - 若是 sin/cos 相关键但 stats 缺失，则默认 [-1, 1]
        keys: list[str] = []
        mins: list[float] = []
        maxs: list[float] = []

        def is_sin_cos_key(k: str) -> bool:
            if k.endswith('_sin') or k.endswith('_cos'):
                return True
            return False

        for k in self.COLS:
            if k in full_stats:
                keys.append(k)
                mins.append(float(full_stats[k]['min']))
                maxs.append(float(full_stats[k]['max']))
            elif is_sin_cos_key(k):
                # 对缺失的 sin/cos 特征，设置默认范围 [-1, 1]
                keys.append(k)
                mins.append(-1.0)
                maxs.append(1.0)
            else:
                # 其他键缺失则忽略（不参与归一化/反归一化）
                continue

        self.keys = keys                                # 固定顺序
        self.mins = np.array(mins, dtype=np.float32)
        self.maxs = np.array(maxs, dtype=np.float32)
        self.ranges = self.maxs - self.mins
        self.ranges[self.ranges == 0] = 1.0             # 避免除 0

    # ---------- NumPy 版本 ----------
    def normalize_cols(self, data: np.ndarray, cols) -> np.ndarray:
        """
        data: (N, len(cols)) 的原始值
        cols: 与 data 列名顺序一致的 list
        return: 归一化后的 (N, len(cols)) 数组
        """
        idx = [self.keys.index(c) for c in cols]
        mins = self.mins[idx]
        ranges = self.ranges[idx]
        return 2.0 * (data - mins) / ranges - 1

    def denormalize_cols(self, data: np.ndarray, cols) -> np.ndarray:
        """
        data: (N, len(cols)) 的归一化值
        cols: 与 data 列名顺序一致的 list
        return: 反归一化后的 (N, len(cols)) 数组
        """
        idx = [self.keys.index(c) for c in cols]
        mins = self.mins[idx]
        ranges = self.ranges[idx]
        return data * ranges + (mins + ranges / 2.0)

    # ---------- PyTorch 版本 ----------
    def normalize_cols_tensor(self, data: torch.Tensor, cols) -> torch.Tensor:
        idx = [self.keys.index(c) for c in cols]
        mins = torch.as_tensor(self.mins[idx], device=data.device, dtype=data.dtype)
        ranges = torch.as_tensor(self.ranges[idx], device=data.device, dtype=data.dtype)
        return 2.0 * (data - mins) / ranges - 1

    def denormalize_cols_tensor(self, data: torch.Tensor, cols) -> torch.Tensor:
        idx = [self.keys.index(c) for c in cols]
        mins = torch.as_tensor(self.mins[idx], device=data.device, dtype=data.dtype)
        ranges = torch.as_tensor(self.ranges[idx], device=data.device, dtype=data.dtype)
        return data * ranges + (mins + ranges / 2.0)


if __name__ == '__main__':
    collect_data = collect_dofbot_dataset(num_envs=6, num_samples=600, show_gui=True)
    print('采集完成，形状：', collect_data.shape)
    visualize_workspace(data=collect_data)
