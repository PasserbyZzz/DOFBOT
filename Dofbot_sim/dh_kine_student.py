# ==========================================
# Dofbot 基于改进DH参数法的正/逆运动学建模作业任务
# ==========================================

# --------------------- 导入常用库 ---------------------
import roboticstoolbox as rtb   # 机器人专用工具箱
import numpy as np              # 矩阵运算
import matplotlib.pyplot as plt # 可视化
import os                       # 路径与保存

# --------------------- 常量定义 ---------------------
pi = 3.1415926          # 自己指定 π，方便后续打印保留 7 位小数
# 连杆长度（单位：m，与实物一致）
l1 = 0.1045             # 连杆1长度（基座→关节2）
l2 = 0.08285            # 连杆2长度（关节2→关节3）
l3 = 0.08285            # 连杆3长度（关节3→关节4）
l4 = 0.12842            # 连杆4长度（关节4→末端）

# ==============================================
# 用改进 DH 法建立机器人模型Demo
# ==============================================
# RevoluteMDH(a, alpha, d, offset)
# 默认 theta 为关节变量，因此只写常数项即可
DH_demo = rtb.DHRobot(
    [
        # rtb.RevoluteMDH(d=l1), # 关节1：绕 z 旋转，d 向上偏移 l1
        # rtb.RevoluteMDH(alpha=-pi/2, offset=-pi/3), # 关节2：x 向下扭转 90°，初始偏置 -90°
        # rtb.RevoluteMDH(a=l2, offset = pi / 6), # 关节3：平移 l2
        # rtb.RevoluteMDH(a=l3, offset=pi * 2 / 3), # 关节4：平移 l3，初始偏置 +90°
        # rtb.RevoluteMDH(alpha=pi/2, d=l4) # 关节5：x 向上扭转 90°，末端延伸 l4
        
        rtb.RevoluteMDH(d=l1), # 关节1：绕 z 旋转，d 向上偏移 l1
        rtb.RevoluteMDH(a=l2), # 关节2：绕 z 旋转，平移 l2
        rtb.PrismaticMDH(a=l3, alpha=pi), # 关节3：沿 x 轴反向平移 l3
        rtb.RevoluteMDH(d=l4) # 关节4：绕 z 旋转，末端延伸 l4
    ],
    name="DH_demo" # 给机器人起个名字，打印时更直观
)

# 打印标准 DH 参数表（alpha、a、d、theta、offset）
# print("========== DH_demo机器人 DH 参数 ==========")
# print(DH_demo)

# --------------------- 零位验证 ---------------------
# fkine_input0 = [0, 0, 0, 0] # 全部关节置 0
# fkine_result0 = DH_demo.fkine(fkine_input0)
# print("\n零位正解齐次变换矩阵:")
# print(fkine_result0)
# DH_demo.plot(q=fkine_input0, block=True) # 3D 可视化（阻塞模式）





# ==============================================
# 仿真任务0、 用改进 DH 法建立Dofbot机器人模型
# ==============================================
# RevoluteMDH(a, alpha, d, offset)
# 默认 theta 为关节变量，因此只写常数项即可
dofbot = rtb.DHRobot(
    [
        rtb.RevoluteMDH(a=0, alpha=0, d=l1, offset=0),
        rtb.RevoluteMDH(a=0, alpha=-pi/2, d=0, offset=-pi/2),
        rtb.RevoluteMDH(a=l2, alpha=0, d=0, offset=0),
        rtb.RevoluteMDH(a=l3, alpha=0, d=0, offset=pi/2),
        rtb.RevoluteMDH(a=0, alpha=pi/2, d=l4, offset=0)
    ],
    name="Dofbot"
)

# 打印标准 DH 参数表（alpha、a、d、theta、offset）
print("========== Dofbot机器人 DH 参数 ==========")
print(dofbot)

# --------------------- 4. Part0 零位验证 ---------------------
fkine_input0 = [0, 0, 0, 0, 0] # 全部关节置 0
fkine_result0 = dofbot.fkine(fkine_input0)
print("\n零位正解齐次变换矩阵:")
print(fkine_result0)
dofbot.plot(q=fkine_input0, block=False) # 3D 可视化（非阻塞模式）
# plt.savefig("figures/task_0", bbox_inches='tight', dpi=150)

# ==============================================
# 仿真任务1、 正运动学 —— 给出DH模型在以下 4 组关节角下的正运动学解
# ==============================================
# poses = [
#     [0., pi/3, pi/4, pi/5, 0.],            # demo
#     [pi/2, pi/5, pi/5, pi/5, pi],          # 1
#     [pi/3, pi/4, -pi/3, -pi/4, pi/2],      # 2
#     [-pi/2, pi/3, -2*pi/3, pi/3, pi/3]     # 3
# ]

# -------- 1.1 demo  pose ----------
q_demo = [0., pi/3, pi/4, pi/5, 0.]
T_demo = dofbot.fkine(q_demo)
print("\n========== Part1-0 (demo) 正解 ==========")
print(T_demo)
dofbot.plot(q=q_demo, block=False)
# plt.savefig("figures/task_1_1", bbox_inches='tight', dpi=150)

# -------- 1.2 pose 1 ----------
q_demo = [pi/2, pi/5, pi/5, pi/5, pi]
T_demo = dofbot.fkine(q_demo)
print("\n========== Part1-1 (pose 1) 正解 ==========")
print(T_demo)
dofbot.plot(q=q_demo, block=False)
# plt.savefig("figures/task_1_2", bbox_inches='tight', dpi=150)

# -------- 1.3 pose 2 ----------
q_demo = [pi/3, pi/4, -pi/3, -pi/4, pi/2]
T_demo = dofbot.fkine(q_demo)
print("\n========== Part1-2 (pose 2) 正解 ==========")
print(T_demo)
dofbot.plot(q=q_demo, block=False)
# plt.savefig("figures/task_1_3", bbox_inches='tight', dpi=150)

# -------- 1.4 pose 3 ----------
q_demo = [-pi/2, pi/3, -2*pi/3, pi/3, pi/3]
T_demo = dofbot.fkine(q_demo)
print("\n========== Part1-3 (pose 3) 正解 ==========")
print(T_demo)
dofbot.plot(q=q_demo, block=False)
# plt.savefig("figures/task_1_4", bbox_inches='tight', dpi=150)





# ==============================================
# 仿真任务2、 逆运动学 —— 给出DH模型在以下 4 组笛卡尔空间姿态下的逆运动学解
# ==============================================
# targets = [
#     # demo
#     np.array([
#         [-1., 0., 0., 0.1],
#         [ 0., 1., 0., 0. ],
#         [ 0., 0.,-1.,-0.1],
#         [ 0., 0., 0., 1. ]
#     ]),
#     # 1
#     np.array([
#         [1., 0., 0., 0.1],
#         [0., 1., 0., 0. ],
#         [0., 0., 1., 0.1],
#         [0., 0., 0., 1. ]
#     ]),
#     # 2
#     np.array([
#         [cos(pi/3), 0.,-sin(pi/3), 0.2],
#         [0.,        1., 0.,        0. ],
#         [sin(pi/3), 0., cos(pi/3), 0.2],
#         [0.,        0., 0.,        1. ]
#     ]),
#     # 3
#     np.array([
#         [-0.866, -0.25,  -0.433, -0.03704],
#         [ 0.5,   -0.433, -0.75,  -0.06415],
#         [ 0.,    -0.866,  0.5,    0.3073 ],
#         [ 0.,     0.,     0.,     1.     ]
#     ])
# ]

# -------- 2.1 demo 目标 ----------
T_des_demo = np.array([
    [-1., 0., 0., 0.1],
    [ 0., 1., 0., 0. ],
    [ 0., 0.,-1.,-0.1],
    [ 0., 0., 0., 1. ]
])
q_ik_demo = dofbot.ik_LM(T_des_demo)[0]   # 取返回元组第 0 个元素
print("\n========== Part2-0 (demo) 逆解 ==========")
print("关节角（rad）：", np.array(q_ik_demo))
dofbot.plot(q=q_ik_demo, block=False)
# plt.savefig("figures/task_2_1", bbox_inches='tight', dpi=150)

# -------- 2.2 目标 1 ----------
T_des_demo = np.array([
    [1., 0., 0., 0.1],
    [0., 1., 0., 0. ],
    [0., 0., 1., 0.1],
    [0., 0., 0., 1. ]
])
q_ik_demo = dofbot.ik_LM(T_des_demo)[0]   # 取返回元组第 0 个元素
print("\n========== Part2-1 (pose 1) 逆解 ==========")
print("关节角（rad）：", np.array(q_ik_demo))
dofbot.plot(q=q_ik_demo, block=False)
# plt.savefig("figures/task_2_2", bbox_inches='tight', dpi=150)

# -------- 2.3 目标 2 ----------
T_des_demo = np.array([
    [np.cos(pi/3), 0.,-np.sin(pi/3), 0.2],
    [0.,        1., 0.,        0. ],
    [np.sin(pi/3), 0., np.cos(pi/3), 0.2],
    [0.,        0., 0.,        1. ]
])
q_ik_demo = dofbot.ik_LM(T_des_demo)[0]   # 取返回元组第 0 个元素
print("\n========== Part2-1 (pose 2) 逆解 ==========")
print("关节角（rad）：", np.array(q_ik_demo))
dofbot.plot(q=q_ik_demo, block=False)
# plt.savefig("figures/task_2_3", bbox_inches='tight', dpi=150)

# -------- 2.4 目标 3 ----------
T_des_demo = np.array([
    [-0.866, -0.25,  -0.433, -0.03704],
    [ 0.5,   -0.433, -0.75,  -0.06415],
    [ 0.,    -0.866,  0.5,    0.3073 ],
    [ 0.,     0.,     0.,     1.     ]
])
q_ik_demo = dofbot.ik_LM(T_des_demo)[0]   # 取返回元组第 0 个元素
print("\n========== Part2-1 (pose 3) 逆解 ==========")
print("关节角（rad）：", np.array(q_ik_demo))
dofbot.plot(q=q_ik_demo, block=False)
# plt.savefig("figures/task_2_4", bbox_inches='tight', dpi=150)





# ==============================================
# 仿真任务3、 工作空间可视化（≥500 点）
#     关节限位（°）→ 弧度
#     J1: [-180, 180]  J2~J5: [0, 180]
# ==============================================
# 配置采样数量（≥500）
num_samples = 3000

# 关节限位（单位：rad）
j1_min, j1_max = -np.pi, np.pi # J1
j_min, j_max = 0.0, np.pi # J2~J5

# 在关节空间均匀随机采样
q1 = np.random.uniform(j1_min, j1_max, num_samples)
q2 = np.random.uniform(j_min, j_max, num_samples)
q3 = np.random.uniform(j_min, j_max, num_samples)
q4 = np.random.uniform(j_min, j_max, num_samples)
q5 = np.random.uniform(j_min, j_max, num_samples)
q_samples = np.vstack((q1, q2, q3, q4, q5)).T # 形状: (N, 5)

# 计算末端位置
points = np.zeros((num_samples, 3), dtype=float)
for i in range(num_samples):
    T = dofbot.fkine(q_samples[i])  # SE3
    points[i] = T.t.reshape(3)

# 绘制 3D 工作空间散点图
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=6, c=points[:, 2], cmap='viridis', alpha=0.6)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title(f'Dofbot Workspace (N={num_samples})')
cb = plt.colorbar(sc, ax=ax, pad=0.1)
cb.set_label('Z (m)')

# 设定等比例坐标轴，便于观察工作空间形状
x_min, x_max = points[:, 0].min(), points[:, 0].max()
y_min, y_max = points[:, 1].min(), points[:, 1].max()
z_min, z_max = points[:, 2].min(), points[:, 2].max()
max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
x_mid = (x_max + x_min) / 2.0
y_mid = (y_max + y_min) / 2.0
z_mid = (z_max + z_min) / 2.0
ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

# 保存图像
plt.tight_layout()
# plt.savefig("figures/task_3", bbox_inches='tight', dpi=150)
plt.show(block=False)