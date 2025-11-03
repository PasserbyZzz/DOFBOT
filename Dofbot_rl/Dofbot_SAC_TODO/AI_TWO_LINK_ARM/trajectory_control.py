import numpy as np
import matplotlib.pyplot as plt
from TwoLinkArm import TwoLinkArm
from controller import PIDController

# 定义机械臂参数
l1 = 1.0
l2 = 1.0
arm = TwoLinkArm(l1, l2)

# 定义PID控制器参数
kp = 1.0
ki = 0.1
kd = 0.01
pid = PIDController(kp, ki, kd)

# 定义轨迹
t = np.linspace(0, 10, 100)
trajectory_x = np.sin(t)
trajectory_y = np.cos(t)

# 初始化关节角度
q1 = 0.0
q2 = 0.0
dt = 0.1

# 存储轨迹和实际位置
trajectory = np.vstack((trajectory_x, trajectory_y)).T
actual_positions = []

for i in range(len(t)):
    # 计算目标位置
    target_x = trajectory[i, 0]
    target_y = trajectory[i, 1]

    # 计算当前末端位置
    current_x, current_y = arm.forward_kinematics(q1, q2)

    # 计算误差
    error_x = target_x - current_x
    error_y = target_y - current_y

    # 更新PID控制器
    control_signal_x = pid.update(error_x, dt)
    control_signal_y = pid.update(error_y, dt)

    # 更新关节角度
    q1 += control_signal_x * dt
    q2 += control_signal_y * dt

    # 存储实际位置
    actual_positions.append([current_x, current_y])

# 绘制轨迹
plt.plot(trajectory_x, trajectory_y, label='Desired Trajectory')
actual_positions = np.array(actual_positions)
plt.plot(actual_positions[:, 0], actual_positions[:, 1], label='Actual Trajectory')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trajectory Tracking')
plt.show()