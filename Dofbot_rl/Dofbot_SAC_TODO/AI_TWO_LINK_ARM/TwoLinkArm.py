import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import threading
import time

class TwoLinkArm:
    def __init__(self, l1, l2, trajectory=None, trajectory_points=100):
        """
        初始化二连杆机械臂
        :param l1: 第一连杆长度
        :param l2: 第二连杆长度
        :param trajectory: 目标轨迹函数，返回目标位置 (x, y)
        :param trajectory_points: 目标轨迹的点数
        """
        self.l1 = l1
        self.l2 = l2
        self.q1 = 0.0  # 第一关节角度
        self.q2 = 0.0  # 第二关节角度
        self.lock = threading.Lock()  # 线程锁，确保线程安全
        self.trajectory = trajectory  # 目标轨迹函数
        self.t = 0.0  # 时间变量，用于轨迹计算
        self.trajectory_points = trajectory_points  # 目标轨迹的点数
        self.actual_trajectory = []  # 用于记录实际轨迹的列表

        # 启动更新线程
        self.update_thread = threading.Thread(target=self.update_loop)
        self.update_thread.daemon = True  # 设置为守护线程
        self.update_thread.start()

        # 启动可视化（在主线程中）
        self.visualize()

    def forward_kinematics(self):
        """
        正运动学计算
        :return: 末端执行器的位置 (x, y)
        """
        x = self.l1 * np.cos(self.q1) + self.l2 * np.cos(self.q1 + self.q2)
        y = self.l1 * np.sin(self.q1) + self.l2 * np.sin(self.q1 + self.q2)
        return x, y

    def inverse_kinematics(self, x, y):
        """
        逆运动学计算
        :param x: 目标位置 x
        :param y: 目标位置 y
        :return: 关节角度 q1, q2
        """
        # 计算距离平方
        r = x**2 + y**2
        # 计算 q2
        q2 = np.arccos((r - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2))
        # 计算 q1
        q1 = np.arctan2(y, x) - np.arctan2(self.l2 * np.sin(q2), self.l1 + self.l2 * np.cos(q2))
        return q1, q2

    def get_link_positions(self):
        """
        获取两连杆的端点位置
        :return: 两个连杆的端点位置
        """
        x1 = self.l1 * np.cos(self.q1)
        y1 = self.l1 * np.sin(self.q1)
        x2 = x1 + self.l2 * np.cos(self.q1 + self.q2)
        y2 = y1 + self.l2 * np.sin(self.q1 + self.q2)
        return (0, 0), (x1, y1), (x2, y2)

    def update_loop(self):
        """
        更新线程的主循环
        """
        while True:
            time.sleep(0.1)  # 模拟更新频率
            if self.trajectory:
                # 获取当前目标位置
                target_x, target_y = self.trajectory(self.t)
                self.t += 0.1  # 更新时间变量
                # 计算逆运动学
                self.q1, self.q2 = self.inverse_kinematics(target_x, target_y)
                # 记录实际轨迹
                actual_x, actual_y = self.forward_kinematics()
                self.actual_trajectory.append((actual_x, actual_y))
            self.update_plot()

    def visualize(self):
        """
        可视化二连杆机械臂
        """
        # 创建图形和轴
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect('equal')
        self.ax.grid()

        # 获取连杆位置
        base, elbow, end_effector = self.get_link_positions()

        # 绘制连杆
        self.link1, = self.ax.plot([base[0], elbow[0]], [base[1], elbow[1]], 'b-', lw=2)
        self.link2, = self.ax.plot([elbow[0], end_effector[0]], [elbow[1], end_effector[1]], 'r-', lw=2)

        # 添加关节和末端执行器
        self.base_circle = plt.Circle(base, 0.05, color='k')
        self.elbow_circle = plt.Circle(elbow, 0.05, color='k')
        self.end_effector_circle = plt.Circle(end_effector, 0.05, color='k')

        self.ax.add_patch(self.base_circle)
        self.ax.add_patch(self.elbow_circle)
        self.ax.add_patch(self.end_effector_circle)

        # 添加滑动块
        axcolor = 'lightgoldenrodyellow'
        ax_q1 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        ax_q2 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

        self.slider_q1 = Slider(ax_q1, 'Joint 1', -np.pi, np.pi, valinit=self.q1)
        self.slider_q2 = Slider(ax_q2, 'Joint 2', -np.pi, np.pi, valinit=self.q2)

        # 滑动块回调函数
        def update(val):
            with self.lock:
                self.q1 = self.slider_q1.val
                self.q2 = self.slider_q2.val
            self.update_plot()

        self.slider_q1.on_changed(update)
        self.slider_q2.on_changed(update)

        # 绘制目标轨迹
        if self.trajectory:
            t_values = np.linspace(0, 2 * np.pi, self.trajectory_points)
            target_x, target_y = zip(*[self.trajectory(t) for t in t_values])
            self.target_trajectory, = self.ax.plot(target_x, target_y, 'g--', lw=1.5, label='Target Trajectory')

        # 绘制实际轨迹
        self.actual_trajectory_line, = self.ax.plot([], [], 'm-', lw=1.5, label='Actual Trajectory')

        # 添加图例
        self.ax.legend()

        # 动画更新函数
        def animate(i):
            self.update_plot()

        self.ani = animation.FuncAnimation(self.fig, animate, interval=50)

        # 显示图形
        plt.show()

    def update_plot(self):
        """
        更新图形
        """
        with self.lock:
            base, elbow, end_effector = self.get_link_positions()

        self.link1.set_data([base[0], elbow[0]], [base[1], elbow[1]])
        self.link2.set_data([elbow[0], end_effector[0]], [elbow[1], end_effector[1]])
        self.base_circle.center = base
        self.elbow_circle.center = elbow
        self.end_effector_circle.center = end_effector

        # 更新实际轨迹
        if self.actual_trajectory:
            actual_x, actual_y = zip(*self.actual_trajectory)
            self.actual_trajectory_line.set_data(actual_x, actual_y)

        self.fig.canvas.draw_idle()

# 示例：创建二连杆机械臂并实时可视化
if __name__ == "__main__":
    # 定义一个圆形轨迹
    def circular_trajectory(t):
        radius = 1.0
        center_x, center_y = 0.5, 0.5
        return center_x + radius * np.cos(t), center_y + radius * np.sin(t)

    # 创建二连杆机械臂并指定轨迹
    arm = TwoLinkArm(l1=1.0, l2=1.0, trajectory=circular_trajectory, trajectory_points=100)