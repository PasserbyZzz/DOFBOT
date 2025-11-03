import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import threading
import time

class TwoLinkArm:
    def __init__(self, l1, l2, q1_start=0.0, q2_start=0.0):
        self.l1 = l1
        self.l2 = l2
        self.q1 = q1_start
        self.q2 = q2_start

    def forward_kinematics(self, q1=None, q2=None):
        if q1 is None:
            q1 = self.q1
        if q2 is None:
            q2 = self.q2
        x = self.l1 * np.cos(q1) + self.l2 * np.cos(q1 + q2)
        y = self.l1 * np.sin(q1) + self.l2 * np.sin(q1 + q2)
        return x, y

    def inverse_kinematics(self, x, y):
        r = np.sqrt(x ** 2 + y ** 2)
        if r > self.l1 + self.l2:
            raise ValueError("Target out of reach")
        cos_q2 = (x ** 2 + y ** 2 - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2)
        q2 = np.arccos(np.clip(cos_q2, -1.0, 1.0))
        k1 = self.l1 + self.l2 * np.cos(q2)
        k2 = self.l2 * np.sin(q2)
        q1 = np.arctan2(y, x) - np.arctan2(k2, k1)
        return q1, q2

    def get_link_positions(self):
        x0, y0 = 0, 0
        x1 = self.l1 * np.cos(self.q1)
        y1 = self.l1 * np.sin(self.q1)
        x2 = x1 + self.l2 * np.cos(self.q1 + self.q2)
        y2 = y1 + self.l2 * np.sin(self.q1 + self.q2)
        return (x0, y0), (x1, y1), (x2, y2)

class ArmVisualizer:
    def __init__(self, arm, target_trajectory=None):
        self.arm = arm
        self.target_trajectory_func = target_trajectory
        self.actual_trajectory = []
        self.errors = []
        self.t = 0

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect('equal')
        self.ax.grid()

        base, elbow, ee = self.arm.get_link_positions()
        self.link1, = self.ax.plot([base[0], elbow[0]], [base[1], elbow[1]], 'b-', lw=2)
        self.link2, = self.ax.plot([elbow[0], ee[0]], [elbow[1], ee[1]], 'r-', lw=2)
        self.ee_circle = plt.Circle(ee, 0.05, color='k')
        self.ax.add_patch(self.ee_circle)

        # sliders
        axcolor = 'lightgoldenrodyellow'
        ax_q1 = plt.axes([0.25, 0.02, 0.65, 0.02], facecolor=axcolor)
        ax_q2 = plt.axes([0.25, 0.06, 0.65, 0.02], facecolor=axcolor)
        self.slider_q1 = Slider(ax_q1, 'q1', -np.pi, np.pi, valinit=self.arm.q1)
        self.slider_q2 = Slider(ax_q2, 'q2', -np.pi, np.pi, valinit=self.arm.q2)

        self.slider_q1.on_changed(self.slider_update)
        self.slider_q2.on_changed(self.slider_update)

        if self.target_trajectory_func:
            t_vals = np.linspace(0, 2*np.pi, 200)
            traj_points = [self.target_trajectory_func(t) for t in t_vals]
            traj_x, traj_y = zip(*traj_points)
            self.ax.plot(traj_x, traj_y, 'g--', lw=1, label='Target Trajectory')

        self.actual_traj_line, = self.ax.plot([], [], 'm-', lw=1, label='EE Trajectory')

        self.ax.legend()
        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=50)

        plt.show()

    def slider_update(self, val):
        self.arm.q1 = self.slider_q1.val
        self.arm.q2 = self.slider_q2.val
        self.update_plot()

    def animate(self, i):
        self.update_plot()

    def update_plot(self):
        base, elbow, ee = self.arm.get_link_positions()
        self.link1.set_data([base[0], elbow[0]], [base[1], elbow[1]])
        self.link2.set_data([elbow[0], ee[0]], [elbow[1], ee[1]])
        self.ee_circle.center = ee

        if len(self.actual_trajectory) > 0:
            traj_x, traj_y = zip(*self.actual_trajectory)
            self.actual_traj_line.set_data(traj_x, traj_y)

        self.fig.canvas.draw_idle()

    def append_trajectory(self, ee_pos, target_pos):
        self.actual_trajectory.append(ee_pos)
        error = np.linalg.norm(np.array(ee_pos) - np.array(target_pos))
        self.errors.append(error)

    def plot_tracking_error(self):
        plt.figure()
        plt.plot(self.errors)
        plt.xlabel("Step")
        plt.ylabel("Tracking Error")
        plt.title("Trajectory Tracking Error Over Time")
        plt.grid()
        plt.show()

def run_tracking_control(arm, visualizer, target_trajectory, duration=20.0, dt=0.02):
    t = 0.0
    while t < duration:
        target_pos = target_trajectory(t)
        try:
            q1_des, q2_des = arm.inverse_kinematics(*target_pos)
        except ValueError:
            t += dt
            continue
        # simple PD control
        arm.q1 += 0.1 * (q1_des - arm.q1)
        arm.q2 += 0.1 * (q2_des - arm.q2)

        ee_pos = arm.forward_kinematics()
        visualizer.append_trajectory(ee_pos, target_pos)

        time.sleep(dt)
        t += dt

    visualizer.plot_tracking_error()

if __name__ == "__main__":
    def circular_trajectory(t):
        radius = 0.8
        center = np.array([0.5, 0.5])
        return center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)

    arm = TwoLinkArm(1.0, 1.0)
    visualizer = ArmVisualizer(arm, target_trajectory=circular_trajectory)

    t = 0.0
    dt = 0.1
    Kp = 0.1

    while plt.fignum_exists(visualizer.fig.number):
        target_pos = circular_trajectory(t)
        try:
            q1_des, q2_des = arm.inverse_kinematics(*target_pos)
            arm.q1 += Kp * (q1_des - arm.q1)
            arm.q2 += Kp * (q2_des - arm.q2)
        except ValueError:
            pass  # skip unreachable points

        ee_pos = arm.forward_kinematics()
        visualizer.append_trajectory(ee_pos, target_pos)
        visualizer.update_plot()

        t += dt
        time.sleep(dt)

    visualizer.plot_tracking_error()