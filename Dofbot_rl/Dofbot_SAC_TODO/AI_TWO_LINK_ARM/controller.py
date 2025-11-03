import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd):
        """
        初始化PID控制器
        :param kp: 比例系数
        :param ki: 积分系数
        :param kd: 微分系数
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_integral = 0
        self.previous_error = 0

    def update(self, error, dt):
        """
        更新PID控制器
        :param error: 当前误差
        :param dt: 时间步长
        :return: 控制信号
        """
        self.error_integral += error * dt
        error_derivative = (error - self.previous_error) / dt
        self.previous_error = error
        return self.kp * error + self.ki * self.error_integral + self.kd * error_derivative