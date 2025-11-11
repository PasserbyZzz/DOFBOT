import time
import numpy as np
# 创建机械臂对象
import rospy
from dofbot_real import RealEnv


def linear_interpolation(src, tat, n=10):
    """简单的线性插值实现"""
    path = np.linspace(src, tat, num=n)
    return path

if __name__ == '__main__':
    # 调用 RealEnv
    env = RealEnv()
    env.reset()

    # 通过简单状态机来实现分段控制
    # 状态机的中间路点
    points = [
        np.asarray([90., 90., 90., 90., 90.]),    # INITIAL_STATE
        np.asarray([133., 48., 52., 2., 90.]),    # PRE_GRASP_STATE
        np.asarray([90., 71., 47., 10., 90.]),    # MOVE_STATE
        np.asarray([40., 57., 42., 7., 90.]),     # SET_STATE
        # np.asarray([90., 90., 90., 90., 90.]),  # BACK_SET_STATE
    ]

    for i in range(len(points) - 1):
        print("From State:", i, "to State:", i + 1)
        # 取出路径点并做路径规划得到路径
        path = linear_interpolation(points[i], points[i + 1], n=30)
        for p in path:
            # 执行路径上各点
            # env.step(joint=...)可以控制关节
            # env.step(gripper=...)可以控制夹爪
            # 建议分开控制

            # 只控制关节，夹爪保持不变
            env.step(joint=p, gripper=None)
        print("Reached State:", i + 1)

        # 关节控制结束后，根据需求控制夹爪
        if i == 0:
            # INITIAL_STATE 结束，进入 PRE_GRASP_STATE：夹爪线性插值到 120 进行夹取
            print("Gripper Close")
            path = linear_interpolation(env.get_state()[-1], 120., n=20)
            for g in path:
                env.step(joint=None, gripper=g)
            time.sleep(0.5)  # 夹取后稍作停顿
        elif i == 2:
            # MOVE_STATE 结束，进入 SET_STATE：夹爪线性插值到 94 进行放置
            print("Gripper Open")
            path = linear_interpolation(env.get_state()[-1], 94., n=20)
            for g in path:
                env.step(joint=None, gripper=g)
            time.sleep(0.5)  # 放置后稍作停顿

    env.reset()
    print("Task Completed")
