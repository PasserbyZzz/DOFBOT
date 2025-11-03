"""
Real Env
"""
import time
import numpy as np
import rospy
from sensor_msgs.msg import JointState

class RealEnv:
    # 面向真实机械臂的控制封装
    def __init__(self,init_node=True):
        if init_node:
            rospy.init_node("Arm2")  # 初始化 ROS 节点
        self.state = None
        self.pub = rospy.Publisher("/dofbot/cmd", JointState, queue_size=10)  # 发布者
        rospy.Subscriber("/dofbot/joint_state", JointState, self.callback)  # 订阅者

    def callback(self, data: JointState):
        # 回调函数，更新机械臂状态
        self.state = np.asarray(data.position)  # [angle1, angle2, angle3, angle4, angle5, gripper]

    def reset(self):
        # 复位机械臂到初始位置
        while self.state is None:
            time.sleep(0.1)
        self.send([90., 90., 90., 90., 90.], 10., 2000)
        time.sleep(2)

    def get_state(self):
        # 获取当前机械臂状态
        return self.state.copy()

    def send(self, angles, gripper, t):
        # 底层发送控制指令
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.position = np.round(angles).tolist() + [gripper] + [int(t)]  # [angle1, angle2, angle3, angle4, angle5] + [gripper] + [t]
        msg.name = [f"Joint{i}" for i in range(6)]

        self.pub.publish(msg)

    def control_gripper(self, g):
        # 控制夹爪
        self.send(self.state[:-1], g, 100)
        time.sleep(0.1)

    def control_joints(self, p):
        # 控制机械臂关节
        while not rospy.is_shutdown():
            self.send(p, self.state[-1], 150)
            time.sleep(0.15)
            if np.isclose(self.state[:-1], p, atol=3.).all():
                break

    def step(self, joint=None, gripper=None):
        # 执行一步控制
        if joint is not None:
            self.control_joints(joint)
        if gripper is not None:
            self.control_gripper(gripper)