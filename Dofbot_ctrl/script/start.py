#!/usr/bin/env python3
# coding=utf-8
import time
from Arm_Lib import Arm_Device
import numpy as np
# 创建机械臂对象
import rospy
from sensor_msgs.msg import JointState
import threading


if __name__ == '__main__':
    # 初始化 ROS 节点
    rospy.init_node("Arm")
    # 创建机械臂对象
    Arm = Arm_Device()

    lock = threading.Lock()
    
    def callback(msg: JointState):
        # 回调函数，接收控制指令
        # 丢弃时间戳落后 1 秒以上的旧消息
        if np.abs(rospy.Time.now()-msg.header.stamp).to_sec() > 1.:
            return
        # 解包控制指令
        angles = np.asarray(msg.position)
        # 发送控制指令给机械臂
        with lock:
            Arm.Arm_serial_servo_write6(angles[0],angles[1], angles[2], angles[3], angles[4], angles[5],
                                int(angles[6]))

    time.sleep(.1)

    threading.Thread(target=rospy.spin).start()

    rospy.Subscriber("/dofbot/cmd", JointState, callback)  # 订阅者
    pub = rospy.Publisher("/dofbot/joint_state", JointState, queue_size=10)  # 发布者

    rate = rospy.Rate(100)

    while not rospy.is_shutdown():

        with lock:
            # 读取机械臂当前状态
            angles = [Arm.Arm_serial_servo_read(i+1) for i in range(6)]
            if None in angles:
                continue

        # 组装消息并发布
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.position = angles
        msg.name = [f"Joint{i}" for i in range(6)]

        pub.publish(msg)

        rate.sleep()

    rospy.signal_shutdown("kill")

    del Arm
