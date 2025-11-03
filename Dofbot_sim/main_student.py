from dofbot import DofbotEnv
import numpy as np
import copy
import time, os, datetime
import pybullet as p

# 1. 准备保存目录
save_dir = "results/record"
os.makedirs(save_dir, exist_ok=True)
mp4_path = os.path.join(save_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4")

if __name__ == '__main__':
    env = DofbotEnv()
    env.reset()
    Reward = False

    # 2. 开始录制
    log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,
                                 mp4_path,
                                 physicsClientId=env.physicsClient)

    '''
    constants here
    '''
    GRIPPER_DEFAULT_ANGLE = 20. / 180. * 3.1415 # 默认夹爪角度
    GRIPPER_CLOSE_ANGLE = -20. / 180. * 3.1415 # 夹爪闭合角度

    # define state machine
    PRE_GRASP_STATE = 0 # 预抓取状态
    GRASP_STATE = 1 # 抓取状态
    LIFT_STATE = 2 # 提起状态
    MOVE_STATE = 3 # 移动状态
    SET_STATE = 4 # 放置状态
    BACK_STATE = 5 # 返回状态
    current_state = PRE_GRASP_STATE

    # 到达/稳定判据
    EPS_POS = 0.025               # 末端到达位置阈值（2.5cm）
    STABLE_FRAMES = 30            # 连续满足帧数
    GRIP_CONTACT_MIN = 2          # 至少两个指爪与方块接触点
    SETTLE_LIN_VEL = 0.05         # 放置后物块线速度阈值
    SETTLE_ANG_VEL = 0.15         # 放置后物块角速度阈值
    RELEASE_STABLE_FRAMES = 15    # 放置稳定连续帧数

    arrive_cnt = 0
    grip_cnt = 0
    settle_cnt = 0

    # 偏移量
    obj_offset_grasp = [-0.013, -0.013, 0.045]  # 抓取偏移
    obj_offset_move = [0, 0, 0.145]             # 移动偏移
    obj_offset_set = [-0.013, 0.013, 0.045]     # 放置偏移

    # 物块尺寸
    print("object size: ", env._object1.size)  # [0.03, 0.03, 0.03]
    # 机械臂初始关节位置
    init_joint_pos, init_gripper_angle = env.get_dofbot_jointPoses()
    print("dofbot initial joint pose: ", init_joint_pos)

    # 获取物块位姿
    # 坐标：（0.2，0.1，0.015）欧拉角：（0，0，pi/6）尺寸：0.03
    block_pos, block_orn, block_euler = env.get_block_pose()
    # 获取目标位置
    # 坐标：（0.2, -0.1, 0.015）
    target_pos = env.get_target_pose()

    start_time = None

    time.sleep(1.0)
    num = 0
    state_num = 10

    while not Reward:
        # 计算当前阶段的末端目标
        if current_state == PRE_GRASP_STATE:
            desired_pos = np.array(block_pos) + obj_offset_grasp
            gripper_angle = GRIPPER_DEFAULT_ANGLE
        elif current_state == GRASP_STATE:
            desired_pos = np.array(block_pos) + obj_offset_grasp
            gripper_angle = GRIPPER_CLOSE_ANGLE
        elif current_state == LIFT_STATE:
            desired_pos = np.array(block_pos) + obj_offset_move
            gripper_angle = GRIPPER_CLOSE_ANGLE
        elif current_state == MOVE_STATE:
            desired_pos = np.array(target_pos) + obj_offset_move
            gripper_angle = GRIPPER_CLOSE_ANGLE
        elif current_state == SET_STATE: 
            desired_pos = np.array(target_pos) + obj_offset_set
            gripper_angle = GRIPPER_CLOSE_ANGLE
        else:  # BACK_STATE
            desired_pos = np.array(target_pos) + obj_offset_move
            gripper_angle = GRIPPER_DEFAULT_ANGLE

        print("Current State:", current_state, "Desired Pos:", desired_pos, "Gripper Angle:", gripper_angle)

        # IK 求解
        jointPoses, _ = env.dofbot_setInverseKine(desired_pos.tolist(), orn=None)
        # 位置控制
        env.dofbot_control(jointPoses, gripper_angle)

        # 到达判据
        current_pos, _, _ = env.get_dofbot_pose()
        if np.linalg.norm(np.array(current_pos) - desired_pos) < EPS_POS:
            arrive_cnt += 1
        else:
            arrive_cnt = 0

        # 稳定判据
        lin_vel, ang_vel = p.getBaseVelocity(env._object1.id)
        if np.linalg.norm(lin_vel) < SETTLE_LIN_VEL and np.linalg.norm(ang_vel) < SETTLE_ANG_VEL:
            settle_cnt += 1
        else:
            settle_cnt = 0

        # 指爪与方块接触判据
        cps = p.getContactPoints(bodyA=env._dofbot.dofbotUid, bodyB=env._object1.id)
        finger_links = {5, 6, 7, 8, 9, 10}
        contact_num = sum(1 for c in cps if (c[3] in finger_links) or (c[4] in finger_links))
        if contact_num >= GRIP_CONTACT_MIN:
            grip_cnt += 1
        else:
            grip_cnt = 0

        print("Arrive Count:", arrive_cnt, "/", STABLE_FRAMES, " Grip Count:", grip_cnt, "/", STABLE_FRAMES, " Settle Count:", settle_cnt, "/", RELEASE_STABLE_FRAMES)

        # 状态切换
        if current_state == PRE_GRASP_STATE and arrive_cnt >= STABLE_FRAMES:
            current_state = GRASP_STATE
            arrive_cnt = grip_cnt = settle_cnt = 0
        elif current_state == GRASP_STATE and grip_cnt >= STABLE_FRAMES:
            current_state = LIFT_STATE
            arrive_cnt = grip_cnt = settle_cnt = 0
        elif current_state == LIFT_STATE and arrive_cnt >= STABLE_FRAMES:
            current_state = MOVE_STATE
            arrive_cnt = grip_cnt = settle_cnt = 0
        elif current_state == MOVE_STATE and arrive_cnt >= STABLE_FRAMES:
            current_state = SET_STATE
            arrive_cnt = grip_cnt = settle_cnt = 0
        elif current_state == SET_STATE and settle_cnt >= RELEASE_STABLE_FRAMES:
            current_state = BACK_STATE
            arrive_cnt = grip_cnt = settle_cnt = 0
        elif current_state == BACK_STATE and arrive_cnt >= STABLE_FRAMES:
            Reward = env.reward()

    # 3. 结束录制
    p.stopStateLogging(log_id)