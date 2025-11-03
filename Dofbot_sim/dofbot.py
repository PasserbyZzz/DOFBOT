import time

import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R

class Observation:
    def __init__(self, pos=None, orn=None, euler=None):
        # 观测数据类
        self.pos = pos # 位置(x,y,z)
        self.orn = orn # 四元数姿态(x,y,z,w)
        self.euler = euler # 欧拉角(RPY)


class dofbot:
    # 机械臂类
    def __init__(self, urdfPath):
        # # upper limits for null space
        self.ll = [-np.pi, 0, 0, 0, 0]
        # upper limits for null space
        self.ul = [np.pi, np.pi, np.pi, np.pi, np.pi]

        # joint ranges for null space
        self.jr = [np.pi * 2.0, np.pi, np.pi, np.pi, np.pi]
        # rest poses for null space
        self.rp = [np.pi / 2.0, np.pi / 2.0, np.pi / 2.0, np.pi / 2.0, np.pi / 2.0]

        self.maxForce = 200.
        self.fingerAForce = 2.5
        self.fingerBForce = 2.5
        self.fingerTipForce = 2

        self.dofbotUid = p.loadURDF(urdfPath,baseOrientation =p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
        # self.numJoints = p.getNumJoints(self.dofbotUid)
        self.numJoints = 5
        self.gripper_joints = [5, 6, 7, 8, 9, 10]

        self.jointStartPositions = [1.57, 1.57, 1.57, 1.57, 1.57]
        self.gripperAngle = 0.0

        self.motorIndices = []
        for jointIndex in range(self.numJoints):
            p.resetJointState(self.dofbotUid, jointIndex, self.jointStartPositions[jointIndex])
            qIndex = p.getJointInfo(self.dofbotUid, jointIndex)[3]
            if qIndex > -1:
                self.motorIndices.append(jointIndex)

        self.jointPositions = self.get_jointPoses()

        self.gripperStartAngle = 0.0
        for i, jointIndex in enumerate(self.gripper_joints):
            p.resetJointState(self.dofbotUid, jointIndex, self.gripperStartAngle)

        # 允许 self-collision
        for i in range(p.getNumJoints(self.dofbotUid)):
            p.setCollisionFilterGroupMask(self.dofbotUid, i,
                                          collisionFilterGroup=1,
                                          collisionFilterMask=1)
        # 让相邻连杆之间也产生碰撞（可选，视 URDF 具体关节类型而定）
        p.setCollisionFilterPair(self.dofbotUid, self.dofbotUid,
                                 linkIndexA=-1, linkIndexB=0, enableCollision=1)


        self.endEffectorPos = []
        self.endEffectorOrn = []
        self.endEffectorEuler = []
        self.endEffectorPos, self.endEffectorOrn, self.endEffectorEuler = self.get_pose()

    def reset(self):
        # 重置5个关节到初始姿态、夹爪角到 0，刷新缓存的末端位姿
        self.gripperAngle = 0.0
        for jointIndex in range(self.numJoints):
            p.resetJointState(self.dofbotUid, jointIndex, self.jointStartPositions[jointIndex])
        for i, jointIndex in enumerate(self.gripper_joints):
            p.resetJointState(self.dofbotUid, jointIndex, self.gripperAngle)
        self.jointPositions = self.get_jointPoses()
        self.endEffectorPos, self.endEffectorOrn, self.endEffectorEuler = self.get_pose()

    def forwardKinematic(self,jointPoses):
        # 直接设置关节角度，计算并返回末端位姿
        for i in range(self.numJoints):
            p.resetJointState(self.dofbotUid,
                              jointIndex=i,targetValue=jointPoses[i],targetVelocity=0)
        return self.get_pose()


    def joint_control(self,jointPoses):
        # 以 POSITION_CONTROL 模式驱动 5 个关节到目标角度（可设置力、速度、增益）
        for i in range(self.numJoints):
            p.setJointMotorControl2(bodyUniqueId=self.dofbotUid, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i], targetVelocity=0, force=200,
                                    maxVelocity=1.0, positionGain=0.3, velocityGain=1)
        # self.jointPositions, self.gripperAngle = self.get_jointPoses()
        # self.endEffectorPos, self.endEffectorOrn, self.endEffectorEuler = self.get_pose()
        # return self.endEffectorPos, self.endEffectorOrn, self.endEffectorEuler

    def setInverseKine(self, pos, orn):
        # 计算逆运动学，返回 5 个关节角度和当前夹爪角度
        if orn == None:
            jointPoses1 = p.calculateInverseKinematics(self.dofbotUid, 6, pos,
                                                      self.ll, self.ul, self.jr, self.rp)
            jointPoses2 = p.calculateInverseKinematics(self.dofbotUid, 8, pos,
                                                      self.ll, self.ul, self.jr, self.rp)
            jointPoses = [(x + y) / 2 for x, y in zip(jointPoses1, jointPoses2)]
            jointPoses3 = p.calculateInverseKinematics(self.dofbotUid, 4, pos,
                                                      self.ll, self.ul, self.jr, self.rp)
            jointPoses[self.numJoints - 1] = jointPoses3[self.numJoints - 1]
        else:
            jointPoses1 = p.calculateInverseKinematics(self.dofbotUid, 6, pos, orn,
                                                      self.ll, self.ul, self.jr, self.rp)
            jointPoses2 = p.calculateInverseKinematics(self.dofbotUid, 8, pos, orn,
                                                      self.ll, self.ul, self.jr, self.rp)
            jointPoses = [(x + y) / 2 for x, y in zip(jointPoses1, jointPoses2)]
            jointPoses3 = p.calculateInverseKinematics(self.dofbotUid, 4, pos,
                                                       self.ll, self.ul, self.jr, self.rp)
            jointPoses[self.numJoints - 1] = jointPoses3[self.numJoints - 1]

        return jointPoses[:self.numJoints], self.gripperAngle

    def get_jointPoses(self):
        # 获取当前 5 个关节角度和夹爪角度
        jointPoses= []
        for i in range(self.numJoints+1):
            state = p.getJointState(self.dofbotUid, i)
            jointPoses.append(state[0])
        return jointPoses[:self.numJoints], self.gripperAngle

    def get_pose(self):
        # 获取当前末端位姿（夹爪中心位置和朝向）
        # 1. 收集 6 个 link 的位姿
        indices = [6, 8]
        positions = []
        quaternions = []

        for idx in indices:
            link_state = p.getLinkState(self.dofbotUid, idx)
            positions.append(np.array(link_state[0]))
            quaternions.append(np.array(link_state[1]))

        # 2. 平均位置
        avg_pos = np.mean(positions, axis=0)

        # 3. 平均朝向（四元数）
        rotations = R.from_quat(quaternions)  # scipy 自动归一化
        avg_rot = rotations.mean()
        avg_orn = avg_rot.as_quat()  # [x,y,z,w] 格式

        # 现在 avg_pos 和 avg_orn 就是“夹爪”整体的均值位姿
        pos = avg_pos
        orn = avg_orn
        euler = p.getEulerFromQuaternion(orn)
        return pos, orn, euler

        # state = p.getLinkState(self.dofbotUid, 4)
        # pos = state[0]
        # orn = state[1]
        # euler = p.getEulerFromQuaternion(orn)
        # return pos, orn, euler


    def getObservation(self):
        # 获取当前末端位姿观测
        pos, orn, euler = self.get_pose()
        return Observation(pos, orn, euler)

    def gripper_control(self, gripperAngle):
        # 对 6 个夹爪关节（5..10）下发位置控制到同一角度（正开负合），并更新 self.gripperAngle
        p.setJointMotorControl2(self.dofbotUid,
                                5,
                                p.POSITION_CONTROL,
                                targetPosition=gripperAngle,
                                force=self.fingerAForce)
        p.setJointMotorControl2(self.dofbotUid,
                                6,
                                p.POSITION_CONTROL,
                                targetPosition=gripperAngle,
                                force=self.fingerBForce)
        p.setJointMotorControl2(self.dofbotUid,
                                7,
                                p.POSITION_CONTROL,
                                targetPosition=gripperAngle,
                                force=self.fingerAForce)
        p.setJointMotorControl2(self.dofbotUid,
                                8,
                                p.POSITION_CONTROL,
                                targetPosition=gripperAngle,
                                force=self.fingerBForce)
        p.setJointMotorControl2(self.dofbotUid,
                                9,
                                p.POSITION_CONTROL,
                                targetPosition=gripperAngle,
                                force=self.fingerAForce)
        p.setJointMotorControl2(self.dofbotUid,
                                10,
                                p.POSITION_CONTROL,
                                targetPosition=gripperAngle,
                                force=self.fingerBForce)

        self.gripperAngle = gripperAngle


class Object:
    # 物体类
    def __init__(self, urdfPath, block, num):
        self.id = p.loadURDF(urdfPath)
        self.half_height = 0.015 if block else 0.0745
        self.num = num
        self.size = self._get_size()  # 自己实现的函数，返回 [x, y, z] 半尺寸
        self.block = block

    def reset(self):
        # 重置物体位置和朝向
        if self.num == 1:
            p.resetBasePositionAndOrientation(self.id,
                                         np.array([ 0.20, 0.1, self.half_height]),
                                        p.getQuaternionFromEuler([0, 0,np.pi/6]))
        else:
            p.resetBasePositionAndOrientation(self.id,
                                         np.array([ 0.2, -0.1, 0.005]),
                                        p.getQuaternionFromEuler([0, 0,0]))

    def getObservation(self):
        # 获取物体当前位姿观测
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        return Observation(pos, orn, euler)

    def _get_size(self):
        """
        对 URDF 里第一个 'box' 几何体提取半尺寸 (halfExtents)。
        如果视觉和碰撞都有 box，优先用碰撞。
        """
        # 1. 先拿碰撞形状
        n_col = p.getCollisionShapeData(self.id, -1)  # base link
        if n_col and n_col[0][2] == p.GEOM_BOX:  # [0][2] == shapeType
            return list(n_col[0][3])  # [0][3] == halfExtents

        # 2. 没有就遍历所有 link 的碰撞形状
        for link_id in range(-1, p.getNumJoints(self.id)):  # -1 代表 base
            col_info = p.getCollisionShapeData(self.id, link_id)
            for info in col_info:
                if info[2] == p.GEOM_BOX:
                    return list(info[3])  # halfExtents

        # 3. 碰撞没有就找视觉形状（视觉形状没有 halfExtents 接口，只能解析 URDF）
        #    这里简单抛出异常，也可以自己解析 xml
        raise RuntimeError("未找到任何 box 几何体，无法自动获取 size")

    def pos_and_orn(self):
        # 获取物体当前位姿（位置、朝向、欧拉角）
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        return pos, orn, euler


def any_self_collision(robot_uid, safety_margin=0.0):
    """
    返回 True  ->  机器人内部至少有一对连杆发生碰撞
    safety_margin：允许的最小距离，<0 表示允许轻微穿透
    """
    pts = p.getClosestPoints(bodyA=robot_uid, bodyB=robot_uid,
                             distance=safety_margin,
                             physicsClientId=0)
    # 过滤掉“连杆自己跟自己”或“固定关节父-子”产生的无效点对
    for pt in pts:
        # pt[3] 是 linkIndexA，pt[4] 是 linkIndexB
        if pt[3] == pt[4]:
            continue          # 同一个几何体
        if abs(pt[3] - pt[4]) == 1:
            continue          # base->link0 是固定关节，通常不视为碰撞
        if pt[3] >= 5 or pt[4] >= 5:
            continue
        # 如果有其他“相邻连杆”也不需要检测，在这里继续过滤
        return True
    return False


def check_pairwise_collisions(bodies):
    # 检测一组物体之间是否有碰撞
    for body1 in bodies:
        for body2 in bodies:
            if body1 != body2 and \
                    len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=0., physicsClientId=0)) != 0:
                return True
    return False


class DofbotEnv:
    # Dofbot 环境类
    def __init__(self, physicsClientId=None):
        self._timeStep = 0.001
        # 如果外部已经连好，直接用；否则默认老行为（兼容旧代码）
        if physicsClientId is None:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = physicsClientId
        p.resetDebugVisualizerCamera(1.0, 90, -40, [0, 0, 0])
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -9.8)


        p.loadURDF("models/floor.urdf", [0, 0, -0.625], useFixedBase=True)
        p.loadURDF("models/table_collision/table.urdf", [0.5, 0, -0.625],p.getQuaternionFromEuler([0, 0, 0]),
                   useFixedBase=True)
        self._dofbot = dofbot("models/dofbot_urdf_with_gripper/dofbot_with_gripper.urdf")
        self._object1 = Object("models/box_green.urdf", block=True, num=1)
        self._object2 = Object("models/box_purple.urdf", block=True, num=2)


        self.target_pos = np.array([0.2, -0.1, 0.015])
        # # 创建红色目标球（无碰撞，仅视觉）
        # target_vis = p.createVisualShape(
        #     shapeType=p.GEOM_SPHERE,
        #     radius=0.005,  # 0.5 cm 小球，可按需调大
        #     rgbaColor=[1, 0, 0, 0.9]  # 红色
        # )
        # # 如果想彻底去掉碰撞，可以把碰撞形状设成一个很小的远点
        # target_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.001)  # 几乎不占地
        # self.target_body_id = p.createMultiBody(
        #     baseMass=0,  # 固定不动
        #     baseCollisionShapeIndex=target_col,
        #     baseVisualShapeIndex=target_vis,
        #     basePosition=self.target_pos  # 放在目标位置
        # )

        # self.end_effector_pos = np.array(self._dofbot.endEffectorPos)
        # # 创建红色目标球（无碰撞，仅视觉）
        # end_vis = p.createVisualShape(
        #     shapeType=p.GEOM_SPHERE,
        #     radius=0.005,  # 5 mm 小球，可按需调大
        #     rgbaColor=[0, 0, 1, 0.9]  # 蓝色
        # )
        # # 如果想彻底去掉碰撞，可以把碰撞形状设成一个很小的远点
        # end_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.001)  # 几乎不占地
        # self.end_body_id = p.createMultiBody(
        #     baseMass=0,  # 固定不动
        #     baseCollisionShapeIndex=end_col,
        #     baseVisualShapeIndex=end_vis,
        #     basePosition=self.end_effector_pos  # 放在目标位置
        # )

        # === 新增：创建滑动条控制关节和夹爪 ===
        self.sliders = []
        joint_names = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5"]
        for i in range(5):
            slider = p.addUserDebugParameter(joint_names[i],
                                             self._dofbot.ll[i],
                                             self._dofbot.ul[i],
                                             self._dofbot.rp[i])
            self.sliders.append(slider)
        self.gripper_slider = p.addUserDebugParameter("Gripper", -1.0, 1.0, 0.0)

        # 添加控制模式滑动条：0 = Joint Control, 1 = EE Pose Control
        self.control_mode_slider = p.addUserDebugParameter("Control_Mode(0=Joint,1=EE)", 0, 1, 0)

        self.ee_text_ids = []  # 每帧用来更新文字的句柄
        text_pos = [0.5, 0.05, 0.6]  # 左上角，按自己视角调
        text_quat = [0.5, 0.05, 0.5]  # 左上角，按自己视角调
        text_euler = [0.5, 0.05, 0.4]  # 左上角，按自己视角调
        line_h = 0.03  # 行间距

        # 初始占位，内容马上会被刷新
        self.ee_text_ids.append(self._make_item("EE  pos:  waiting...", text_pos, [1, 0, 0]))
        self.ee_text_ids.append(self._make_item("EE quat:  waiting...", text_quat, [0, 1, 0]))
        self.ee_text_ids.append(self._make_item("EE euler: waiting...", text_euler, [0, 0, 1]))

        self.end_effector_arrow_id = None

        self.object_arrow_id = None

        self.control_mode = 1

    def update_arrow_display(self, pos, orn):
        # 显示末端坐标系箭头
        arrow_start = pos

        # 长度可自由调节
        arrow_length = 0.05

        # 分别旋转单位向量 [1,0,0], [0,1,0], [0,0,1]（分别对应 X, Y, Z）
        x_dir = p.multiplyTransforms([0, 0, 0], orn, [arrow_length, 0, 0], [0, 0, 0, 1])[0]
        y_dir = p.multiplyTransforms([0, 0, 0], orn, [0, arrow_length, 0], [0, 0, 0, 1])[0]
        z_dir = p.multiplyTransforms([0, 0, 0], orn, [0, 0, arrow_length], [0, 0, 0, 1])[0]

        arrow_end_x = [arrow_start[i] + x_dir[i] for i in range(3)]
        arrow_end_y = [arrow_start[i] + y_dir[i] for i in range(3)]
        arrow_end_z = [arrow_start[i] + z_dir[i] for i in range(3)]

        arrow_items = []
        arrow_items.append(p.addUserDebugLine(
            arrow_start, arrow_end_x, [1, 0, 0], lineWidth=3, lifeTime=0
        ))
        arrow_items.append(p.addUserDebugLine(
            arrow_start, arrow_end_y, [0, 1, 0], lineWidth=3, lifeTime=0
        ))
        arrow_items.append(p.addUserDebugLine(
            arrow_start, arrow_end_z, [0, 0, 1], lineWidth=3, lifeTime=0
        ))

        return arrow_items

    def _make_item(self, text, text_pos, rgb):
        # 创建单行文字显示，返回句柄
        return p.addUserDebugText(text, text_pos,
                                  textColorRGB=rgb,
                                  textSize=0.5,
                                  lifeTime=self._timeStep)  # 0.1 s 后自动消失，下一帧再写

    def _update_ee_text_window(self):
        # 更新末端位姿文字显示
        pos, orn, euler = self.get_dofbot_pose()

        new_txt = [
            f"EE  pos:  [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]",
            f"EE quat:  [{orn[0]:.3f}, {orn[1]:.3f}, {orn[2]:.3f}, {orn[3]:.3f}]",
            f"EE euler: [{euler[0]:.3f}, {euler[1]:.3f}, {euler[2]:.3f}]"
        ]
        rgb = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        # 删掉旧文字，写新文字
        for i in range(3):
            p.removeUserDebugItem(self.ee_text_ids[i])
            self.ee_text_ids[i] = p.addUserDebugText(
                new_txt[i],
                [-0.50, 0.30, 0.7 - i * 0.05],
                textColorRGB=rgb[i],
                textSize=1.2,
                lifeTime=0.1)

    def reset(self):
        # 重置环境
        self._object1.reset()
        self._object2.reset()
        self._dofbot.reset()
        p.stepSimulation()

    def step_with_sliders(self):
        # 环境步进，根据滑动条控制机械臂
        """
        根据控制模式滑动条切换：
        0 - 关节角度控制
        1 - 末端位姿控制
        """
        mode = p.readUserDebugParameter(self.control_mode_slider)

        if mode < 0.5:
            # === Joint Control Mode ===
            jointPoses = [p.readUserDebugParameter(slider) for slider in self.sliders]
            gripperAngle = p.readUserDebugParameter(self.gripper_slider)
            print(jointPoses)
            self.dofbot_forward_control(jointPoses, gripperAngle)
            # self._dofbot.joint_control(jointPoses)
            # self._dofbot.gripper_control(gripperAngle)
            # pos, orn, euler = self._dofbot.get_pose()
            # # print("pos: ", pos, "orn: ", orn, "euler: ", euler)
            # p.stepSimulation()
            # time.sleep(self._timeStep)
        else:
            mode = mode

        # self.end_effector_pos = self._dofbot.endEffectorPos
        # p.resetBasePositionAndOrientation(self.end_body_id, self.end_effector_pos, [0, 0, 0, 1])

        # 实时显示末端位置与朝向
        # 删除上次显示
        if self.end_effector_arrow_id is not None:
            for item in self.end_effector_arrow_id:
                p.removeUserDebugItem(item)

        if self.object_arrow_id is not None:
            for item in self.object_arrow_id:
                p.removeUserDebugItem(item)

        pos, orn, euler = self.get_dofbot_pose()
        self.update_arrow_display(pos, orn)

        object_pos, object_orn, object_euler = self._object1.pos_and_orn()

        self.end_effector_arrow_id = self.update_arrow_display(self._dofbot.endEffectorPos, self._dofbot.endEffectorOrn)
        self.object_arrow_id = self.update_arrow_display(object_pos, object_orn)
        self._update_ee_text_window()  # <-- 新增


    def dofbot_control(self,jointPoses,gripperAngle):
        # 控制机械臂运动
        '''
        :param jointPoses: 数组，机械臂五个关节角度
        :param gripperAngle: 浮点数，机械臂夹爪角度，负值加紧，真值张开
        :return:
        '''
        self._dofbot.joint_control(jointPoses)
        self._dofbot.gripper_control(gripperAngle)
        p.stepSimulation()
        time.sleep(self._timeStep)

    def dofbot_forward_control(self, jointPoses, gripperAngle):
        # 控制机械臂运动（正运动学）
        self._dofbot.forwardKinematic(jointPoses)
        self._dofbot.gripper_control(gripperAngle)
        p.stepSimulation()
        time.sleep(self._timeStep)

    def dofbot_setInverseKine(self,pos,orn = None):
        # 控制机械臂运动（逆运动学）
        '''
        :param pos: 机械臂末端位置，xyz
        :param orn: 机械臂末端方向，四元数
        :return: 机械臂各关节角度
        '''
        jointPoses = self._dofbot.setInverseKine(pos, orn)
        return jointPoses

    # def dofbot_forwardKine(self,jointStates):
    #     return self._dofbot.forwardKinematic(jointStates)

    def get_dofbot_jointPoses(self):
        # 获取机械臂当前关节角度
        '''
        :return: 机械臂五个关节位置+夹爪角度
        '''
        jointPoses, gripper_angle = self._dofbot.get_jointPoses()

        return jointPoses, gripper_angle

    def get_dofbot_pose(self):
        # 获取机械臂末端位姿
        '''
        :return: 机械臂末端位姿，xyz+四元数+欧拉角
        '''
        pos, orn, euler = self._dofbot.get_pose()
        return pos, orn, euler

    def get_block_pose(self):
        # 获取物块位姿
        '''
        :return: 物块位姿，xyz+四元数
        '''
        pos, orn, euler = self._object1.pos_and_orn()
        return pos, orn, euler

    def get_target_pose(self):
        # 获取目标位置
        '''
        :return: 目标位置，xyz
        '''
        return self.target_pos

    def set_target_pos(self, target_pos):
        # 设置目标位置
        self.target_pos = target_pos
        # p.resetBasePositionAndOrientation(self.target_body_id, target_pos, [0, 0, 0, 1])
    
    def reward(self):
        '''
        :return: 是否完成抓取放置
        '''
        pos, orn, euler = self._object1.pos_and_orn()
        dist = np.sqrt((pos[0] - self.target_pos[0]) ** 2 + (pos[1] - self.target_pos[1]) ** 2)
        if dist < 0.01 and pos[2] < 0.02:
            return True
        return False




