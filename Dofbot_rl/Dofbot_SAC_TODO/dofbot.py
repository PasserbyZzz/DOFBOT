import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R

class Observation:
    def __init__(self, pos=None, orn = None, euler=None):
        self.pos = pos
        self.orn = orn
        self.euler = euler


class dofbot:
    def __init__(self, urdfPath):
        # # upper limits for null space
        self.ll = [-np.pi, 0, 0, 0, -np.pi]
        # upper limits for null space
        self.ul = [np.pi, np.pi, np.pi, np.pi, np.pi]

        # joint ranges for null space
        self.jr = [np.pi * 2.0, np.pi, np.pi, np.pi, 2.0 * np.pi]
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

        self.jointStartPositions = [1.57,  1, 1.57, 1.57, 1.57]
        # self.jointStartPositions = [1.57, 1.57, 1.57, 1.57, 1.57]
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


        self.endEffectorPos = []
        self.endEffectorOrn = []
        self.endEffectorEuler = []
        self.endEffectorPos, self.endEffectorOrn, self.endEffectorEuler = self.get_pose()

    def reset(self):
        self.gripperAngle = 0.0
        for jointIndex in range(self.numJoints):
            p.resetJointState(self.dofbotUid, jointIndex, self.jointStartPositions[jointIndex])
        for i, jointIndex in enumerate(self.gripper_joints):
            p.resetJointState(self.dofbotUid, jointIndex, self.gripperAngle)
        self.jointPositions = self.get_jointPoses()
        self.endEffectorPos, self.endEffectorOrn, self.endEffectorEuler = self.get_pose()

    # def forwardKinematic(self,jointPoses):
    #     for i in range(self.numJoints):
    #         p.resetJointState(self.dofbotUid,
    #                           jointIndex=i,targetValue=jointPoses[i],targetVelocity=0)
    #     return self.get_pose()


    def joint_control(self,jointPoses):
        for i in range(self.numJoints):
            p.setJointMotorControl2(bodyUniqueId=self.dofbotUid, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i], targetVelocity=0, force=200,
                                    maxVelocity=10.0, positionGain=0.3, velocityGain=1)
        self.jointPositions, self.gripperAngle = self.get_jointPoses()
        self.endEffectorPos, self.endEffectorOrn, self.endEffectorEuler = self.get_pose()
        return self.endEffectorPos, self.endEffectorOrn, self.endEffectorEuler

    def setInverseKine(self, pos, orn):
        if orn == None:
            jointPoses = p.calculateInverseKinematics(self.dofbotUid, 4, pos,
                                                      self.ll, self.ul, self.jr, self.rp)
        else:
            jointPoses = p.calculateInverseKinematics(self.dofbotUid, 4, pos, orn,
                                                      self.ll, self.ul, self.jr, self.rp)
        return jointPoses[:self.numJoints], self.gripperAngle

    def get_jointPoses(self):
        jointPoses= []
        for i in range(self.numJoints+1):
            state = p.getJointState(self.dofbotUid, i)
            jointPoses.append(state[0])
        return jointPoses[:self.numJoints], self.gripperAngle

    def get_pose(self):
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

    def getObservation(self):
        pos, orn, euler = self.get_pose()
        return Observation(pos, orn, euler)

    def gripper_control(self, gripperAngle):

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
                                force=self.fingerAForce)

        self.gripperAngle = gripperAngle


class Object:
    def __init__(self, urdfPath, block,num):
        self.id = p.loadURDF(urdfPath)
        self.half_height = 0.015 if block else 0.0745
        self.num = num

        self.block = block
    def reset(self):

        if self.num==1:
            p.resetBasePositionAndOrientation(self.id,
                                         np.array([ 0.20, 0.1,
                                                   self.half_height]),
                                        p.getQuaternionFromEuler([0, 0,np.pi/6]))
        else:
            p.resetBasePositionAndOrientation(self.id,
                                         np.array([ 0.2, -0.1,
                                                   0.005]),
                                        p.getQuaternionFromEuler([0, 0,0]))

    def getObservation(self):
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        return Observation(pos, orn, euler)

    def pos_and_orn(self):
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        return pos, orn, euler


def check_pairwise_collisions(bodies):
    for body1 in bodies:
        for body2 in bodies:
            if body1 != body2 and \
                    len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=0., physicsClientId=0)) != 0:
                return True
    return False


class DofbotEnv:
    def __init__(self, physicsClientId=None):
        self._timeStep = 0.01
        # 如果外部已经连好，直接用；否则默认老行为（兼容旧代码）
        if physicsClientId is None:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = physicsClientId
        p.resetDebugVisualizerCamera(1.0, 120, -20, [0.2, 0, 0.2])
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -9.81)


        p.loadURDF("models/floor.urdf", [0, 0, -0.625], useFixedBase=True)
        p.loadURDF("models/table_collision/table.urdf", [0.5, 0, -0.625],p.getQuaternionFromEuler([0, 0, 0]),
                   useFixedBase=True)
        self._dofbot = dofbot("models/dofbot_urdf_with_gripper/dofbot_with_gripper.urdf")
        self._object1 = Object("models/box_green.urdf", block=True,num=1)
        # self._object2 = Object("models/box_red.urdf", block=True,num=2)


        self.target_pos = np.array([0.2, -0.15, 0.15])
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

        self.end_effector_pos = np.array(self._dofbot.endEffectorPos)
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
        return p.addUserDebugText(text, text_pos,
                                  textColorRGB=rgb,
                                  textSize=0.5,
                                  lifeTime=self._timeStep)  # 0.1 s 后自动消失，下一帧再写

    def _update_ee_text_window(self):
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
        self._object1.reset()
        self._dofbot.reset()
        p.stepSimulation()

    def step(self, action):
        """
        action - np.array(5)
        """
        jointPoses, gripperAngle = self.get_dofbot_jointPoses()
        delta_q = action
        # desire_jointPoses = jointPoses + delta_q
        desire_jointPoses = self._dofbot.jointStartPositions
        print("desire_jointPoses:", desire_jointPoses)
        self._dofbot.joint_control(desire_jointPoses)
        self._dofbot.gripper_control(gripperAngle)
        p.stepSimulation()
        jointPoses, gripperAngle = self.get_dofbot_jointPoses()
        print("qpos:", jointPoses)
        pos, orn, euler = self._dofbot.get_pose()
        
        
        if self.end_effector_arrow_id is not None:
            for item in self.end_effector_arrow_id:
                p.removeUserDebugItem(item)

        if self.object_arrow_id is not None:
            for item in self.object_arrow_id:
                p.removeUserDebugItem(item)

        object_pos, object_orn, object_euler = self._object1.pos_and_orn()

        self.end_effector_arrow_id = self.update_arrow_display(self._dofbot.endEffectorPos, self._dofbot.endEffectorOrn)
        self.object_arrow_id = self.update_arrow_display(object_pos, object_orn)
        self._update_ee_text_window()  # <-- 新增
        
    def step_with_sliders(self):
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
            self._dofbot.joint_control(jointPoses)
            self._dofbot.gripper_control(gripperAngle)
            pos, orn, euler = self._dofbot.get_pose()
            print("pos: ", pos)
            jointPoses, gripperAngle = self.get_dofbot_jointPoses()
            print("qpos:", jointPoses)
            p.stepSimulation()
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

        object_pos, object_orn, object_euler = self._object1.pos_and_orn()

        self.end_effector_arrow_id = self.update_arrow_display(self._dofbot.endEffectorPos, self._dofbot.endEffectorOrn)
        self.object_arrow_id = self.update_arrow_display(object_pos, object_orn)
        self._update_ee_text_window()  # <-- 新增


    def dofbot_control(self,jointPoses,gripperAngle):
        '''
        :param jointPoses: 数组，机械臂五个关节角度
        :param gripperAngle: 浮点数，机械臂夹爪角度，负值加紧，真值张开
        :return:
        '''
        self._dofbot.joint_control(jointPoses)
        self._dofbot.gripper_control(gripperAngle)
        p.stepSimulation()
        # time.sleep(self._timeStep)

    def dofbot_setInverseKine(self,pos,orn = None):
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
        '''
        :return: 机械臂五个关节位置+夹爪角度
        '''
        jointPoses, gripper_angle = self._dofbot.get_jointPoses()

        return jointPoses, gripper_angle

    def get_dofbot_pose(self):
        '''
        :return: 机械臂末端位姿，xyz+四元数+欧拉角
        '''
        pos, orn, euler = self._dofbot.get_pose()
        return pos, orn, euler

    def get_block_pose(self):
        '''
        :return: 物块位姿，xyz+四元数
        '''
        pos, orn, euler = self._object1.pos_and_orn()
        return pos, orn, euler

    def get_target_pose(self):
        '''
        :return: 目标位置，xyz
        '''
        return self.target_pos

    def set_target_pos(self, target_pos):
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




