import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R

class Observation:
    # 观察类
    def __init__(self, pos=None, orn = None, euler=None):
        self.pos = pos # 位置
        self.orn = orn # 方向
        self.euler = euler # 欧拉角


class dofbot:
    # Dofbot 机械臂类
    def __init__(self, urdfPath):
        # lower limits for null space
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

        # self.jointStartPositions = [1.57, 0, 1.57, 1.57, 1.57]
        self.jointStartPositions = [1.57, 1, 1.57, 1.57, 1.57]
        self.desire_qpos = np.array(self.jointStartPositions)
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
        # 重置机械臂到初始姿态
        self.gripperAngle = 0.0
        for jointIndex in range(self.numJoints):
            p.resetJointState(self.dofbotUid, jointIndex, self.jointStartPositions[jointIndex])
        for i, jointIndex in enumerate(self.gripper_joints):
            p.resetJointState(self.dofbotUid, jointIndex, self.gripperAngle)
        self.jointPositions = self.get_jointPoses()
        self.endEffectorPos, self.endEffectorOrn, self.endEffectorEuler = self.get_pose()
        self.desire_qpos = np.array(self.jointStartPositions)

    # def forwardKinematic(self,jointPoses):
    #     for i in range(self.numJoints):
    #         p.resetJointState(self.dofbotUid,
    #                           jointIndex=i,targetValue=jointPoses[i],targetVelocity=0)
    #     return self.get_pose()


    def joint_control(self, dqpos):
        # 接收目标关节角度增量，关节位置控制（不包括夹爪）
        self.desire_qpos = self.desire_qpos + dqpos
        jointPoses = self.desire_qpos
        for i in range(self.numJoints):
            p.setJointMotorControl2(bodyUniqueId=self.dofbotUid, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i], targetVelocity=0, force=200,
                                    maxVelocity=10.0, positionGain=0.3, velocityGain=1)
        self.jointPositions, self.gripperAngle = self.get_jointPoses()
        self.endEffectorPos, self.endEffectorOrn, self.endEffectorEuler = self.get_pose()
        return self.endEffectorPos, self.endEffectorOrn, self.endEffectorEuler

    def setInverseKine(self, pos, orn=None):
        # 逆运动学解算，据目标末端位置和朝向计算对应的关节角度
        if orn is None:
            jointPoses = p.calculateInverseKinematics(self.dofbotUid, 4, pos,
                                                      self.ll, self.ul, self.jr, self.rp)
        else:
            jointPoses = p.calculateInverseKinematics(self.dofbotUid, 4, pos, orn,
                                                      self.ll, self.ul, self.jr, self.rp)
        return jointPoses[:self.numJoints], self.gripperAngle

    def get_jointPoses(self):
        # 获取当前所有关节的角度和夹爪角度
        jointPoses= []
        for i in range(self.numJoints + 1):
            state = p.getJointState(self.dofbotUid, i)
            jointPoses.append(state[0])
        return jointPoses[:self.numJoints], self.gripperAngle
    
    def get_qvel(self):
        # 获取当前所有关节的速度
        jointVels= []
        for i in range(self.numJoints+1):
            state = p.getJointState(self.dofbotUid, i)
            jointVels.append(state[1])
        return np.array(jointVels[:self.numJoints])

    def update_arrow_display(self, pos, orn):
        # 更新箭头的显示
        arrow_start = pos

        # 长度可自由调节
        arrow_length = 0.3
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
    
    def get_pose(self):
        # 获取末端执行器的位姿
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
        
        # 4. gripper pos
        grip_pos = R.from_quat(avg_orn).apply(np.array([0, 0, 0.02])) + avg_pos

        # 现在 avg_pos 和 avg_orn 就是“夹爪”整体的均值位姿
        pos = grip_pos
        orn = avg_orn
        euler = p.getEulerFromQuaternion(orn)
        return pos, orn, euler

    def getObservation(self):
        # 获取末端执行器的位姿观测
        dofbot_obs = dict()
        qpos, gripper = self.get_jointPoses()
        qpos.append(gripper)
        dofbot_obs["qpos"] = np.array(qpos) # 机械臂关节位置和夹爪角度（5+1维）
        pos, orn, euler = self.get_pose()
        dofbot_obs["eepose"] =np.array(list(pos) + list(orn)) # 末端执行器位置和朝向（3+4维）
        # self.update_arrow_display(pos, orn)
        
        return dofbot_obs

    def gripper_control(self, gripperAngle):
        # 夹爪控制
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