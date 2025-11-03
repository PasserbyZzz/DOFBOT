import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from typing import Optional
from gymnasium.envs.registration import register
import time

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


    def joint_control(self,dqpos):
        self.desire_qpos = self.desire_qpos + dqpos
        jointPoses = self.desire_qpos
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
    
    def get_qvel(self):
        jointVels= []
        for i in range(self.numJoints+1):
            state = p.getJointState(self.dofbotUid, i)
            jointVels.append(state[1])
        return np.array(jointVels[:self.numJoints])

    def update_arrow_display(self, pos, orn):
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
        dofbot_obs = dict()
        qpos, gripper = self.get_jointPoses()
        qpos.append(gripper)
        dofbot_obs["qpos"] = np.array(qpos)
        pos, orn, euler = self.get_pose()
        dofbot_obs["eepose"] =np.array(list(pos) + list(orn))
        # self.update_arrow_display(pos, orn)
        
        return dofbot_obs

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
            # p.resetBasePositionAndOrientation(self.id,
            #                              np.array([ 0.20, 0.1,
            #                                        self.half_height]),
            #                             p.getQuaternionFromEuler([0, 0,np.pi/6]))
            p.resetBasePositionAndOrientation(self.id,
                                         np.array([ 0.18, 0.07,
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

register(
    id="DofbotReachEnv-v1",
    entry_point="dofbotGymReachEnv:DofbotEnv",
    max_episode_steps=500,
)
class DofbotEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}
    def __init__(self, render_mode="human", physicsClientId=None):
        self._timeStep = 1 / 120
        self.simuRepeatNum = 5
        self.render_mode = render_mode
        # 如果外部已经连好，直接用；否则默认老行为（兼容旧代码）
        if physicsClientId is None:
            if render_mode == "human":
                self.physicsClient = p.connect(p.GUI)
            else:
                self.physicsClient = p.connect(p.DIRECT)
        else:
            self.physicsClient = physicsClientId
        p.resetDebugVisualizerCamera(1.0, 100, -20, [0, 0, 0])
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -9.81)


        p.loadURDF("models/floor.urdf", [0, 0, -0.625], useFixedBase=True)
        p.loadURDF("models/table_collision/table.urdf", [0.5, 0, -0.625],p.getQuaternionFromEuler([0, 0, 0]),
                   useFixedBase=True)
        self._dofbot = dofbot("models/dofbot_urdf_with_gripper/dofbot_with_gripper.urdf")
        self._object1 = Object("models/box_green.urdf", block=True,num=1)
        # self._object2 = Object("models/box_red.urdf", block=True,num=2)
        self.end_effector_arrow_id = None
        self.object_arrow_id = None

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

        # TODO: observation space and action space
        # self.observation_space =
        # self.action_space =

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._object1.reset()
        self._dofbot.reset()
        p.stepSimulation()
        obs = self._get_obs()
        info = self._get_info()
        return obs, info
    
    def is_grasped(self):
        min_force = 0.5
        contact_finger1 = p.getContactPoints(bodyA=self._dofbot.dofbotUid, bodyB=self._object1.id, linkIndexA=6)
        contact_finger2 = p.getContactPoints(bodyA=self._dofbot.dofbotUid, bodyB=self._object1.id, linkIndexA=8)
        if bool(contact_finger1):
            # print("contact_finger1", contact_finger1)
            # print("contact_finger1[7]", contact_finger1[0][7])
            normal1 = abs(contact_finger1[0][7][1])  ## y轴方向力
        else:
            normal1 = 0
        if bool(contact_finger2):
            normal2 = abs(contact_finger2[0][7][1])  ## y轴方向力
        else:
            normal2 = 0
        
        if (normal1 > min_force) and (normal2 > min_force):
            return True
        return False

    # TODO: 增加你觉得必要的info，计算奖励的时候可以调用，获取仿真环境中一些特权信息
    def _get_info(self):
        info = dict()

        return info
    
    def angle_diff(self, q1, q2):
        dot_product = np.dot(q1, q2)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_difference = 2 * np.arccos(dot_product)
        return angle_difference

    # TODO: 完善奖励函数
    def _get_reward(self):
        obs = self._get_obs_dict()
        info = self._get_info()
        
        reward = 0
        return reward
    
    def rotate2grasp_pose(self, pose):
        if pose.shape != (7,):
            raise ValueError("Input pose must be a 7D numpy array.") 
        
        position = pose[:3]  
        quaternion = pose[3:] 
        rotation = R.from_quat(quaternion)
        rotation_x_180 = R.from_euler('x', 180, degrees=True)
        new_rotation = rotation * rotation_x_180
        new_quaternion = new_rotation.as_quat()
        new_pose = np.concatenate([position, new_quaternion])
        return new_pose
    
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
    
    def _get_obs(self):
        Observation = self._get_obs_dict()

        values = list(Observation.values())
        self._observation = np.concatenate([v if isinstance(v, np.ndarray) else np.array([v], dtype=np.int32) for v in values])
        self._observation = self._observation.astype(np.float32)
        if self.end_effector_arrow_id is not None:
            for item in self.end_effector_arrow_id:
                p.removeUserDebugItem(item)

        if self.object_arrow_id is not None:
            for item in self.object_arrow_id:
                p.removeUserDebugItem(item)

        self.end_effector_arrow_id = self.update_arrow_display(Observation["eepose"][:3], Observation["eepose"][3:])
        self.object_arrow_id = self.update_arrow_display(Observation["grasp_pose"][:3], Observation["grasp_pose"][3:])
        return self._observation


    def _get_obs_dict(self):
        Observation = self._dofbot.getObservation()

        # TODO:完善observation

        return Observation

    def step(self, action):
        """
        action - np.array(5)
        """

        # TODO: 完善control指令

        for i in range(self.simuRepeatNum):
            p.stepSimulation()
        
        if self.render_mode == "human":
            time.sleep(self._timeStep)
        terminated = self._termination()
        truncated = False
        self._observation = self._get_obs()
        reward = self._get_reward()
        info = self._get_info()
        return self._observation, reward, terminated, truncated, info

    def _termination(self):
        info = self._get_info()
        if info["success"]:
            return True
        return False
    
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
    # def reward(self):
    #     '''
    #     :return: 是否完成抓取放置
    #     '''
    #     pos, orn, euler = self._object1.pos_and_orn()
    #     dist = np.sqrt((pos[0] - self.target_pos[0]) ** 2 + (pos[1] - self.target_pos[1]) ** 2)
    #     if dist < 0.01 and pos[2] < 0.02:
    #         return True
    #     return False




