import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from typing import Optional
from gymnasium.envs.registration import register
import time
from dofbot import dofbot, Observation

register(
    id="DofbotReachEnv-v1",
    entry_point="dofbot_env:DofbotEnv",
    max_episode_steps=500,
)

class Object:
    # object 类
    def __init__(self, urdfPath, block,num):
        self.id = p.loadURDF(urdfPath)
        self.half_height = 0.015 if block else 0.0745
        self.num = num

        self.block = block

    def reset(self):
        # 重置物体位置
        if self.num == 1:
            # p.resetBasePositionAndOrientation(self.id,
            #                              np.array([0.18, 0.07,
            #                                        self.half_height]),
            #                             p.getQuaternionFromEuler([0, 0, np.pi/6]))
            p.resetBasePositionAndOrientation(self.id,
                                         np.array([0.3, 0.12,
                                                   self.half_height]),
                                        p.getQuaternionFromEuler([0, 0, np.pi/6]))
        else:
            p.resetBasePositionAndOrientation(self.id,
                                         np.array([0.2, -0.1,
                                                   0.005]),
                                        p.getQuaternionFromEuler([0, 0, 0]))

    def getObservation(self):
        # 获取物体的位姿观测
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        return Observation(pos, orn, euler)

    def pos_and_orn(self):
        # 获取物体的当前位置、四元数朝向和欧拉角
        pos, orn = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(orn)
        return pos, orn, euler


def check_pairwise_collisions(bodies):
    # 检查物体之间的碰撞
    for body1 in bodies:
        for body2 in bodies:
            if body1 != body2 and \
                    len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=0., physicsClientId=0)) != 0:
                return True
    return False


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
        p.loadURDF("models/table_collision/table.urdf", [0.5, 0, -0.625], p.getQuaternionFromEuler([0, 0, 0]),
                   useFixedBase=True)
        self._dofbot = dofbot("models/dofbot_urdf_with_gripper/dofbot_with_gripper.urdf")
        self._object1 = Object("models/box_green.urdf", block=True, num=1)
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
        # 观测空间: 关节位置(6维) + 末端位姿(7维) + 物体在末端坐标系下位姿(7维) = 20维
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        # 动作空间: 5个关节的角度
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # 重置环境
        super().reset(seed=seed)
        self.terminated = False
        self._object1.reset()
        self._dofbot.reset()
        p.stepSimulation()
        obs = self._get_obs()
        info = self._get_info()
        return obs, info
    
    def is_grasped(self):
        # 检查物体是否被夹持
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
        # 获取环境中的特权信息
        info = dict()
        # 计算末端到物体的距离
        maxDist = 0.01
        obs = self._get_obs_dict()
        ee_pos = obs["eepose"][:3]
        obj_pos, obj_orn, obj_euler = self._object1.pos_and_orn()
        ee_to_obj_dist = np.linalg.norm(ee_pos - obj_pos)
        # 如果距离小于阈值，视为任务成功
        info["success"] = ee_to_obj_dist <= maxDist
        return info
    
    def angle_diff(self, q1, q2):
        # 计算两个四元数之间的角度差
        dot_product = np.dot(q1, q2)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_difference = 2 * np.arccos(dot_product)
        return angle_difference

    # TODO: 完善奖励函数
    def _get_reward(self):
        # 计算奖励
        obs = self._get_obs_dict()
        info = self._get_info()
        
        ## reaching reward
        ee_pos = obs["eepose"][:3]
        obj_pos, obj_orn, obj_euler = self._object1.pos_and_orn()
        ee_to_obj_pos = ee_pos - obj_pos
        ee_to_object_dist = (ee_to_obj_pos[0]**2 + ee_to_obj_pos[1]**2 + ee_to_obj_pos[2]**2)**0.5
        r_reaching = -10 * ee_to_object_dist  ## 越接近物体reward越高

        ## quat reward
        ee_quat = obs["eepose"][3:]
        obj_pose = np.concatenate([obj_pos, obj_orn])
        grasp_pose = self.rotate2grasp_pose(obj_pose)
        angle_difference = self.angle_diff(ee_quat, grasp_pose[3:])
        r_quat = -1 * angle_difference  ## 角度差越小reward越高

        ## arrival reward
        r_arrival = 0
        if ee_to_object_dist < 0.01:
            r_arrival = 10

        reward = r_reaching + r_arrival + r_quat
        return reward
    
    def rotate2grasp_pose(self, pose):
        # 将姿态旋转180度以适应抓取姿势
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
        # 更新箭头的显示
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
        # 获取观测
        Observation = self._get_obs_dict()
        # 数据展平与拼接
        values = list(Observation.values())
        self._observation = np.concatenate([v if isinstance(v, np.ndarray) else np.array([v], dtype=np.int32) for v in values])
        self._observation = self._observation.astype(np.float32)
        if self.end_effector_arrow_id is not None:
            for item in self.end_effector_arrow_id:
                p.removeUserDebugItem(item)

        # 清除旧的可视化箭头
        if self.object_arrow_id is not None:
            for item in self.object_arrow_id:
                p.removeUserDebugItem(item)

        # 绘制新的可视化箭头
        # 末端箭头
        self.end_effector_arrow_id = self.update_arrow_display(Observation["eepose"][:3], Observation["eepose"][3:])
        # 抓取点箭头
        # obj_pos, obj_orn, obj_euler = self._object1.pos_and_orn()
        # obj_pose = np.concatenate([obj_pos, obj_orn])
        # grasp_pose = self.rotate2grasp_pose(obj_pose)
        # self.object_arrow_id = self.update_arrow_display(grasp_pose[:3], grasp_pose[3:])
        self.object_arrow_id = self.update_arrow_display(Observation["grasp_pose"][:3], Observation["grasp_pose"][3:])
        return self._observation

    def _get_obs_dict(self):
        # 获取观测字典
        # 机械臂关节位置和夹爪角度（5+1维），末端执行器位置和朝向（3+4维）
        Observation = self._dofbot.getObservation() 

        # TODO:完善observation
        # # 物体在机械臂夹爪坐标系下的位姿（3+4维）
        # obj_pos, obj_orn, obj_euler = self._object1.pos_and_orn() # 世界坐标系下物体位置和朝向 T world→gripper
        # inv_gripper_pos, inv_gripper_orn = p.invertTransform(Observation["eepose"][:3], Observation["eepose"][3:]) # T gripper→world
        # # T gripper→object =T gripper→world × T world→object
        # obj_rel_pos, obj_rel_orn = p.multiplyTransforms(inv_gripper_pos, inv_gripper_orn, obj_pos, obj_orn) # 物体在夹爪坐标系下的位置和朝向
        # Observation["obj_pose_in_gripper"] = np.concatenate((obj_rel_pos, obj_rel_orn))

        # 期望抓取位姿（3+4维）
        obj_pos, obj_orn, obj_euler = self._object1.pos_and_orn()
        obj_pose = np.concatenate([obj_pos, obj_orn])
        grasp_pose = self.rotate2grasp_pose(obj_pose)
        Observation["grasp_pose"] = grasp_pose

        return Observation

    def step(self, action):
        # 环境步进
        # action - np.array(5)

        # 限制动作范围
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # 将归一化的动作转换为实际的关节角度增量
        scale = 0.05
        dqpos = action * scale
        self._dofbot.joint_control(dqpos)

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
        # 判断当前 Episode 是否结束
        info = self._get_info()
        if info["success"]:
            return True
        return False
    
    # def _termination(self):
    #     # 判断当前 Episode 是否结束
    #     if self.terminated:
    #         return True
        
    #     # 计算末端到物体的距离
    #     maxDist = 0.01
    #     obs = self._get_obs_dict()
    #     ee_pos = obs["eepose"][:3]
    #     obj_pos, obj_orn, obj_euler = self._object1.pos_and_orn()
    #     ee_to_obj_dist = np.linalg.norm(ee_pos - obj_pos)
    #     # 如果距离小于阈值，触发抓取尝试
    #     if ee_to_obj_dist < maxDist:
    #         self.terminated = True
    #         print("terminating, closing gripper, attempting grasp")
            
    #         # 执行闭合夹爪动作
    #         for i in range(20):
    #             angle = -0.05 * i # 0 to -1.0 approx
    #             self._dofbot.gripper_control(angle)
    #             p.stepSimulation()
    #             if self.render_mode == "human":
    #                 time.sleep(self._timeStep)
            
    #         # # Ensure fully closed/tight
    #         # self._dofbot.gripper_control(-1.5) # Tighten
    #         # for _ in range(20):
    #         #     p.stepSimulation()

    #         # 执行抬升动作
    #         current_pos, current_orn, _ = self._dofbot.get_pose()
    #         target_pos = current_pos.copy()
    #         target_pos[2] += 0.1
            
    #         # 线性插值
    #         steps = 50
    #         for i in range(steps):
    #             alpha = (i + 1) / steps
    #             interp_pos = current_pos * (1 - alpha) + target_pos * alpha
    #             # IK
    #             jointPoses, _ = self._dofbot.setInverseKine(interp_pos, current_orn)
    #             # Apply joint control
    #             for j in range(5):
    #                 p.setJointMotorControl2(self._dofbot.dofbotUid, j, p.POSITION_CONTROL, jointPoses[j], force=self._dofbot.maxForce)
                
    #             # # Keep gripper closed
    #             # self._dofbot.gripper_control(-1.5)
                
    #             p.stepSimulation()
    #             if self.render_mode == "human":
    #                 time.sleep(self._timeStep)

    #         # 检查物体高度是否超过 0.05m，若超过则视为成功
    #         obj_pos_new, _, _ = self._object1.pos_and_orn()
    #         if obj_pos_new[2] > 0.05:
    #             print("BLOCKPOS!")
            
    #         return True
            
    #     return False
    
    def dofbot_control(self, jointPoses, gripperAngle):
        # 控制机械臂关节和夹爪
        '''
        param jointPoses: 数组，机械臂五个关节角度
        param gripperAngle: 浮点数，机械臂夹爪角度，负值加紧，正值张开
        '''
        self._dofbot.joint_control(jointPoses)
        self._dofbot.gripper_control(gripperAngle)
        p.stepSimulation()
        # time.sleep(self._timeStep)

    def dofbot_setInverseKine(self, pos, orn = None):
        # 逆运动学解算，据目标末端位置和朝向计算对应的关节角度
        '''
        param pos: 机械臂末端位置，xyz
        param orn: 机械臂末端方向，四元数
        '''
        jointPoses = self._dofbot.setInverseKine(pos, orn)
        # 返回机械臂各关节角度
        return jointPoses

    # def dofbot_forwardKine(self,jointStates):
    #     return self._dofbot.forwardKinematic(jointStates)

    def get_dofbot_jointPoses(self):
        # 获取机械臂五个关节位置和夹爪角度
        jointPoses, gripper_angle = self._dofbot.get_jointPoses()

        return jointPoses, gripper_angle

    def get_dofbot_pose(self):
        # 获取机械臂末端当前位置、四元数朝向和欧拉角
        pos, orn, euler = self._dofbot.get_pose()
        return pos, orn, euler

    def get_block_pose(self):
        # 获取物体的当前位置、四元数朝向和欧拉角
        pos, orn, euler = self._object1.pos_and_orn()
        return pos, orn, euler

    def get_target_pose(self):
        # 获取目标位置
        return self.target_pos

    def set_target_pos(self, target_pos):
        # 设置目标位置
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




