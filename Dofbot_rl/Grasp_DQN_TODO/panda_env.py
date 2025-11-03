import pybullet as p
import numpy as np
import time
from panda import Panda

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from typing import Optional
from gymnasium.envs.registration import register

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

register(
    id="PandaEnv-v1",
    entry_point="panda_env:PandaEnv",
    max_episode_steps=300,
)
## place is_grasped判断方式改了
class ObjectPanda:
    def __init__(self, urdfPath, block,num):
        self.id = p.loadURDF(urdfPath)
        self.half_height = 0.025 if block else 0.0745
        self.num = num

        self.block = block
    def reset(self):

        if self.num==1:
            p.resetBasePositionAndOrientation(self.id,
                                         np.array([ 0.615, 0.1,
                                                   self.half_height]),
                                        p.getQuaternionFromEuler([0, 0,0]))
        else:
            p.resetBasePositionAndOrientation(self.id,
                                         np.array([ 0.615, -0.1,
                                                   0.005]),
                                        p.getQuaternionFromEuler([0, 0,0]))

    def pos_and_orn(self):
        pos, orn = p.getBasePositionAndOrientation(self.id)
        # euler = p.getEulerFromQuaternion(quat)
        return pos, orn
    
class PandaEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array', 'human_image'], 'render_fps': 120}
    
    def __init__(self, obs_mode="state_dict", render_mode="rgb_array"):
        self._timeStep = 1/ 120
        self._observation = []
        self.terminated = 0
        self._cam_dist = 2
        self._cam_yaw = 90
        self._cam_pitch = -10
        self.object_size = 0.05
        self.obs_mode = obs_mode
        self.render_mode = render_mode
        
        self._p = p
        if render_mode == "human" or render_mode == "human_image":
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
                # p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)
        # p.connect(p.GUI)
        # p.setRealTimeSimulation(1)
        # p.resetDebugVisualizerCamera(2.0, 90, -40, [0, 0, 0])
        p.resetDebugVisualizerCamera(1.8, 90, -10, [0.615, 0, 0.2])
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -9.8)

        # TODO: observation space
        # if obs_mode == "state": ## 训练模式下使用
        #     self.observation_space =
        # elif obs_mode == "state_dict":  ## 字典形式，方便读取数据
        #     self.observation_space =

        # TODO: action space
        # self.action_space =
    
        p.loadURDF("models/floor.urdf", [0, 0, -0.625], useFixedBase=True)
        p.loadURDF("models/table_collision/table.urdf", [0.5, 0, -0.625],p.getQuaternionFromEuler([0, 0, 0]),
                   useFixedBase=True)
        self._panda = Panda()
        self._object1 = ObjectPanda("models/box_green.urdf", block=True,num=1)
        # self._object2 = ObjectPanda("models/box_purple.urdf", block=True,num=2)
        self.object =  self._object1.id
        self.target_pos = np.array([0.615, 0, 0.1])

    def create_box(self, half_size, color, pos, orn, collision=True, mass=0):
        if collision:
            collision = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[half_size, half_size, half_size]
            )
        else:
            collision = -1
            visual = p.createVisualShape(
                            shapeType=p.GEOM_BOX,
                            halfExtents=[half_size, half_size, half_size],
                            rgbaColor=color
                            )
            box_id = p.createMultiBody(
                            baseMass=mass,
                            baseCollisionShapeIndex=collision,
                            # baseVisualShapeIndex=visual,
                            basePosition=pos,
                            baseOrientation=orn
                            )
        return box_id

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.terminated = 0
        super().reset(seed=seed)
        # p.resetBasePositionAndOrientation(self.goal, self._panda.inital_eepose[0], self._panda.inital_eepose[1])
        self._object1.reset()
        self._panda.reset()
        self.realAction = np.array([0, 0, 0, 0.04])
        p.stepSimulation()
        Observation = self._get_obs()
        info = self._get_info()
        return Observation, info


    def is_grasped(self):
        min_force = 0.5
        contact_finger1 = p.getContactPoints(bodyA=self._panda.pandaUid, bodyB=self.object, linkIndexA=9)
        contact_finger2 = p.getContactPoints(bodyA=self._panda.pandaUid, bodyB=self.object, linkIndexA=10)
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
        
    def _get_obs(self):
        Observation = self._get_obs_dict()

        if self.obs_mode == "state_dict":
            self._observation = Observation
            return self._observation
        elif self.obs_mode == "state":
            # ee_to_object_pos = Observation["ee_to_object_pos"]
            # dist = (ee_to_object_pos[0]**2 + ee_to_object_pos[1]**2 + ee_to_object_pos[2]**2)**0.5
            # print("ee_to_obj_dist:", dist)
            values = list(Observation.values())
            self._observation = np.concatenate([v if isinstance(v, np.ndarray) else np.array([v], dtype=np.int32) for v in values])
            self._observation = self._observation.astype(np.float32)
            return self._observation
        
    def _get_obs_dict(self):
        Observation = self._panda.getObservation()
        
        # TODO： add suitable observations here

        return Observation
    
    ## discrete action: -dx, dx, -dy, dy, -dz, dz, static
    def step(self, action):

        # TODO: Define suitable realAction here
        # self.realAction = np.array([dx, dy, dz, 0.04])


        if self.terminated:
            self.realAction = np.array([0, 0, 0, 0])
        self._panda.applyAction(self.realAction)
        p.stepSimulation()
        if self.render_mode == "human":
            time.sleep(self._timeStep)
        
        terminated = self._termination()  ## task success check
        truncated = False  ## step limitation
        self._observation = self._get_obs()
        reward = self._get_reward()
        info = self._get_info()
        
        return self._observation, reward, terminated, truncated, info
  
    # continous action
    # def step(self, actions):
    #     self.realAction = actions
    #     self._panda.applyAction(self.realAction)
    #     p.stepSimulation()
    #     if self.render_mode == "human":
    #         time.sleep(self._timeStep)
        
    #     terminated = self._termination()  ## task success check
    #     truncated = False  ## step limitation
    #     self._observation = self._get_obs()
    #     reward = self._get_reward()
    #     info = {}
        
    #     return self._observation, reward, terminated, truncated, info
    
    def _termination(self):
        #print (self._kuka.endEffectorPos[2])
        state = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)
        actualEndEffectorPos = state[0]

        #print("self._envStepCounter")
        #print(self._envStepCounter)
        if self.terminated:
            self._observation = self.getExtendedObservation()
            return True
        maxDist = 0.012
        obs = self._get_obs_dict()  
        ee_to_object_pos = obs["ee_to_object_pos"] 
        ee_to_object_dist = (ee_to_object_pos[0]**2 + ee_to_object_pos[1]**2 + ee_to_object_pos[2]**2)**0.5
        if ee_to_object_dist < maxDist:
            self.terminated = 1
        
            print("terminating, closing gripper, attempting grasp")
            #start grasp and terminate
            fingerAngle = 0.3
            for i in range(100):
                graspAction = [0, 0, 0, 0]
                self._panda.applyAction(graspAction)
                p.stepSimulation()
                if self.render_mode == "human":
                    time.sleep(self._timeStep)
                fingerAngle = fingerAngle - (0.3 / 100.)

            for i in range(1000):
                graspAction = [0, 0, 0.005, 0]
                self._panda.applyAction(graspAction)
                p.stepSimulation()
                if self.render_mode == "human":
                    time.sleep(self._timeStep)
                object_pos, object_orn = p.getBasePositionAndOrientation(self.object)
                if (object_pos[2] > 0.23):
                    #print("BLOCKPOS!")
                    #print(blockPos[2])
                    break
                state = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)
                actualEndEffectorPos = state[0]
                if (actualEndEffectorPos[2] > 0.5):
                    break                
            return True
        
        return False
    
    def _get_info(self):
        return dict()

    # TODO: 完善reward function
    def _get_reward(self):
        obs = self._get_obs_dict()
        info = self._get_info()
        
        reward = 0
        return reward
    
    def render(self, ):
        if self.render_mode != "rgb_array" and self.render_mode != "human_image":
            return np.array([])

        base_pos, orn = self._p.getBasePositionAndOrientation(self._panda.pandaUid)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                distance=self._cam_dist,
                                                                yaw=self._cam_yaw,
                                                                pitch=self._cam_pitch,
                                                                roll=0,
                                                                upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                        nearVal=0.1,
                                                        farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                height=RENDER_HEIGHT,
                                                viewMatrix=view_matrix,
                                                projectionMatrix=proj_matrix,
                                                renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        #renderer=self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array
    
    def get_init_eepose(self,):
        return self._panda.inital_eepose