import pybullet as p
import numpy as np
import pybullet_data
import os


class Panda:
    def __init__(self, urdfRootPath=pybullet_data.getDataPath(), initial_pos=[0, 0, 0]):
        self.urdfRootPath = urdfRootPath
        self.pandaEndEffectorIndex = 11
        # # upper limits for null space
        self.ll = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        # upper limits for null space
        self.ul = [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]
        # joint ranges for null space
        self.jr = [5.7946, 3.5256, 5.7946, 3.002,  5.7946, 3.77,   5.7946]
        # rest poses for null space
        self.rp = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]

        self.maxForce = 200.
        self.fingerForce = 10
        self.initial_pos = initial_pos
        orn = [0, 0, 0, 1]
        self.pandaUid = p.loadURDF(os.path.join(self.urdfRootPath, "franka_panda/panda.urdf"), self.initial_pos, orn, useFixedBase=True)
        self.numJoints = 9
        self.reset_jointPositions = [0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,]
        self.motorIndices = []  # [0, 1, 2, 3, 4, 5, 6]
        index = 0
        for j in range(p.getNumJoints(self.pandaUid)):
            p.changeDynamics(self.pandaUid, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.pandaUid, j)
            jointIndex = info[0]
            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC):
                p.resetJointState(self.pandaUid, j, self.reset_jointPositions[index]) 
                self.motorIndices.append(jointIndex)
                index += 1
            if (jointType == p.JOINT_REVOLUTE):
                p.resetJointState(self.pandaUid, j, self.reset_jointPositions[index]) 
                self.motorIndices.append(jointIndex)
                index += 1
        state = p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)
        self.inital_eepose = [state[0], state[1]]

    def reset(self):        
        index = 0
        for j in range(self.numJoints):
            p.changeDynamics(self.pandaUid, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.pandaUid, j)
        
            jointName = info[1]
            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC):
                p.resetJointState(self.pandaUid, j, self.reset_jointPositions[index]) 
                index=index+1
            if (jointType == p.JOINT_REVOLUTE):
                p.resetJointState(self.pandaUid, j, self.reset_jointPositions[index]) 
                index=index+1
        state = p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)


    def joint_control(self,jointPoses):          
        for i in range(self.numJoints - 2):
            p.setJointMotorControl2(bodyUniqueId=self.pandaUid, jointIndex=self.motorIndices[i], controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses[i], targetVelocity=0, force=200,
                                maxVelocity=1.0, positionGain=0.3, velocityGain=1)
        self.gripper_control(jointPoses[-2:])

    def setInverseKine(self, pos, orn=None):
        if orn == None:
            jointPoses = p.calculateInverseKinematics(self.pandaUid, self.pandaEndEffectorIndex, pos,
                                                      self.ll, self.ul, self.jr, self.rp)
        else:
            orn_new = [orn[0], orn[1], orn[3], orn[2]]
            jointPoses = p.calculateInverseKinematics(self.pandaUid, self.pandaEndEffectorIndex, pos, orn_new,
                                                      self.ll, self.ul, self.jr, self.rp)
        return jointPoses[:7]


    def get_jointPoses(self):
        jointPoses= []
        for i in range(self.numJoints):
            state = p.getJointState(self.pandaUid, self.motorIndices[i])
            jointPoses.append(state[0])
        return np.array(jointPoses)

    def get_qvel(self):
        qvel= []
        for i in range(self.numJoints):
            state = p.getJointState(self.pandaUid, self.motorIndices[i])
            qvel.append(state[1])
        return np.array(qvel)
    
    def get_gripper_pose(self):
        state = p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)
        pos = state[0]
        orn = state[1]
        return pos,orn


    def getObservation(self):
        observation = dict()
        observation["qpos"] = self.get_jointPoses()
        
        pos, orn = self.get_gripper_pose()
        # euler = p.getEulerFromQuaternion(orn)
        observation["eepose"] = np.array(pos+orn)
        
        
        return observation

    def gripper_control(self,gripperAngle):
        p.setJointMotorControl2(self.pandaUid,
                                9,
                                p.POSITION_CONTROL,
                                targetPosition=gripperAngle[0],
                                force=self.fingerForce)
        p.setJointMotorControl2(self.pandaUid,
                                10,
                                p.POSITION_CONTROL,
                                targetPosition=gripperAngle[1],
                                force=self.fingerForce)
        
    ## 输入ee delta pos(3维)， orn不变
    def applyAction(self, actions):
        state = p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)
        curr_eepos = np.array(state[0])
        desire_eepos = curr_eepos + np.array(actions[:3])
        desire_qpos = list(self.setInverseKine(desire_eepos, self.inital_eepose[1]))
        desire_qpos = desire_qpos + [actions[3], actions[3]]
        self.joint_control(desire_qpos)
        obs = self.getObservation()
        return obs
    
        