# 《机器人学》大作业 2

徐恺阳 523030910085

## 引言

大作业 2 包含以下内容：

1. 了解节点、话题、消息等 ROS 基础知识；
2. 结合 ROS 与 Dofbot 机械臂进行通信，控制机械臂完成实机抓取放置任务。

## 任务一：

### 任务描述

已知：
- 物体初始位置和放置位置
- 机械臂参数
- 通过示教获取初始位置夹取和最后放置位置的机械臂关节参数

在实机上完成“机械臂抓取初始位置处物块，并放置到目标位置”的任务。

### 任务实现

具体步骤如下：

1. 基于状态机控制思想，设计如下的 4 个状态:
    ```python
    np.asarray([90., 90., 90., 90., 90.]),    # INITIAL_STATE
    np.asarray([133., 48., 52., 2., 90.]),    # PRE_GRASP_STATE
    np.asarray([90., 71., 47., 10., 90.]),    # MOVE_STATE
    np.asarray([40., 57., 42., 7., 90.]),     # SET_STATE
    ```
    其中，每个状态下的关节参数由示教程序获取。

2. 取出相邻状态的路径点并做**线性插值**得到关节的路径。
    ```python
    path = linear_interpolation(points[i], points[i + 1], n=30)
    for p in path:
        # 只控制关节，夹爪保持不变
        env.step(joint=p, gripper=None)
    ``` 

3. 关节控制结束后，根据需求控制夹爪。
   - `INITIAL_STATE` 结束，进入 `PRE_GRASP_STATE`：夹爪线性插值到 120 进行夹取
        ```python
        if i == 0:
            path = linear_interpolation(env.get_state()[-1], 120., n=20)
            for g in path:
                env.step(joint=None, gripper=g)
        ```
    - `MOVE_STATE` 结束，进入 `SET_STATE`：夹爪线性插值到 94 进行放置
        ```python
        elif i == 2:
            path = linear_interpolation(env.get_state()[-1], 94., n=20)
            for g in path:
                env.step(joint=None, gripper=g)
        ```

4. 抓取放置任务执行结束，回到 `INITIAL_STATE`
    ```python
    env.reset()
    ```

### 结果展示


