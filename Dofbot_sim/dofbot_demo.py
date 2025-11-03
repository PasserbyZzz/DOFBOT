from dofbot import DofbotEnv
import numpy as np
import time

if __name__ == '__main__':
    env = DofbotEnv()
    env.reset()
    Reward = False

    num_fk_data = 0
    dataset = []
    sleep_time = 0.01
    target_pos = np.array([0.2, 0.15, 0.15])
    while True:
        env.step_with_sliders()
        num_fk_data += 1
        time.sleep(sleep_time)
        print(num_fk_data)
        # if num_fk_data == 100:
        #     env.set_target_pos(target_pos)


        if num_fk_data >= 10000:
            break