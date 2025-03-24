# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
with open("policy_table_pretrained.plk", "rb") as f:
    Q_table = pickle.load(f)
def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.


    state_key = tuple(obs)  # 視你的 obs 結構而定，這裡假設 obs 是可轉換為 tuple 的
    
    # 如果 Q_table 中有該狀態，選擇具有最高 Q 值的動作
    if state_key in Q_table:
        action = int(np.argmax(Q_table[state_key]))
    else:
        # 如果找不到狀態，回退到隨機策略
        action = random.choice([0, 1, 2, 3, 4, 5])
    
    return action

    # You can submit this random agent to evaluate the performance of a purely random strategy.

