# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
with open("q_table.plk", "rb") as f:
    Q_table = pickle.load(f)
def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.

    
    return np.argmax(Q_table[(obs[0], obs[1], obs[10], obs[11], obs[12], obs[13])])

    # You can submit this random agent to evaluate the performance of a purely random strategy.

