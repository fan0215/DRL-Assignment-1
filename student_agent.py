import numpy as np
import gym
import random
import pickle

# 創建 Taxi 環境
env = gym.make("Taxi-v3")

# 參數設定
alpha = 0.1    # 學習率
gamma = 0.9    # 折扣因子
epsilon = 0.1  # 探索率 (ε-greedy)
episodes = 5000  # 訓練次數

# 初始化 Q-table (狀態數量 x 動作數量)
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# 訓練代理
for episode in range(episodes):
    state, _ = env.reset()  # 初始化環境
    done = False

    while not done:
        # ε-greedy 探索策略
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # 探索 (隨機選擇動作)
        else:
            action = np.argmax(q_table[state])  # 利用 (選擇最高 Q 值的動作)

        # 執行動作，獲得回饋
        next_state, reward, done, truncated, _ = env.step(action)

        # 增加對牆壁的懲罰
        if reward == -5:  # 撞牆的懲罰
            reward = -100  # 設定較大的負獎勵

        # 更新 Q-table
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

        # 移動到下一個狀態
        state = next_state

# 儲存訓練好的 Q-table
with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)


# 測試代理
def get_action(obs):
    # 加載訓練好的 Q-table
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
    
    # 若狀態不在 Q-table，則隨機選擇動作
    if obs not in q_table:
        return random.choice([0, 1, 2, 3, 4, 5])
    
    # 選擇最優動作
    return np.argmax(q_table[obs])
