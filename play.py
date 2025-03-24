import pickle
import numpy as np

# 隨機生成 policy table
policy_table = {}

# 設定隨機種子，確保每次執行結果一致
np.random.seed(42)

# 假設 state 由 6 個整數和 2 個布林值組成
num_states = 50  # 生成 50 個不同的 state
for _ in range(num_states):
    state = tuple(np.random.randint(-1, 2, size=6)) + (bool(np.random.randint(0, 2)), bool(np.random.randint(0, 2)))
    q_values = np.random.rand(6)  # 6 個動作的 Q-value
    policy_table[state] = q_values

# 將 policy_table 存成 pickle 格式
with open("policy_table_pretrained.pkl", "wb") as f:
    pickle.dump(policy_table, f)

print("Generated policy_table_pretrained.pkl successfully!")
