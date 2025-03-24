import numpy as np
import random
import gym

# Global Q-table (simulating pre-training)
Q = {}
trained = False

def compute_state_key(obs):
    """Convert observation to a custom state key."""
    taxi_row = obs['taxi_row']
    taxi_col = obs['taxi_col']
    passenger_location = obs['passenger_location']  # 0-3 or 4
    destination = obs['destination']  # 0-3
    locations = obs['locations']  # [(row, col), ...] for R,G,Y,B
    grid_size = obs['grid_size']
    obstacles = obs['obstacles']  # [(row, col), ...]

    # Passenger status
    passenger_status = 0 if passenger_location != 4 else 1

    # Direction to passenger
    if passenger_status == 0:
        p_row, p_col = locations[passenger_location]
        dx_p = p_col - taxi_col
        dy_p = p_row - taxi_row
        dir_passenger = (int(dx_p > 0) - int(dx_p < 0), int(dy_p > 0) - int(dy_p < 0))
    else:
        dir_passenger = (0, 0)

    # Direction to destination
    d_row, d_col = locations[destination]
    dx_d = d_col - taxi_col
    dy_d = d_row - taxi_row
    dir_destination = (int(dx_d > 0) - int(dx_d < 0), int(dy_d > 0) - int(dy_d < 0))

    # Local obstacles (1 if obstacle or wall, 0 otherwise)
    ob_n = 1 if taxi_row == 0 or (taxi_row - 1, taxi_col) in obstacles else 0
    ob_s = 1 if taxi_row == grid_size - 1 or (taxi_row + 1, taxi_col) in obstacles else 0
    ob_e = 1 if taxi_col == grid_size - 1 or (taxi_row, taxi_col + 1) in obstacles else 0
    ob_w = 1 if taxi_col == 0 or (taxi_row, taxi_col - 1) in obstacles else 0
    obstacles_tuple = (ob_n, ob_s, ob_e, ob_w)

    return (passenger_status, dir_passenger, dir_destination, obstacles_tuple)

def heuristic_action(state_key):
    """Heuristic for unseen states."""
    passenger_status, dir_passenger, dir_destination, obstacles = state_key
    ob_n, ob_s, ob_e, ob_w = obstacles

    if passenger_status == 0:
        if dir_passenger == (0, 0):
            return 4  # Pickup
        dx, dy = dir_passenger
    else:
        if dir_destination == (0, 0):
            return 5  # Dropoff
        dx, dy = dir_destination

    # Move toward target, avoiding obstacles
    if dx > 0 and not ob_e:
        return 2  # East
    elif dx < 0 and not ob_w:
        return 3  # West
    elif dy > 0 and not ob_s:
        return 0  # South
    elif dy < 0 and not ob_n:
        return 1  # North
    else:
        # Random move among unblocked directions
        moves = []
        if not ob_n: moves.append(1)
        if not ob_s: moves.append(0)
        if not ob_e: moves.append(2)
        if not ob_w: moves.append(3)
        return random.choice(moves) if moves else random.choice([0, 1, 2, 3])

def train_agent():
    """Train the Q-table (simulated as a one-time initialization)."""
    global Q, trained
    if trained:
        return

    # Initialize environment (assuming custom Taxi-v3 is available)
    env = gym.make('Taxi-v3')  # Placeholder; assumes modified env
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.999
    min_epsilon = 0.01
    episodes = 5000  # Reduced for practicality; adjust as needed

    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            state_key = compute_state_key(obs)
            if state_key not in Q:
                Q[state_key] = np.zeros(6)

            # Epsilon-greedy action
            if random.random() < epsilon:
                action = random.choice([0, 1, 2, 3, 4, 5])
            else:
                action = np.argmax(Q[state_key])

            next_obs, reward, done, _ = env.step(action)
            next_state_key = compute_state_key(next_obs)
            if next_state_key not in Q:
                Q[next_state_key] = np.zeros(6)

            # Q-update
            Q[state_key][action] += alpha * (reward + gamma * np.max(Q[next_state_key]) - Q[state_key][action])
            obs = next_obs

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    trained = True
    # Note: In practice, save Q with pickle.dump(Q, open('q_table.pkl', 'wb'))

def get_action(obs):
    """Return the agent's action based on the observation."""
    # Train once (simulating pre-training)
    train_agent()

    state_key = compute_state_key(obs)
    if state_key in Q:
        return int(np.argmax(Q[state_key]))  # Ensure integer action
    else:
        return heuristic_action(state_key)

