import gym
import numpy as np

env = gym.make("FrozenLake-v0")

EPISODES = 5000
ALPHA = .1
GAMMA = .9  # discounted feature reward

Q = np.zeros((env.observation_space.n, env.action_space.n))

def get_best_action(state):
    return np.argmax(Q[state, :])


def get_random_action():
    return env.action_space.sample()


def get_max_policy(state):
    return np.max(Q[state, :])


for i in range(EPISODES):
    s = env.reset()
    terminated = False
    while not terminated:
        a = get_random_action()
        s_prime, r, terminated, _ = env.step(a)
        if not terminated:
            s = s_prime

    R = 100 if r == 1 else -100
    Q[s, a] += ALPHA * (R + GAMMA * get_max_policy(s_prime) - Q[s, a])

print(Q)

won = 0
for i in range(EPISODES):
    s = env.reset()
    terminated = False
    while not terminated:
        a = get_best_action(s)
        s_prime, r, terminated, _ = env.step(a)
        if not terminated:
            s = s_prime
    if r == 1:
        won += 1

print("Won", won, 'times')
