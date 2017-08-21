import gym
import numpy as np

env = gym.make("FrozenLake-v0")


def get_best_action(Q, state):
    return np.argmax(Q[state, :])


def get_random_action():
    return env.action_space.sample()


def get_max_policy(Q, state):
    return np.max(Q[state, :])


def run(alpha, gamma, episodes):
    # training
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for i in range(episodes):
        s = env.reset()
        terminated = False
        while not terminated:
            a = get_random_action()
            s_prime, r, terminated, _ = env.step(a)
            if not terminated:
                s = s_prime

        R = 100 if r == 1 else -100
        Q[s, a] += alpha * (R + gamma * get_max_policy(Q, s_prime) - Q[s, a])

    # evaluating
    won = 0
    for i in range(episodes):
        s = env.reset()
        terminated = False
        while not terminated:
            a = get_best_action(Q, s)
            s_prime, r, terminated, _ = env.step(a)
            if not terminated:
                s = s_prime
        if r == 1:
            won += 1

    return won


def grid_search(alpha, gamma, episodes):
    results = []
    for a in alpha:
        for g in gamma:
            results.append([a, g, run(a, g, episodes)])
    return results


results = grid_search(np.linspace(0, 1, 11), np.linspace(0, 1, 11), 5000)
results = np.asarray(results)
best = results[np.argmax(results[:, 2])]
print(f'Best alpha: {best[0]}, best gamma: {best[1]}, won: {int(best[2])} times.')

