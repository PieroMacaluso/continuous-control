import pickle
from collections import deque

from unityagents import UnityEnvironment
import numpy as np

import matplotlib.pyplot as plt

from models.ddpg.agent import Agent
from test import test
from utils.config import read_hp

if __name__ == '__main__':
    hp = read_hp("configs/reacher_ddpg.yaml")
    env = UnityEnvironment(file_name=hp['unity_env_path'])
    min_solved = None
    target_reward = 30.0
    # Get the default brain
    brain_name = env.brain_names[0]

    scores = []
    test_scores = []
    test_scores_i = []
    avg_scores = []
    scores_window = deque(maxlen=100)

    agent = Agent(hp)
    for i_episode in range(1, hp['n_episodes'] + 1):
        # Reset the environment and the score
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        score = np.zeros(hp['num_agents'])
        while True:
            # damping_noise = abs((target_reward - np.mean(scores_window))) / target_reward + 0.05
            # damping_noise = 1.0 if len(scores_window) == 0 else damping_noise
            actions = agent.act(state, add_noise=True)
            env_info = env.step(actions)[brain_name]
            next_states, rewards, dones = env_info.vector_observations, env_info.rewards, env_info.local_done
            agent.step(state, actions, rewards, next_states, dones)
            state = next_states
            score += rewards
            if np.any(dones):
                break
        scores_window.append(score)
        scores.append(score)
        avg_scores.append(np.mean(scores_window))
        print(
            '\rEpisode {}\tLast Score: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, np.mean(score),
                                                                             np.mean(scores_window)),
            end="")
        if i_episode % 50 == 0:
            test_scores.append(test(env, agent, i_episode, hp))
            test_scores_i.append(i_episode)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(range(len(scores)), scores, label="Score")
            plt.savefig(agent.dir + f'plot_scores_{i_episode}.png', dpi=300)
            plt.show()
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(range(len(avg_scores)), avg_scores, label="Avg Score")
            ax1.plot(test_scores_i, test_scores, label="Test Score")
            plt.savefig(agent.dir + f'plot_avg_{i_episode}.png', dpi=300)
            plt.legend()
            plt.show()
            agent.save_weights(i_episode)
            with open(agent.dir + "scores.txt", "wb") as fp:
                pickle.dump(scores, fp)
            with open(agent.dir + "test_scores.txt", "wb") as fp:
                pickle.dump(test_scores, fp)
            with open(agent.dir + "avg_scores.txt", "wb") as fp:
                pickle.dump(avg_scores, fp)
        if min_solved is None or np.mean(scores_window) >= min_solved:
            min_solved = np.mean(scores_window)
            print('\nNew best in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                               np.mean(scores_window)))
            agent.save_weights(i_episode)
    env.close()
