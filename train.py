from collections import deque

from unityagents import UnityEnvironment
import numpy as np

import matplotlib.pyplot as plt

from models.ddpg.agent import Agent
from test import test
from utils.config import Config, generate_configuration_d4qn, generate_configuration_ddpg, read_config

if __name__ == '__main__':
    config = read_config("configs/reacher_ddpg.yaml")
    env = UnityEnvironment(file_name="./Reacher_Linux/Reacher.x86_64")
    min_solved = None
    # Get the default brain
    brain_name = env.brain_names[0]

    scores = []
    test_scores = []
    test_scores_i = []
    avg_scores = []
    scores_window = deque(maxlen=100)
    agent = Agent(config)
    for i_episode in range(1, config['n_episodes'] + 1):
        # Reset the environment and the score
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        score = np.zeros(config['num_agents'])
        while True:
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
        if i_episode % 20 == 0:
            test_scores.append(test(env, agent, i_episode, config))
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
            plt.show()
            agent.save_weights(i_episode)
        if min_solved is None or np.mean(scores_window) >= min_solved:
            min_solved = np.mean(scores_window)
            print('\nNew best in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                               np.mean(scores_window)))
            agent.save_weights(i_episode)
    df = pd.DataFrame(list(zip(scores, avg_scores)),
                      columns=['Scores', 'Last100'])
    env.close()
