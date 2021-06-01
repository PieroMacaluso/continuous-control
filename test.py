from collections import deque
from time import sleep

import numpy as np
import torch
from unityagents import UnityEnvironment

from models.ddpg.agent import Agent
from utils.config import generate_configuration_ddpg, read_hp


def test(env, agent, n_ep_train, config, n_episodes=10, sleep_t=0.0):
    # Get the default brain
    brain_name = env.brain_names[0]

    scores = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):
        # Reset the environment and the score
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations
        score = np.zeros(config['num_agents'])
        while True:
            actions = agent.act(state, add_noise=False)
            env_info = env.step(actions)[brain_name]
            next_states, rewards, dones = env_info.vector_observations, env_info.rewards, env_info.local_done
            state = next_states
            score += rewards
            if np.any(dones):
                break
        scores_window.append(score)
        scores.append(score)
        print('\rTest Episode {}\tLast Score: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, np.mean(score),
                                                                                    np.mean(scores_window)),
              end="")
    print('\rTest after {} episode mean {:.2f}'.format(n_ep_train, np.mean(scores_window)))
    return np.mean(scores_window)


if __name__ == '__main__':
    env = UnityEnvironment(file_name="./Reacher_Linux/Reacher.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state_size = len(env_info.vector_observations[0])
    config = read_hp("configs/reacher_ddpg.yaml")

    agent = Agent(config)
    agent.load_weights("./checkpoint.pth")
    print(test(env, agent, 0, n_episodes=100, sleep_t=0))
