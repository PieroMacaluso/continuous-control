from datetime import datetime
from typing import NamedTuple, Tuple, Callable

import yaml

# from models.q_net import QNet
# from models.q_cnn import QCNN
# from models.dueling_q_net import DuelingQNet
# from memories.replay_buffer import ReplayBuffer
import torch
import torch.optim as optim

from utils.noises import OUNoise


class Config(NamedTuple):
    state_size: Tuple[int]
    action_size: Tuple[int]
    seed: int
    device: Callable
    actor_net: Callable
    critic_net: Callable
    actor_opt: Callable
    critic_opt: Callable
    memory: Callable
    noise: Callable
    checkpoint_dir: str
    buffer_size: int = int(1e5)
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 1e-3
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    update_every: int = 4
    double_qn: bool = True
    visual: bool = False
    n_episodes: int = 1000
    eps_start: float = 1.0
    eps_decay: float = 0.99
    eps_min: float = 0.01


def generate_configuration_d4qn(action_size, state_size):
    config = Config(
        seed=1,
        n_episodes=1000,
        eps_start=1.0,
        eps_decay=0.99,
        eps_min=0.01,
        device=lambda: torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        action_size=action_size,
        state_size=state_size,
        actor_net=QNet,
        critic_net=QNet,
        actor_opt=optim.Adam,
        critic_opt=optim.Adam,
        buffer_size=int(1e5),
        batch_size=64,
        gamma=0.99,
        tau=1e-3,
        actor_lr=5e-4,
        critic_lr=5e-4,
        update_every=4,
        double_qn=True,
        visual=False,
        memory=lambda: ReplayBuffer(config.action_size, config.buffer_size, config.batch_size, config.seed,
                                    config.device()),
        checkpoint_dir=f"./checkpoints/DoubleDQNet_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    )
    return config


def generate_configuration_ddpg(action_size, state_size):
    config = Config(
        seed=1,
        n_episodes=1000,
        eps_start=1.0,
        eps_decay=0.99,
        eps_min=0.01,
        device=lambda: torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        action_size=action_size,
        state_size=state_size,
        actor_net=QNet,
        critic_net=QNet,
        actor_opt=optim.Adam,
        critic_opt=optim.Adam,
        buffer_size=int(1e5),
        batch_size=64,
        gamma=0.99,
        tau=1e-3,
        actor_lr=5e-4,
        critic_lr=5e-4,
        update_every=4,
        double_qn=True,
        visual=False,
        noise=lambda: OUNoise(action_size, config.seed),
        memory=lambda: ReplayBuffer(config.action_size, config.buffer_size, config.batch_size, config.seed,
                                    config.device()),
        checkpoint_dir=f"./checkpoints/DDPG_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    )
    return config


def read_config(path):
    """
    Return python dict from .yml file.

    Args:
        path (str): path to the .yml config.

    Returns (dict): configuration object.
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg
