import numpy as np
import matplotlib.pyplot as plt

import time
import torch
import datetime

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class MetricLogger:
    def __init__(self, configs=None, save_dir=None):
        if not save_dir:
            # Save directory
            dt = datetime.datetime.now().strftime("%y-%m-%dT%H-%M-%S")
            save_dir = Path("checkpoints") / dt
            save_dir.mkdir(parents=True)
        self.save_dir = save_dir

        # Experiment Configs
        if configs is not None:
            self.config_dir = save_dir / "configs"
            with open(self.config_dir, "w") as f:
                for hparam, value in configs.items():
                    f.write(f"{hparam}: {value}\n")

        # Metric logs
        self.log_dir = save_dir / "logs"
        self.header = (f"{'Episode':>8}{'Step':>8}{'Reward':>15}"
                       f"{'Length':>15}"
                    #    f"{'ActorLoss':>15}"
                    #    f"{'CriticLoss':>15}"
                    #    f"{'Value':>15}"
                       f"{'TimeDelta':>15}{'Time':>20}\n")
        # print(self.header)
        with open(self.log_dir, "w") as f:
            f.write(self.header)

        # Plots
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        # self.ep_actor_losses_plot = save_dir / "actor_loss_plot.jpg"
        # self.ep_critic_losses_plot = save_dir / "critic_loss_plot.jpg"
        # self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # Tensorboard writer
        self.writer = SummaryWriter(save_dir)
        self.curr_step = 0

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        # self.ep_actor_losses = []
        # self.ep_critic_losses = []
        # self.ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Time
        self.record_time = time.time()

        # Col names
        self.col_names = [
            "ep_rewards", 
            "ep_lengths", 
            # "ep_actor_losses", 
            # "ep_critic_losses", 
            # "ep_avg_qs", 
            ]

    def log_step(self, reward, q, ep_len):
        # if isinstance(q, torch.Tensor):
        #     q = q.clone().detach().cpu().numpy()
        self.curr_ep_reward += reward
        self.curr_ep_length += ep_len
        # self.curr_ep_q += q

    def log_episode(self, actor_loss, critic_loss):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        # self.ep_actor_losses.append(actor_loss)
        # self.ep_critic_losses.append(critic_loss)
        # self.ep_avg_qs.append(self.curr_ep_q/self.curr_ep_length)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        # self.curr_ep_q = 0.0

    def record(self, episode, step):
        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(
            self.record_time - last_record_time, 3)

        log = (f"{episode:8d}{step:8d}"
               f"{self.ep_rewards[-1]: 15.3f}{self.ep_lengths[-1]: 15.3f}"
            #    f"{self.ep_actor_losses[-1]:15.3f}"
            #    f"{self.ep_critic_losses[-1]:15.3f}"
            #    f"{self.ep_avg_qs[-1]: 15.3f}"
               f"{time_since_last_record:15.3f}"
               f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n")

        self.curr_step += step

        # print on terminal
        # print(log)
        # write on logs
        with open(self.log_dir, "a") as f:
            f.write(log)
        # write on tensorboard
        for metric in self.col_names:
            self.writer.add_scalar(
                f'Metric/{metric}', getattr(self, metric)[-1], self.curr_step)

    def plot(self):
        for metric in self.col_names:
            plt.plot(getattr(self, f"{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()
