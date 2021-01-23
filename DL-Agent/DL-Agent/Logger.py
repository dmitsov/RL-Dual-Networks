import sys
import time
import gym
import os
import numpy as np
import matplotlib.pyplot as plt
from PolicyUtility import calculateEpsilon

import json



def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s" % classname)

class Logger(object):
    def __init__(self, env, epsilonSchedule):
        self.env = env
        self.epsilonSchedule = epsilonSchedule

        self.mean_episode_reward = -float("nan")
        self.best_mean_episode_reward = -float("inf")
        self.start_time = time.time()

        self.iteration_list = []
        self.mean_episode_reward_list = []
        self.best_mean_episode_reward_list = []
        self.save_threshold = 10000

    def stats(self, iteration, savePath):
        episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()

        # print last 10 mean rewards
        #print(episode_rewards[-10:])

        eps = calculateEpsilon(iteration, **self.epsilonSchedule)
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])

        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = max(
                self.best_mean_episode_reward, mean_episode_reward
            )

            self.iteration_list.append(iteration)
            self.mean_episode_reward_list.append(mean_episode_reward)
            self.best_mean_episode_reward_list.append(self.best_mean_episode_reward)

        print()
        print("-" * 30)
        print("Timestep %d." % (iteration,))
        print("mean reward (over 100 episodes) %f." % mean_episode_reward)
        print("best mean reward (ever) %f." % self.best_mean_episode_reward)
        print("episode # %d." % len(episode_rewards))
        print("exploration eps %f." % eps)
        # print("learning_rate %f" % self.optimizer_spec.lr_schedule.value(self.t))
        if self.start_time is not None:
            print("running time %f" % ((time.time() - self.start_time) / 60.0))
        print("-" * 30)
        sys.stdout.flush()

        if iteration > self.save_threshold or iteration == 5000000:
            self._saveRewards(savePath)
            self.save_threshold += 50000

        self.start_time = time.time()

    def plot(self, env_name, plotFile=None):

        iters = self.iteration_list
        mean_reward = self.mean_episode_reward_list
        best_mean_reward = self.best_mean_episode_reward_list

        plt.plot(iters, mean_reward, label="Average per Epoch Reward")
        plt.plot(iters, best_mean_reward, label="Best Mean Reward")
        plt.legend()
        plt.xlabel("Time Steps")
        plt.ylabel("Reward")
        plt.title(env_name)

        if plotFile is not None:
            plt.savefig(plotFile)

        plt.show()

    def _saveRewards(self, fileName):
        rewardsSaveDict = dict(iteration_list=self.iteration_list, mean_rewards=self.mean_episode_reward_list, best_mean_rewards=self.best_mean_episode_reward_list)

        with open(fileName, 'w+') as saveFile:
            json.dump(rewardsSaveDict, saveFile)



