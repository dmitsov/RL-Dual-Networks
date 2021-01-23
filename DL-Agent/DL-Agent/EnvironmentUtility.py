import gym
from gym import wrappers
from AtariWrappers import wrap_deepmind
import os.path as osp

def getEnvironment(seed, envName, videoLogFrequency, exportDir):
    env = gym.make(envName)
    env.seed(seed)

    env.action_space.np_random.seed(seed)

    def video_schedule(episode_id):
        return episode_id % videoLogFrequency == 0

    env = wrappers.Monitor(env, osp.join(exportDir, "video"), force=True,
                           video_callable=(video_schedule if videoLogFrequency > 0 else False))

    return wrap_deepmind(env)

