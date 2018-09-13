import gym
import tensorflow as tf
import rl_env as rl
from DQNAnn import *

NB_ENTRY = 4
NB_HIDDENS = [16]
NB_OUT = 2 # Left and right

EPSILON_START = 1.0
EPSILON_FINAL = 0.02
EPSILON_DECAY = 1e4

REPLAY_SIZE = int(1e5)
REPLAY_START_SIZE = 10

LEARNING_RATE = 1e-2

if __name__ == "__main__":
    cartpole = gym.make('CartPole-v0')
    env_wrapper = rl.EnvironmentWrappers.CumulateEnvWrapper(cartpole)

    ann = DQNAnn(NB_ENTRY, NB_HIDDENS, NB_OUT, "main_network")
    target = DQNAnn(NB_ENTRY, NB_HIDDENS, NB_OUT, "target_network")
    action_selector = rl.ActionSelectors.GreedyArgmaxActionSelector(EPSILON_START, EPSILON_FINAL, EPSILON_DECAY)
    agent = rl.Agents.DQNAgent(ann, target, action_selector, LEARNING_RATE)

    experience_source = rl.ExperienceSources.ExperienceSource(env_wrapper, agent)
    replay_buffer = rl.ReplayBuffer(REPLAY_SIZE, REPLAY_START_SIZE)
    learner = rl.Learner(agent, experience_source, replay_buffer, "logs")

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        while True:
            learner(nb_replay=10)
