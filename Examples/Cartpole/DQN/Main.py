import tensorflow as tf
import sys
sys.path.insert(0, '../../../')
import rl_env as rl
from CartpoleWrapper import *
from DQNAnn import *

NB_ENTRY = 4
NB_HIDDENS = [128, 32]
NB_OUT = 2 # Left and right

EPSILON_START = 1.0
EPSILON_FINAL = 0.02
EPSILON_DECAY = 1e4

REPLAY_SIZE = 10000
REPLAY_START_SIZE = 100
REPLAY_BATCH_SIZE = 32

LEARNING_RATE = 1e-4
DISCOUNT_RATE = 0.99

TARGET_UPDATE_RATE = 500 # Update target network each x training step

TRY_PER_EPOCH = 10

LOG_DIR = "logs"

if __name__ == "__main__":
    env = CartpoleWrapper()

    ann = DQNAnn(NB_ENTRY, NB_HIDDENS, NB_OUT, "main_network")
    target = DQNAnn(NB_ENTRY, NB_HIDDENS, NB_OUT, "target_network")
    argmax_selector = rl.ActionSelectors.ArgmaxActionSelector()
    greedy_selector = rl.ActionSelectors.GreedyActionSelector(argmax_selector, EPSILON_START, \
                                                              EPSILON_FINAL, EPSILON_DECAY)
    agent = rl.Agents.DQNAgent(ann, target, greedy_selector, LEARNING_RATE, DISCOUNT_RATE, \
                               TARGET_UPDATE_RATE)

    experience_source = rl.ExperienceSources.ExperienceSource(env, agent)
    replay_buffer = rl.ReplayBuffer(REPLAY_SIZE, REPLAY_START_SIZE)

    summary_writer = tf.summary.FileWriter(LOG_DIR)
    reward_tracker = rl.Trackers.ScalarTracker("Reward", False, summary_writer)
    loss_tracker = rl.Trackers.ScalarTracker("Loss", False, summary_writer)
    epsilon_tracker = rl.Trackers.ScalarTracker("Epsilon", False, summary_writer)

    learner = rl.Learner(agent, experience_source, replay_buffer, reward_tracker, \
                         loss_tracker, epsilon_tracker)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        while True:
            learner(nb_replay=REPLAY_BATCH_SIZE)

    summary_writer.close()
