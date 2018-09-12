import gym
import rl_env as rl

if __name__ == "__main__":
    gym_env = gym.make('CartPole-v0')
    env_wrapper = rl.EnvironmentWrapper(gym_env)
    agent = rl.AbstractAgent()
    experience_source = rl.Experience.DiscountedExperienceSource(env_wrapper, agent)
    replay_buffer = rl.ReplayBuffer(100)
    learner = rl.BasicLearner(agent, experience_source, replay_buffer)

    learner.do_train_episod(3, 3)
