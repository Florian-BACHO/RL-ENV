class BasicLearner:
    def __init__(self, agent, experience_source, replay_buffer=None):
        self.agent = agent
        self.exp_src = experience_source
        self.replay_buf = replay_buffer
        self.all_tries_exp = []

    def do_train_episod(self, nb_tries, nb_replay):
        self.all_tries_exp = []

        for _ in range(nb_tries):
            exps = self.exp_src.do_episod()
            self.all_tries_exp.append(exps)
            self.replay_buf.add_experiences(exps)

        if self.replay_buf:
            replay = self.replay_buf.get_sample(nb_replay)
        else:
            replay = None
        self.agent.train(self.all_tries_exp, replay)
