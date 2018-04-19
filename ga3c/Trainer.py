from threading import Thread
import numpy as np

import Config


class Trainer(Thread):

    def __init__(self, ga3c, tid):

        super(Trainer, self).__init__()
        self.setDaemon(True)

        self.ga3c = ga3c
        self.id = tid
        self.running = True

    def run(self):

        while self.running:

            batch_size = 0

            while batch_size <= Config.TRAINING_MIN_BATCH:

                s, a, r = self.ga3c.training_queue.get()

                if batch_size == 0:
                    states = s
                    actions = a
                    rewards = r
                else:
                    states = np.concatenate((states, s))
                    actions = np.concatenate((actions, a))
                    rewards = np.concatenate((rewards, r))

                batch_size += s.shape[0]
            
            if not Config.PLAY_MODE:
                self.ga3c.model.train(states, actions, rewards)
                self.ga3c.logger.total_training_iterations.value += 1
