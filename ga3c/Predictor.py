from threading import Thread

import numpy as np

import Config


class Predictor(Thread):

    def __init__(self, ga3c, pid):

        super(Predictor, self).__init__()
        self.setDaemon(True)

        self.ga3c = ga3c
        self.id = pid
        self.running = True

    def run(self):

        ids = np.zeros(Config.PREDICTION_MIN_BATCH, dtype=np.uint16)
        states = np.zeros((Config.PREDICTION_MIN_BATCH, Config.IMAGE_HEIGHT,
                           Config.IMAGE_WIDTH, Config.STACKED_FRAMES), dtype=np.float32)

        while self.running:

            ids[0], states[0] = self.ga3c.prediction_queue.get()

            total_requests = 1
            while total_requests < Config.PREDICTION_MIN_BATCH and not self.ga3c.prediction_queue.empty():
                ids[total_requests], states[total_requests] = self.ga3c.prediction_queue.get()
                total_requests += 1

            batch = states[:total_requests]
            policy, value = self.ga3c.model.predict(batch)

            for i in range(total_requests):
                self.ga3c.actor_learners[ids[i]].requested_predictions.put((policy[i], value[i]))
