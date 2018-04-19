from multiprocessing import Queue

import time
from threading import RLock

import Config

from ActorCriticNetwork import ActorCriticNetwork
from ActorLearner import ActorLearner
from Logger import Logger
from Predictor import Predictor
from Trainer import Trainer


class GA3C:
    def __init__(self):

        if Config.PLAY_MODE:
            Config.ACTOR_LEARNERS = 1
            Config.PREDICTORS = 1
            Config.TRAINERS = 1
            Config.RENDER_MODE = 1
            Config.LOAD = True
            Config.TRAIN_MODELS = False
        self.render_lock = RLock()

        self.model = ActorCriticNetwork()

        self.training_queue = Queue(maxsize=Config.TRAINING_QUEUE_SIZE)
        self.prediction_queue = Queue(maxsize=Config.PREDICTION_QUEUE_SIZE)

        self.actor_learners = [ActorLearner(self, i) for i in range(Config.ACTOR_LEARNERS)]
        self.predictors = [Predictor(self, i) for i in range(Config.ACTOR_LEARNERS)]
        self.trainers = [Trainer(self, i) for i in range(Config.TRAINERS)]

        self.logger = Logger()
        if Config.LOAD:
            self.logger.episodes.value = self.model.load()
            print("Loaded model!")
        else:
            print("Starting new model!")

    def run(self):

        for trainer in self.trainers: trainer.start()
        for predictor in self.predictors: predictor.start()
        for actor_learner in self.actor_learners: actor_learner.start()

        self.logger.start()

        while self.logger.episodes.value < Config.EPISODES:
            time.sleep(Config.SAVE_INTERVAL)
            self.model.save(self.logger.episodes.value)
            print("Saved model!")

        for actor_learner in self.actor_learners:
            actor_learner.running.value = False
            actor_learner.join()

        for predictor in self.predictors:
            predictor.running = False
            predictor.join()

        for trainer in self.trainers:
            trainer.running = False
            trainer.join()

        self.logger.running = False
        self.logger.join()


if __name__ == '__main__':
    agent = GA3C()
    agent.run()
