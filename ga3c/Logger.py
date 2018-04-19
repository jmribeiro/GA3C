import sys
import time

from queue import Queue as queueQueue
from multiprocessing import Process, Queue, Value


class Logger(Process):

    def __init__(self):

        super(Logger, self).__init__()

        self.logger = Queue(maxsize=100)

        self.episodes = Value('i', 0)
        self.total_training_iterations = Value('i', 0)
        self.total_frames = 0
        self.total_score = 0

        self.stats = queueQueue(maxsize=1000)

        self.running = True

    def run(self):

        while self.running:

            score, episode_frames = self.logger.get()

            self.total_frames += episode_frames
            self.episodes.value += 1

            self.total_score += score

            if self.stats.full():
                old_score, old_frames = self.stats.get()
                self.total_frames -= old_frames
                self.total_score -= old_score

            self.stats.put((score, episode_frames))

            print(f'Episode #{self.episodes.value}, '
                  f'Score: {score}] '
                  f'(Avg. Score: {self.total_score / self.stats.qsize()})')

            sys.stdout.flush()

    def log(self, score, frames):
        self.logger.put((score, frames))