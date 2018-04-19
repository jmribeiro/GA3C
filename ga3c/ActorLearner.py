import time
import numpy as np

from EnvironmentHandler import EnvironmentHandler
from multiprocessing import Process, Queue, Value

import Config


class ActorLearner(Process):

    def __init__(self, ga3c, actor_learner_id):
        super(ActorLearner, self).__init__()

        self.ga3c = ga3c
        self.id = actor_learner_id

        render = True if (Config.RENDER_MODE == 1 and self.id == 0) or Config.RENDER_MODE == 2 else False
        self.env = EnvironmentHandler(render, self.ga3c.render_lock)
        self.num_actions = self.env.action_space
        self.actions = np.arange(self.num_actions)

        self.requested_predictions = Queue(maxsize=1)
        self.running = Value('b', True)

    def run(self):

        time.sleep(1.0)
        np.random.seed(np.int32(time.time() % 1 * self.id * 5555))

        while self.running.value:

            total_score = 0
            total_frames = 0

            for states, actions, rewards, score, terminal in self.run_episode():

                total_score += score
                total_frames += len(rewards) + 1

                self.ga3c.training_queue.put((states, actions, rewards))

            self.ga3c.logger.log(total_score, total_frames)

    def run_episode(self):

        state = self.env.reset()

        terminal = False
        batch = []
        t = 0
        score = 0.0

        while not terminal:

            policy, value = self.predict(state)
            action = self.select_action(policy)
            next_state, reward, terminal = self.env.step(action)

            score += reward

            batch.append(Datapoint(state, action, policy, value, reward, terminal))

            if terminal or t == Config.TIME_MAX:

                states, actions, rewards = self.prepare_batch(batch)
                yield states, actions, rewards, score, terminal
                t = 0
                batch = [batch[-1]]
                score = 0.0

            t += 1
            state = next_state

    def accumulate_rewards(self, batch):

        last_datapoint = batch[-1]

        accumulator = last_datapoint.value if not last_datapoint.terminal else 0

        for t in reversed(range(0, len(batch) - 1)):
            reward = np.clip(batch[t].reward, Config.REWARD_MIN, Config.REWARD_MAX)
            accumulator = + reward + Config.DISCOUNT_FACTOR * accumulator
            batch[t].reward = accumulator

        return batch[:-1]

    def prepare_batch(self, batch):
        batch = self.accumulate_rewards(batch)
        states = np.array([exp.state for exp in batch])
        actions = np.eye(self.num_actions)[np.array([exp.action for exp in batch])].astype(np.float32)
        rewards = np.array([exp.reward for exp in batch])
        return states, actions, rewards

    def predict(self, state):
        self.ga3c.prediction_queue.put((self.id, state))
        policy, value = self.requested_predictions.get()
        return policy, value

    def select_action(self, policy):
        if Config.PLAY_MODE: return np.argmax(policy)
        else: return np.random.choice(self.actions, p=policy)


class Datapoint:
    def __init__(self, state, action, policy, value, reward, terminal):
        self.state = state
        self.action = action
        self.policy = policy
        self.value = value
        self.reward = reward
        self.terminal = terminal
