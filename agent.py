import random
import numpy as np
import Config
from log import *
FINAL_EPSILONS = [0.01, 0.01, 0.05]
INITIAL_EPSILONS = [0.4, 0.3,0.3]
EPSILONS = 3
OBSERVE = 5
EXPLORE = 400000

cf = Config.Config()


class Agent():
    def __init__(self,sess):
        epsilon_index = random.randrange(EPSILONS)
        self.INITIAL_EPSILON = INITIAL_EPSILONS[epsilon_index]
        self.FINAL_EPSILON = FINAL_EPSILONS[epsilon_index]
        self.epsilon = self.INITIAL_EPSILON
        self.actions = cf.actionDim
        self.a_batch = []
        self.y_batch = []
        self.s_batch = []

        self.exploit = 0
        self.session = sess

        self.t = 0

    def choose_action_play(self,net,s,flag):
        a = np.zeros([self.actions])
        if flag == 'b':
            best_q = net.best_q.eval(session=self.session, feed_dict = {net.s:[s]})[0]
            action_index = np.argmax(best_q)
        else:
            pi = net.avg_a.eval(session=self.session,feed_dict = {net.s:[s]})[0]
            action_index = np.random.choice(len(pi),p = pi)


        a[action_index] = 1

        return a

    def choose_action(self,net,s):
        a_t = np.zeros([self.actions])

        if random.random() <= self.epsilon or self.t <= OBSERVE:
            action_index = random.randrange(self.actions)
        else:
            best_q = net.best_q.eval(session = self.session, feed_dict = {net.s :[s]})[0]
            action_index = np.argmax(best_q)
        self.t += 1
        self.decrease_epsilon()

        a_t[action_index] = 1


        return a_t

    def best_action(self, net , s):

        a_t = np.zeros([self.actions])
        action_index = np.argmax(net.best_q.eval(session = self.session, feed_dict = {net.s:[s]})[0])
        a_t[action_index] = 1

        return a_t

    def decrease_epsilon(self):
        if self.epsilon > self.FINAL_EPSILON and self.t > OBSERVE:
            self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / EXPLORE


    def store_memory(self,net,transition):
        s = transition[0]
        a = transition[1]
        s_t = transition[2]
        r_t = transition[3]
        terminal = transition[4]
        q_t = net.best_q_t.eval(session = self.session,feed_dict = {net.s:[s_t]})
        if terminal:
            self.y_batch.append(r_t)
        else:
            self.y_batch.append(r_t + cf.discount * np.max(q_t))

        self.a_batch.append(a)

        self.s_batch.append(s)


















