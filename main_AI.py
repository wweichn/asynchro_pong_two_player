#!/usr/bin/env python
import random
import threading
import time

import cv2
import matplotlib
import numpy as np
import tensorflow as tf

from Wrapped_Game_Code import pong_fun as game

matplotlib.use("Agg")
import agent as Agent
import best_avgNet as Net
import Config
import argparse
import os
from log import *
import matplotlib.pyplot as plt



#Shared global parameters
T = 0
scores = []
losses = []

OBSERVE = 5
EXPLORE = 400000
WISHED_SCORE = 5

parser = argparse.ArgumentParser()
parser.add_argument('--threads', help ='threads',default='12')
parser.add_argument('--DEVICES',help = 'CUDA_VISIBLE_DEVICES',default='0')
args = parser.parse_args()

GAME = 'pong'

os.environ["CUDA_VISIBLE_DEVICES"] = args.DEVICES

cf = Config.Config()


def actorLearner(num, sess,saver,net1,net2, lock):

    global T,scores

    score = 0
    # Open up a game state to communicate with emulator
    game_state = game.GameState()

    agent1 = Agent.Agent(sess)

    lock.acquire()
    x_t = game_state.get_init_frame()
    lock.release()

    x_t = cv2.cvtColor(cv2.resize(x_t,(80,80)), cv2.COLOR_BGR2GRAY)
    s_t = np.stack((x_t, x_t,x_t,x_t), axis = 2)

    time.sleep(3 * num)


    # choose action from avg or best for a

    print("THREAD ", num, "STARTING...", "EXPLORATION POLICY => INITIAL_EPSILON:", "agent1", agent1.INITIAL_EPSILON,
    ", FINAL_EPSILON:", "agent1", agent1.FINAL_EPSILON)

    while T < cf.TMAX and score < WISHED_SCORE:
        action = agent1.choose_action(net1, s_t)
        lock.acquire()
        x_t1_col, r_t, terminal = game_state.frame_step(action)
        lock.release()
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80,80)), cv2.COLOR_BGR2GRAY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        aux_s = np.delete(s_t, 0, axis = 2)
        s_t1 = np.append(aux_s, x_t1, axis = 2)
        transition = [s_t,action,s_t1,r_t,terminal]

        agent1.store_memory(net1,transition)
        #
        s_t = s_t1

        score += r_t

        T += 1

        if T % cf.updateT == 0:
            sess.run(net1.update_op)                # target network in net1

        if agent1.t % cf.asyncT == 0 or terminal:
            if agent1.s_batch:
                loss,_ = sess.run([net1.loss, net1.train_best_op], feed_dict = {
                                                                net1.y: agent1.y_batch,
                                                                net1.a: agent1.a_batch,
                                                                net1.s:agent1.s_batch
                                                                })

                losses.append(loss)
                lock.acquire()
                plt.clf()
                plt.plot(losses)
                plt.savefig("save/AI/loss.jpeg")
                lock.release()
                agent1.y_batch = []
                agent1.a_batch = []
                agent1.s_batch = []

        if agent1.t > 0 and agent1.t % cf.saveT== 0:
            saver.save(sess,'save/AI/pong-dqn',global_step=agent1.t)

        if agent1.t <= OBSERVE:
            state = "oberver"

        elif agent1.t > OBSERVE and agent1.t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        if terminal:
            print "THREAD:", num, "/ TIME", T, "/ TIMESTEP", "agent1", agent1.t, "/ STATE", state, "/ EPSILON", "agent1", \
                    agent1.epsilon, "scores", "agent1", score
            scores.append(score)
            lock.acquire()
            plt.clf()
            plt.plot(scores)
            plt.savefig("save/AI/score.jpeg")
            lock.release()
            score = 0


if __name__ == "__main__":

    net1 = Net.best_avgNet("net1")
    net2 = Net.best_avgNet("net2")


    sess = tf.Session()
    sess.run(tf.initialize_all_variables())


    sess.run(net1.update_op)
    #sess.run(net2.update_op())

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("save/AI")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print "Sucessfully loaded:", checkpoint.model_checkpoint_path

    lock = threading.Lock()
    threads = list()
    path = 'save/AI'
    if not os.path.exists(path):
        os.mkdir(path)

    for i in range(int(args.threads)):
        t =threading.Thread(target = actorLearner, args = (i, sess,saver, net1, net2,lock))
        threads.append(t)

    for x in threads:
        x.start()

    for x in threads:
        x.join()

    print "ALL DONE!"
































