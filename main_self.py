#!/usr/bin/env python
import random
import threading
import time

import cv2
import matplotlib
import numpy as np
import tensorflow as tf
import sys
sys.path.append("Wrapped_Game_Code")

matplotlib.use("Agg")
import agent as Agent
import best_avgNet as Net
import Config
import argparse
from log import *
import matplotlib.pyplot as plt
import pong_fun_self as game


#Shared global parameters
T = 0
durations  = []
catches1 = []
catches2 = []


OBSERVE = 5
EXPLORE = 400000

parser = argparse.ArgumentParser()
parser.add_argument('--threads', help ='threads',default='12')
parser.add_argument('--DEVICES',help = 'CUDA_VISIBLE_DEVICES',default='0')
parser.add_argument('--pong',help = 'pong_version', default='4' )
args = parser.parse_args()

GAME = 'pong'

os.environ["CUDA_VISIBLE_DEVICES"] = args.DEVICES

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


cf = Config.Config()

path_model = 'save/self_v' + args.pong + '/pong-dqn'

path_catch1 = 'save/self_v' + args.pong + '/catch1.jpeg'
path_catch2 = 'save/self_v' + args.pong + '/catch2.jpeg'
path_duration = 'save/self_v' + args.pong + '/duration.jpeg'


def actorLearner(num, sess,saver,net1,net2, lock):

    global T,durations,catches1,catches2,misses1,misses2
    t1 = 0
    score = 0
    score1 = 0
    # Open up a game state to communicate with emulator
    game_state = game.GameState()

    agent1 = Agent.Agent(sess)
    agent2 = Agent.Agent(sess)

    lock.acquire()
    x_t = game_state.get_init_frame()
    lock.release()

    x_t = cv2.cvtColor(cv2.resize(x_t,(80,80)), cv2.COLOR_BGR2GRAY)
    s_t = np.stack((x_t, x_t,x_t,x_t), axis = 2)

    time.sleep(3 * num)


    print("THREAD ", num, "STARTING...", "EXPLORATION POLICY => INITIAL_EPSILON:", "agent1", agent1.INITIAL_EPSILON, "agent2",
    agent2.INITIAL_EPSILON, ", FINAL_EPSILON:", "agent1", agent1.FINAL_EPSILON
    , "agent2", agent2.FINAL_EPSILON)

    while T < cf.TMAX:#and score < WISHED_SCORE:
        # choose action from avg or best for a


        action1 = agent1.choose_action(net1, s_t)
        action2 = agent2.choose_action(net2, s_t)
        lock.acquire()
        x_t1_col, r_t, terminal, catch1, catch2, miss1, miss2 = game_state.frame_step(action1, action2)
        lock.release()
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80,80)), cv2.COLOR_BGR2GRAY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        aux_s = np.delete(s_t, 0, axis = 2)
        s_t1 = np.append(aux_s, x_t1, axis = 2)
        #
        #
        transition = [s_t,action1,s_t1,r_t,terminal]

        agent1.store_memory(net1,transition)

        transition = [s_t, action2, s_t1, -r_t, terminal]

        agent2.store_memory(net2,transition)
        #
        s_t = s_t1

        score += r_t
        score1 -= r_t
        #
        #
        #update the old values

        T += 1


        if T % cf.updateT == 0:
            sess.run(net1.update_op)                # target network in net1
            sess.run(net2.update_op)

        if agent1.t > 0 and agent1.t % cf.asyncT == 0 or terminal:
            if agent1.s_batch:
                net1.train_best_op.run(session=sess, feed_dict={net1.y: agent1.y_batch,
                                                               net1.a: agent1.a_batch,
                                                               net1.s: agent1.s_batch})
                agent1.y_batch = []
                agent1.a_batch = []
                agent1.s_batch = []

        if agent2.t > 0 and agent2.t % cf.asyncT == 0 or terminal:
            if agent2.s_batch:
                net2.train_best_op.run(session = sess, feed_dict = {net2.y:agent2.y_batch,
                                                                net2.a:agent2.a_batch,
                                                                net2.s: agent2.s_batch})
                agent2.y_batch = []
                agent2.a_batch = []
                agent2.s_batch = []


        if agent1.t % cf.saveT== 0:
            saver.save(sess,path_model,global_step=agent1.t)


        if agent1.t <= OBSERVE:
            state = "oberver"

        elif agent1.t > OBSERVE and agent1.t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        if terminal:
            duration = agent1.t - t1
            print "THREAD:", num, "/ TIME", T, "/ TIMESTEP", "agent1", agent1.t, "agent2", agent2.t, "/ STATE", state, "/ EPSILON", "agent1", \
                    agent1.epsilon, "agent2", agent2.epsilon, "scores", "agent1", score, "agent2", score1, "duration", duration, "catches", "agent1", catch1, "agent2",catch2
            score = 0
            score1 = 0
            game_state.catch1 = 0
            game_state.catch2 = 0
            game_state.miss1 = 0
            game_state.miss2 = 0
            t1 = agent1.t
            catches1.append(catch1)
            catches2.append(catch2)
            lock.acquire()
            plt.clf()
            plt.plot(catches1)
            plt.savefig(path_catch1)
            plt.clf()
            plt.plot(catches2)
            plt.savefig(path_catch2)
            plt.clf()
            lock.release()




if __name__ == "__main__":

    net1 = Net.best_avgNet("net1")
    net2 = Net.best_avgNet("net2")


    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())


    sess.run(net1.update_op)
    sess.run(net2.update_op)

    path = 'save/self_v' + args.pong
    if not os.path.exists(path):
        os.mkdir(path)

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print "Sucessfully loaded:", checkpoint.model_checkpoint_path

    lock = threading.Lock()
    threads = list()

    for i in range(int(args.threads)):
        t =threading.Thread(target = actorLearner, args = (i, sess,saver, net1, net2,lock))
        threads.append(t)

    for x in threads:
        x.start()

    for x in threads:
        x.join()

    print "ALL DONE!"
































