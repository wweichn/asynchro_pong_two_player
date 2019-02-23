#!/usr/bin/env python
import threading

import cv2
import matplotlib
import numpy as np
import tensorflow as tf
import sys
sys.path.append("../Wrapped_Game_Code")
import pong_fun_self as game

matplotlib.use("Agg")
sys.path.append("../")
import agent as Agent
import best_avgNet as Net
import os
import argparse
#Shared global parameters

parser = argparse.ArgumentParser()
parser.add_argument('--threads', help ='threads',default='1')
parser.add_argument('--DEVICES',help = 'CUDA_VISIBLE_DEVICES',default='0')
parser.add_argument('--pong',help = 'pong_version', default='0' )
args = parser.parse_args()

GAME = 'pong'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ACTIONS = 3

def playGame(sess,agent1,net1,agent2,net2):
    # open up a game state to communicate with emulator
    game_state = game.GameState()
    agent = Agent.Agent(sess)

    x_t  = game_state.get_init_frame()
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    aux_s = s_t

    # get the first state by doing nothing and preprocess the image to 80x80x4

    score = 0
    terminal = False
    while not terminal:

        # choose an action
        action1 = agent1.choose_action_play(net1, s_t)
        action2 = agent2.choose_action_play(net2,s_t)

        # run the selected action and observe next state and reward
        x_t1_col, r_t, terminal, catch1, catch2, miss1, miss2= game_state.frame_step(action1,action2)
        score += r_t
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        aux_s = np.delete(s_t, 0, axis = 2)
        s_t1 = np.append(aux_s, x_t1, axis = 2)

        # update state and score
        s_t = s_t1


    # Print final score
    print "FINAL SCORE1", score," catch1 ", catch1, " catch2 ",catch2, "miss1",miss1, "miss2",miss2
    #print "FINAL SCORE2", score1

if __name__ == "__main__":

    net1 = Net.best_avgNet("net1")
    net2 = Net.best_avgNet("net2")
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())


    sess.run(net1.update_op)
    sess.run(net2.update_op)

    saver = tf.train.Saver()

    path = '../save/self_v' + args.pong

    checkpoint = tf.train.get_checkpoint_state(path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print "Sucessfully loaded:", checkpoint.model_checkpoint_path

    agent1 = Agent.Agent(sess)
    agent2 = Agent.Agent(sess)

    for i in range(25):
        playGame(sess, agent1,net1,agent2,net2)

    print "ALL DONE!"
































