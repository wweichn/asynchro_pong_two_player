from Wrapped_Game_Code import pong_fun_self_AI_test as game
import cv2
import numpy as np
import random
import agent as Agent
import best_avgNet as Net
import tensorflow as tf
import threading
import Config
import argparse

cf = Config.Config()
ETA = 0
parser = argparse.ArgumentParser()
parser.add_argument('--threads', help ='threads',default='1')
parser.add_argument('--DEVICES',help = 'CUDA_VISIBLE_DEVICES',default='0')
parser.add_argument('--pong',help = 'pong_version', default='0' )
args = parser.parse_args()



def playGame(sess,agent1, net1,net2, lock):
    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    x_t = game_state.get_init_frame()
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)
    score1 = 0
    score2 = 0
    for i in range(cf.test_iteration):
        for j in range(2):
            j = 1
            score1 = 0
            score2 = 0
            terminal = False
            if j == 0:
                while not terminal:

                    action = agent1.choose_action_play(net1, s_t)

                    lock.acquire()
                    x_t1_col, r_t, terminal = game_state.frame_step(action,1)
                    lock.release()
                    x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
                    x_t1 = np.reshape(x_t1, (80, 80, 1))
                    aux_s = np.delete(s_t, 0, axis=2)
                    s_t = np.append(aux_s, x_t1, axis=2)
                    score1 += r_t
                if terminal:
                    print("the ", i, "th left play against AI, score", score1 )
            else:
                while not terminal:

                    action = agent1.choose_action_play(net2, s_t)
                    lock.acquire()
                    x_t1_col, r_t, terminal = game_state.frame_step(action,-1)
                    lock.release()
                    x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
                    x_t1 = np.reshape(x_t1, (80, 80, 1))
                    aux_s = np.delete(s_t, 0, axis=2)
                    s_t = np.append(aux_s, x_t1, axis=2)
                    score2 -= r_t
                if terminal:
                    print('the ', i, 'th right play against AI, score', score2)


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
    lock = threading.Lock()


    for i in range(25):
        playGame(sess, agent1,net1,net2,lock)


    print "ALL DONE!"



