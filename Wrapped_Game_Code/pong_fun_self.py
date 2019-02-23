#!/usr/bin/env python
#Modified from http://www.pygame.org/project-Very+simple+Pong+game-816-.html

import numpy
import pygame
import os
from pygame.locals import *
from sys import exit
import random
import pygame.surfarray as surfarray
import sys
sys.path.append("../")
import math
import Config

position = 5, 325
os.environ['SDL_VIDEO_WINDOW_POS'] = str(position[0]) + "," + str(position[1])
pygame.init()
screen = pygame.display.set_mode((640,480),0,32)
#screen = pygame.display.set_mode((640,480),pygame.NOFRAME)
#Creating 2 bars, a ball and background.
back = pygame.Surface((640,480))
background = back.convert()
background.fill((0,0,0))
bar = pygame.Surface((10,50))
bar1 = bar.convert()
bar1.fill((255,255,255))
bar2 = bar.convert()
bar2.fill((255,255,255))
circ_sur = pygame.Surface((15,15))
circ = pygame.draw.circle(circ_sur,(255,255,255),(15/2,15/2),15/2)
circle = circ_sur.convert()
circle.set_colorkey((0,0,0))
font = pygame.font.SysFont("calibri",40)

cf = Config.Config()
ai_speed = 7

HIT_REWARD = cf.HIT_REWARD
LOSE_REWARD = cf.LOSE_REWARD
SCORE_REWARD = cf.SCORE_REWARD


cf = Config.Config()
class GameState:
    def __init__(self):
        self.bar1_x, self.bar2_x = 10. , 620.
        self.bar1_y, self.bar2_y = 215., 215.
        self.initial_x, self.initial_y = 307.5, 232.5
        self.circle_x, self.circle_y = 307.5, 232.5
        self.bar1_move, self.bar2_move = 0. , 0.
        self.bar1_score, self.bar2_score = 0,0
        self.speed_x, self.speed_y = 7, 7
        self.count = 0
        self.catch1 = 0
        self.catch2 = 0
        self.miss1 = 0
        self.miss2 = 0

    def get_init_frame(self):
        screen.blit(background, (0, 0))
        frame = pygame.draw.rect(screen, (255, 255, 255), Rect((5, 5), (630, 470)), 2)
        middle_line = pygame.draw.aaline(screen, (255, 255, 255), (330, 5), (330, 475))
        screen.blit(bar1, (self.bar1_x, self.bar1_y))
        screen.blit(bar2, (self.bar2_x, self.bar2_y))
        screen.blit(circle, (self.circle_x, self.circle_y))
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()

        return image_data


    def frame_step(self,input_vect1,input_vect2):

        pygame.event.pump()
        reward = 0

        if sum(input_vect1) != 1:
            raise ValueError('Multiple input actions!')

                  #left player
        if input_vect1[1] == 1:#Key down
            self.bar1_move = -ai_speed
        elif input_vect1[2] == 1:#Key up
            self.bar1_move = ai_speed
        else: # don't move
            self.bar1_move = 0

        self.bar1_y += self.bar1_move

        if input_vect2[1] == 1:#Key down
            self.bar2_move = -ai_speed
        elif input_vect2[2] == 1:#Key up
            self.bar2_move = ai_speed
        else: # don't move
            self.bar2_move = 0

        self.bar2_y += self.bar2_move


        self.score1 = font.render(str(self.bar1_score), True,(255,255,255))
        self.score2 = font.render(str(self.bar2_score), True,(255,255,255))

        # bounds of movement
        if self.bar1_y >= 420.:
            self.bar1_y = 420.
        elif self.bar1_y <= 10.:
            self.bar1_y = 10.
        if self.bar2_y >= 420.:
            self.bar2_y = 420.
        elif self.bar2_y <= 10.:
            self.bar2_y = 10.

        #since i don't know anything about collision, ball hitting bars goes like this.
        if self.circle_x <= self.bar1_x + 10.:
            if self.circle_y >= self.bar1_y - 7.5 and self.circle_y <= self.bar1_y + 42.5:
                self.circle_x = 20.
                self.speed_x = -self.speed_x
                reward = HIT_REWARD
                self.catch1 += 1               # player 1 receive the ball

        if self.circle_x >= self.bar2_x - 15.:
            if self.circle_y >= self.bar2_y - 7.5 and self.circle_y <= self.bar2_y + 42.5:
                self.circle_x = 605.
                self.speed_x = -self.speed_x
                reward = -HIT_REWARD
                self.catch2 += 1             # player 2 receive the ball

        # scoring
        if self.circle_x < 5.:
            self.bar2_score += 1
            reward = LOSE_REWARD
            self.miss1 += 1

            index = random.randrange(2)
            self.circle_x, self.circle_y = 320, 232.5
            if index == 0:
                self.speed_x = - self.speed_x


        elif self.circle_x > 620.:
            self.bar1_score += 1
            reward = SCORE_REWARD
            self.miss2 += 1

            index = random.randrange(2)
            self.circle_x, self.circle_y = 320, 232.5
            if index == 0:
                self.speed_x = - self.speed_x

        # collisions on sides
        if self.circle_y <= 10.:
            self.speed_y = -self.speed_y
            self.circle_y = 10.
        elif self.circle_y >= 457.5:
            self.speed_y = -self.speed_y
            self.circle_y = 457.5

        self.circle_x += self.speed_x
        self.circle_y += self.speed_y

        screen.blit(background, (0, 0))
        frame = pygame.draw.rect(screen, (255, 255, 255), Rect((5, 5), (630, 470)), 2)
        middle_line = pygame.draw.aaline(screen, (255, 255, 255), (330, 5), (330, 475))
        screen.blit(bar1, (self.bar1_x, self.bar1_y))
        screen.blit(bar2, (self.bar2_x, self.bar2_y))
        screen.blit(circle, (self.circle_x, self.circle_y))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())


        # I move the score here so the pixels of the numbers do not appear in the image_data
        screen.blit(self.score1,(250.,210.))
        screen.blit(self.score2,(380.,210.))

        pygame.display.update()

        terminal = False
        if max(self.bar1_score, self.bar2_score) >= 20:

            self.count = 0
            self.bar1_score = 0
            self.bar2_score = 0
            terminal = True
            self.bar1_y, self.bar2_y = 215., 215.


        return image_data, reward, terminal, self.catch1, self.catch2,self.miss1, self.miss2
