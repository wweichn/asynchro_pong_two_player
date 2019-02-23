import Config
import random
import numpy as np

cf = Config.Config()
class ReservoirBuffer(object):

    def __init__(self):
        self.buffer = []
        self.count = 0


    def add(self, experience):
        if self.count == cf.maxSize:
            index = np.random.choice(cf.maxSize + 1)
            if index < cf.maxSize:
                self.buffer[index] = experience
            else:
                print("drop the newest experience")
        else:
            self.buffer.append(experience)
            self.count += 1

    def sample_batch(self):
        sample_ids = np.random.randint(len(self.buffer),size = cf.batchSize)
        boards, actions = list(zip(*[self.buffer[i] for i in sample_ids]))
        return boards, actions




