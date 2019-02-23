import ConfigParser
import os
cf = ConfigParser.ConfigParser()
path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"config.ini")
cf.read(path)



class Config(object):

    HIT_REWARD = cf.getint('REWARD','HIT_REWARD')
    LOSE_REWARD = cf.getint('REWARD','LOSE_REWARD')
    SCORE_REWARD = cf.getint('REWARD','SCORE_REWARD')

    TMAX = cf.getint('Train','TMAX')
    updateT = cf.getint('Train','updateT')
    asyncT = cf.getint('Train','asyncT')
    saveT = cf.getint('Train','saveT')
    batchSize = cf.getint('Train','batchSize')



    actionDim = cf.getint('Model','actionDim')
    ETA = cf.getfloat('Model','ETA')
    avg_lr = cf.getfloat('Model','avg_lr')
    best_lr = cf.getfloat('Model','best_lr')
    discount = cf.getfloat('Model','discount')

    maxSize = cf.getint('Memory','maxSize')

    start = cf.getint('Test','start')
    end = cf.getint('Test','end')
    test_iteration = cf.getint('Test','testIteration')



