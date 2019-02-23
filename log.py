import os



class Log():

    def wf(self,name,chr):
        path = "logs"
        if not os.path.exists(path):
            os.mkdir(path)
        filepath = './logs/'+ name + '.txt'
        F = open(filepath, 'a')
        F.write(str(chr))
        F.write('\n')
        F.close()

    def wf_check(self,path,its):
        filepath = path + '/checkpoint'
        F = open(filepath, 'w')
        F.write('model_checkpoint_path: "pong-dqn-' + str(its) + '"' + '\n')
        F.write('all_model_checkpoint_paths: "pong-dqn-' + str(its) + '"')


if __name__ == "__main__":
    a = str([1,1,1])
    Log().wf('1',a)
