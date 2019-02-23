import threading

def actorLearner(i):
    i = i + 1
    print(i)

if __name__ == "__main__":
    # Start n concurrent actor threads
    lock = threading.Lock()
    threads = list()
    j = 5
    for i in range(10):
        t = threading.Thread(target=actorLearner, args=(j))
        threads.append(t)

    # Start all threads
    for x in threads:
        x.start()

    # Wait for all of them to finish
    for x in threads:
        x.join()

    print "ALL DONE!!"
