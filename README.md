# asynchro_pong_two_player

A code implementation of asynchronous two-player ping-pong.

It's a reimplementation of "Multiagent Cooperation and Competition with Deep Reinforcement Learning" with Tensorflow. And it
uses asynchnronous architecture to speed up training.

The code is based on [Zeta36's work](https://github.com/Zeta36/Asynchronous-Methods-for-Deep-Reinforcement-Learning)(asynchronous ping-pong)+ my tweaks

# Result

run```python main_self.py -threads $threads -pong $storage_name```. This is for two-player ping-pong game(players are both AI, studying from 0)

After 5-6 hours' training, playing durations increased and catches of both player increased.

catches of player1(x-axis is iterations)
![catches of player1](/save/pic/catch1.jpeg)

catches of player2(x-axis is iterations)
![catches of player2](/save/pic/catch2.jpeg)

durations of game(x-axis is iterations, y-axis is lasting time)
![durations of game](/save/pic/duration.jpeg)

run ```python main_AI.py``` . This is for AI-computer ping-pong game.

After serveral hours' training, playing durations increased and catches of players increased.

catches of player(x-axis is iterations)
![catches of player](/save/pic/pong-dqn-adapted-score.jpeg)

# Test

run ```python Test_code/compete_AI.py```. This is for testing two-player training effect by using trained models competing with computer.

run ```python Test_code/compete_Self.py```. This is for testing two-player training effect by using trained models competing with each other.

run ```python Test_code/compete_Self_AI.py```. This is for testing two-player training effect by using trained models competing with computer. And it considers
the position of player.(left or right)

You need to edit the model path.

