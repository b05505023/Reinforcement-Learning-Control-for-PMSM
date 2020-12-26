# motor_control

W.L. Peng, Y.W. Lan, S.G Chen, F.J Lin, R.I. Chang and J.M. Ho, Reinforcement Learning Control for Six-Phase Permanent Magnet Synchronous Motor Position Servo Drive, 2020 IEEE 3nd International Conference on Knowledge Innovation and Invention (ICKII), Kaohsiung, Taiwan

## To train the model from begining:
1. change the file, train.py, in line 44 "begin = True"
2. specify your own hyperparameters in train.py
3. cd into the motor control file
4. $python train.py

## To test the model from begining:
1. change the file, test.py, in line 10 "env_name" to be same as your training file env_name.
2. change the file, test.py, in line 32 "random_seed" to be same as your training file random seed.
3. cd into the motor control file
4. $python test.py

## To resume the model from training:
1. change the file, train.py, in line 10 "env_name" to be same as your training file env_name.
2. change the file, train.py, in line 31 "random_seed" to be same as your training file random seed.
3. change the file, train.py, in line 44 "begin = False"
4. cd into the motor control file
5. $python train.py
