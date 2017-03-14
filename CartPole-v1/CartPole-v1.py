"""
Solves the cartpole-v1 enviroment on OpenAI gym using policy search

Same algortihm as for cartpole-v0

A neural network is used to store the policy

At the end of each episode the target value for each taken action is
updated with the total normalized reward (up to a learning rate)

Then a standard supervised learning backprop on the entire batch is
executed

"""

import numpy as np
import numpy.matlib 

import gym
from gym import wrappers

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

#initialize neural network to store policy
ActorNet = Sequential()
ActorNet.add(Dense(200,init='he_normal',input_dim=4,activation='relu'))
ActorNet.add(Dense(200,init='he_normal',activation='relu'))
ActorNet.add(Dense(2,init='he_normal',activation='sigmoid'))
ActorNet.compile(loss='mse',optimizer='RMSprop',metrics=['mae'])

NumEpisodes = 300

#load environment
env = gym.make('CartPole-v1')
env = gym.wrappers.Monitor(env, 'monitor')

TotalReward = 0
BufferSize = 0
eps = 1

#start learning
for episode in range(NumEpisodes):

    #initial state
    observation = env.reset() #observe initial state

    States = []
    ActionValues = []
    Actions = []

    t = 0
    loss = 0
    EpisodeReward = 0

    #decrease epsilon after each episode
    eps -= 0.01
    if eps<0:
        eps = 0

    while True:
        
        #show graphical environment
        #env.render()

        #evaluate NN to find action probabilities for current state

        #normalize inputs
        observation[0] /= 2.5
        observation[1] /= 2.5
        observation[2] /= 0.2
        observation[3] /= 2.5

        ActionValue = ActorNet.predict(observation.reshape(1,4),verbose=0).reshape(2,)

        #select best action eps-greedy with decay
        greedy = np.random.random()
        if greedy < eps:
            Action = np.random.randint(2)
        else:
            Action = np.argmax(ActionValue)
        
        #execute action
        observation_new, reward, done, info = env.step(Action)

        #normalize reward, maximum reward per episode is 500
        reward /= 500.0

        EpisodeReward += reward
        
        #save current movement in memory to assign rewards at end of episode
        States.append(observation)
        ActionValues.append(ActionValue)
        Actions.append(Action)

        #update state
        observation = observation_new

        #next time step
        t += 1

        #end episode
        if done:
            break


    #update finished episode memory with new reward
    #only update action value for actions that were taken, leave others unchanged
    alpha = 0.1
    for i in range(t):
        ActionValues[i][Actions[i]] = ActionValues[i][Actions[i]] * (1-alpha) + EpisodeReward * alpha

    #update weights of NN based on last completed episode
    batch_in = np.empty([t,4]) #input state
    batch_tar = np.empty([t,2]) #target action values
    for i in range(t):
        batch_in[i] = States[i]
        batch_tar[i] = ActionValues[i]
    loss += ActorNet.train_on_batch(batch_in, batch_tar)[0]

    print('Episode {0}, reward = {1}'.format(episode,EpisodeReward))

    TotalReward += EpisodeReward

print('Total reward = {0}'.format(TotalReward))
#ActorNet.save('CPv1_model.h5')

env.close()
