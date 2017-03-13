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

num_episodes = 500

memory_index = 0
memory_full = 1
memory_max = 200 #number of examples stored in memory (one episode)
memory = np.empty([memory_max,7]) # 4 states - 2 actions values - 1 selected action

#load environment
env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, '/tmp/CartPolev0-1')

TotalReward = 0

#start learning
for episode in range(num_episodes):

    #initial state
    observation = env.reset() #observe initial state

    loss = 0
    EpisodeReward = 0
    episode_mem = np.empty([memory_max,7])

    for t in range(300):

        #show graphical environment
        env.render()

        #evaluate NN to find Q-values for current state

        #normalize input to NN
        observation[0] /= 2.5
        observation[1] /= 2.5
        observation[2] /= 0.2
        observation[3] /= 2.5
        
        ActionValue = ActorNet.predict(observation.reshape(1,4),verbose=0).reshape(2,)

        #act = np.argmax(ActionValue)

        #select best action (eps-greedy)
        eps = 1 - (episode / num_episodes)
        if eps<0.1:
            eps = 0.1
        greedy = np.random.random()
        if greedy < eps:
            act = np.random.randint(2)
        else:
            act = np.argmax(ActionValue)
        
        #execute action
        observation_new, reward, done, info = env.step(act)

        #normalize reward
        reward /= 200.0

        EpisodeReward += reward
        
        #save movement in memory to assign rewards at end of episode
        episode_mem[t][0:4] = observation
        episode_mem[t][4:6] = ActionValue
        episode_mem[t][6] = act
        
        #update state
        observation = observation_new

        #end episode
        if done:
            break

    #update finished episode memory with new reward
    #only update action value for actions that were taken, leave others unchanged
    #TODO - vectorize!!!
    alpha = 0.1
    for i in range (t+1):
        episode_mem[i,int(episode_mem[i,6])+4] = episode_mem[i,int(episode_mem[i,6])+4] *(1-alpha) + EpisodeReward * alpha

    #put episode into memory pool (experience replay)
    #TODO
    """    
    memory_index += 1
    if memory_index >= memory_max:
        memory_index = 0
        memory_full = 1
    """

    #update weights of NN based on random picks from memory targets
    if memory_full:
        batch = np.empty([t+1,7])
        for i in range(t+1):
            batch[i] = episode_mem[i]
        batch_in = batch[:,[0,1,2,3]]   #state
        batch_tar = batch[:,[4,5]]    #action value
        loss += ActorNet.train_on_batch(batch_in, batch_tar)[0]

    print('Episode {0}, reward = {1}'.format(episode,EpisodeReward))

    TotalReward += EpisodeReward

print('Total reward = {0}'.format(TotalReward))
ActorNet.save('CPv0_model.h5')
