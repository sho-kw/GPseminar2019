from hforl.agents import DQNAgent
from hforl.policy import Boltzmann
from hforl.trainer import GymTrainer
import numpy as np

from keras import backend as K

from keras.optimizers import Adam
from keras.utils import plot_model

from keras.models import Sequential, Model
from keras.layers import InputLayer, Input, Dense, Lambda, Add

import math
import gym
import time

def build(input, *nodes):
    x = input
    for node in nodes:
        if callable(node):
            x = node(x)
        elif isinstance(node, list):
            x = [build(x, branch) for branch in node]
        elif isinstance(node, tuple):
            x = build(x, *node)
        else:
            x = node
    return x

def main():
    env = gym.make('CartPole-v0')
    env.env.theta_threshold_radians = 3 * 2 * math.pi / 360
    #
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n

    i = Input(shape=state_shape)
    core = build(i,
                 Dense(64, activation='relu'),
                 Dense(64, activation='relu'),
                 Dense(64, activation='relu'),
                 Dense(num_actions+1),
                 Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - 0.0*K.mean(a[:, 1:], keepdims=True),
                             output_shape=(num_actions,))
                 )
      
    core = Model(i, core)
    agent = DQNAgent(core_model=core,
                     action_space=num_actions,
                     optimizer='adam',
                     policy=Boltzmann(),
                     memory=1<<16,
                     target_model_update=1,
                     warmup=100,
                     batch_size=32,
                     ) 
    trainer = GymTrainer(env, agent)

    # training
    result = trainer.train(200)
    # test
    result = trainer.test(5, render=True)
    for i, steps in enumerate(result['steps']):
        print('episode {}: {} steps'.format(i, steps))

if __name__ == '__main__':
    start = time.time()
    main()
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
