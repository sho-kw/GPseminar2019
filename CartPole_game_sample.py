import numpy as np
import numpy.random as nprand
from time import sleep
import math
import gym


def main():
    env = gym.make('CartPole-v0')
    nprand.seed(123)
    env.seed(123)
    env.render()

    LEVEL = 1
    if LEVEL == 0:
        env.env.theta_threshold_radians = 45 * 2 * math.pi / 360
        SLOW = 0.1
    elif LEVEL == 1:
        env.env.theta_threshold_radians = 15 * 2 * math.pi / 360
        SLOW = 0.01
    else :
        env.env.theta_threshold_radians = 12 * 2 * math.pi / 360
        SLOW = 0
        
   

    RIGHT = 1
    LEFT = 0
    key_action = 0
    
    def choose_action(state):
        x, dx, w, dw = state
        if dw >= 0:
            return RIGHT
        if dw < 0:
            return LEFT
                
    def key_press(key, mod):
        nonlocal key_action
        if key == 65363:
            key_action = RIGHT
        elif key == 65361:
            key_action = LEFT

    env.env.unwrapped.viewer.window.on_key_press = key_press

    
    
    for episode in range(100):
        state = env.reset()
        done = False
        t = 0
        while not done:
            sleep(SLOW)
            env.render()
            action = choose_action(state)      #行動の選択#
            state, _, done, _ = env.step(action)
            t += 1
            if done:
                print("Episode {} finished after {} timesteps".format(episode, t + 1))
                if t >= 200:
                    print(state, 'CLEAR')
                else:
                    print(state)
                break
        sleep(1.0)

if __name__ == '__main__':
    main()
