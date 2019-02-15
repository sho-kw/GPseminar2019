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
     
    gamma = 0.99 # PARAM
    eps = 0.05 # PARAM
    alpha = 1.0 # PARAM

    RIGHT = 1
    LEFT = 0
    
    training = True

    X = np.linspace(-1, 1, 10) # PARAM
    dX = np.linspace(-1, 1, 10) # PARAM
    W = np.linspace(-1, 1, 10) # PARAM
    dW = np.linspace(-1, 1, 10) # PARAM
    
    Q = nprand.rand(len(X) + 1, len(dX) + 1, len(W) + 1, len(dW) + 1, 2)

    def choose_action(state):
        if training and nprand.rand() < eps:
            return nprand.choice([RIGHT, LEFT])
        else:
            return np.argmax(Q[state_to_index(state)])
    
    def update(state, action, reward, end, next_state):
        if end:
            val = Q[state_to_index(state)][action]
            Q[state_to_index(state)][action] =  reward
        else:
            Q[state_to_index(state)][action] = reward + gamma * np.max(Q[state_to_index(next_state)])
    
    def state_to_index(state):
        x, dx, w, dw = state

        def search_index(val, l):
            idx = 0
            for boundary in l:
                if val > boundary:
                    idx += 1
                else:
                    break
            return idx

        return search_index(x, X), search_index(dx, dX), search_index(w, W), search_index(dw, dW)

    clear_count = 0
    for episode in range(2000 + 1):
        state = env.reset()
        done = False
        t = 0
        training = (episode % 200 != 0)
        while not done:
            if not training:
                env.render()
                sleep(0.01)
            action = choose_action(state)  # action = env.action_space.sample()
            next_state, _, done, _ = env.step(action)  # state, reward, done, info = env.step(action)
            if done:
                reward = 0
            else:
                reward = 1 - abs(next_state[2])
            update(state, action, reward, done, next_state)
            state = next_state
            t += 1
            if done:
                if not training:
                    print("Episode {} finished after {} timesteps".format(episode, t + 1))
                    if t >= 200:
                        print(state, 'CLEAR')
                        clear_count += 1
                    else:
                        print(state)
                break
    
    print('Result : ', clear_count)

if __name__ == '__main__':
    main()
