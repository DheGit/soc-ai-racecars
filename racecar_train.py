import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adamax
from keras.layers import Dense, Activation
from keras.models import load_model
import random
from datetime import datetime

vector_size = 111
env = env = gym.make('CarRacing-v1')

SAVE_FILE = 'save.h5'
BOTTOM_BAR_HEIGHT = 12

class Agent:
    def __init__(self, env):
        self.env = env
        self.model = createNetwork()  # one feedforward nn for all actions.

    def predict(self, s):
        return self.model.predict(s.reshape(-1, vector_size), verbose=0)[0]

    def update(self, s, G):
        self.model.fit(s.reshape(-1, vector_size), np.array(G).reshape(-1, 11), epochs=1, verbose=0)

    def sample_action(self, s, eps):
        qval = self.predict(s)
        if np.random.random() < eps:
            return random.randint(0, 10), qval
        else:
            return np.argmax(qval), qval

def generateAction(output_value):
    # we reduce the action space to 15 values.  9 for steering, 6 for gas/brake.
    # to reduce the action space, gas and brake cannot be applied at the same time.
    # as well, steering input and gas/brake cannot be applied at the same time.
    # similarly to real life drive, you brake/accelerate in straight line, you coast while sterring.

    gas = 0.0
    brake = 0.0
    steer = 0.0

    # output value ranges from 0 to 10

    if output_value <= 8:
        # steering. brake and gas are zero.
        output_value -= 4
        steer = float(output_value) / 4
    elif output_value >= 9 and output_value <= 9:
        output_value -= 8
        gas = float(output_value) / 3 # 33%
    elif output_value >= 10 and output_value <= 10:
        output_value -= 9
        brake = float(output_value) / 2 # 50% brakes
    else:
        print("[WARNING] Invalid input for conversion to action values")

    return [steer, gas, brake]

def computeCarParameters(bottom_bar):
    right_steer = bottom_bar[BOTTOM_BAR_HEIGHT//2, 36:46].mean() / 255
    left_steer = bottom_bar[BOTTOM_BAR_HEIGHT//2, 26:36].mean() / 255
    steer = (right_steer - left_steer + 1.0) / 2

    left_gyro = bottom_bar[BOTTOM_BAR_HEIGHT//2, 46:60].mean() / 255
    right_gyro = bottom_bar[BOTTOM_BAR_HEIGHT//2, 60:76].mean() / 255
    gyro = (right_gyro - left_gyro + 1.0) / 2

    speed = bottom_bar[:, 0][:-2].mean() / 255
    a1 = bottom_bar[:, 6][:-2].mean() / 255
    a2 = bottom_bar[:, 8][:-2].mean() / 255
    a3 = bottom_bar[:, 10][:-2].mean() / 255
    a4 = bottom_bar[:, 12][:-2].mean() / 255

    return [steer, speed, gyro, a1, a2, a3, a4]

def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

def transform(state):
    bottom_black_bar = state[84:, 12:]
    img = cv2.cvtColor(bottom_black_bar, cv2.COLOR_RGB2GRAY)
    res_bottom_bar = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]
    res_bottom_bar = cv2.resize(res_bottom_bar, (84, 12), interpolation = cv2.INTER_NEAREST)

    upper_field = state[:84, 6:90]
    img = cv2.cvtColor(upper_field, cv2.COLOR_RGB2GRAY)
    res_upper_field = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1]
    res_upper_field = cv2.resize(res_upper_field, (10, 10), interpolation = cv2.INTER_NEAREST)
    res_upper_field = res_upper_field.astype('float') / 255

    car_field = state[66:78, 43:53]
    img = cv2.cvtColor(car_field, cv2.COLOR_RGB2GRAY)
    car_field_bw = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)[1]
    res_car = [
        car_field_bw[:, 3].mean() / 255,
        car_field_bw[:, 4].mean() / 255,
        car_field_bw[:, 5].mean() / 255,
        car_field_bw[:, 6].mean() / 255
    ]

    return res_bottom_bar, res_upper_field, res_car

def createNetwork():
    if os.path.exists(SAVE_FILE):
        return load_model(SAVE_FILE)
        
    model = Sequential()
    model.add(Dense(512, input_shape=(vector_size,)))# 7x7 + 3.  or 14x14 + 3
    model.add(Activation('relu'))

    model.add(Dense(11))
    model.add(Activation('linear'))

    adamax = Adamax()
    model.compile(loss='mse', optimizer=adamax)
    model.summary()
    
    return model

def runEpisode(env, model, eps, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done:
        bottom_bar, field, car = transform(observation)

        state = np.concatenate((np.array([computeCarParameters(bottom_bar)]).reshape(1,-1).flatten(), field.reshape(1,-1).flatten(), car), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1      
        prev_state = state

        argmax_qval, qval = model.sample_action(state, eps)
        action = generateAction(argmax_qval)
        observation, reward, done, info = env.step(action)

        bottom_bar, field, car = transform(observation)        
        state = np.concatenate((np.array([computeCarParameters(bottom_bar)]).reshape(1,-1).flatten(), field.reshape(1,-1).flatten(), car), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1      
        
        # update the model
        # standard Q learning TD(0)
        next_qval = model.predict(state)
        G = reward + gamma*np.max(next_qval)
        y = qval[:]
        y[argmax_qval] = G
        model.update(prev_state, y)
        totalreward += reward
        iters += 1
        
        if iters > 1500:
            print("This episode is stuck")
            break
        
    return totalreward, iters

def getEpsilon(episode_index):
    return 0.5/np.sqrt(episode_index + 900)

gamma = 0.99
NUM_EPISODES = 100

def main():
    model = Agent(env)
    net_reward_list = np.empty(NUM_EPISODES)

    for n in range(NUM_EPISODES):
        eps = getEpsilon(n)
        total_reward, iters = runEpisode(env, model, eps, gamma)
        net_reward_list[n] = total_reward
        if n % 1 == 0:
            print("="*30)
            print("\n\tEp:", n, "\n\tsteps", iters, "\n\tTotal Reward:", total_reward, "\n\teps:", eps, "\n\tAverage Reward (last 100):", net_reward_list[max(0, n-100):(n+1)].mean())        
        if n % 10 == 0:
            model.model.save(SAVE_FILE)

    print("Average reward (last 100 episodes):", net_reward_list[-100:].mean())
    print("Total steps:", net_reward_list.sum())

    plt.plot(net_reward_list)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(net_reward_list)

if __name__=='__main__':
    main()