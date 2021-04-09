import numpy as np
import gym
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

env = gym.make('CartPole-v1')
desired_state = np.array([0, 0, 0, 0])
desired_mask = np.array([0, 0, 1, 0])

P, I, D = 0.1, 0.01, 0.5
obs = []
obs1 = []
obs2 = []
obs3 = []
dat = []

for i_episode in range(20):
    state = env.reset()
    integral = 0
    derivative = 0
    prev_error = 0
    for t in range(500):
        env.render()
        error = state - desired_state

        integral += error
        derivative = error - prev_error
        prev_error = error

        pid = np.dot(P * error + I * integral + D * derivative, desired_mask)
        dat.append(error[2])
        obs3.append(state[3])
        obs2.append(state[2])
        obs1.append(state[1])
        obs.append(state[0])
        action = sigmoid(pid)
        action = np.round(action).astype(np.int32)

        state, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            plt.subplot(5,1,1)
            plt.plot(obs, label = 'Position of cart')
            plt.legend()
            plt.subplot(5,1,2)
            plt.plot(obs1, label = 'Velocity of cart')
            plt.legend()
            plt.subplot(5,1,3)
            plt.plot(obs2, label = 'Angle of pole')
            plt.legend()
            plt.subplot(5,1,4)
            plt.plot(obs3, label = 'Pole velocity at tip')
            plt.legend()
            plt.subplot(5,1,5)
            plt.plot(dat, label = 'Error')
            plt.legend()
            plt.show()
            break
env.close()