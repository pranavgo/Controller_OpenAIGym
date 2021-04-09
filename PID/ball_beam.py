import gym
import ballbeam_gym
import matplotlib.pyplot as plt
import numpy as np

# pass env arguments as kwargs
kwargs = {'timestep': 0.05, 
          'beam_length': 1.0,
          'max_angle': 0.5,
          'init_velocity': 0.5,
          'max_timesteps': 100,
           }

# create env
env = gym.make('BallBeamBalance-v0', **kwargs)

integral = 0
prev_error = 0
# constants for PID calculation
Kp = 2.0
Kd = 1.0
ki = 0.0
ballPosition =[] 
beamAngle = []

for i_episode in range(3):
    obs = env.reset()
# simulate 1000 steps
    for t in range(1000): 
        env.render()
        integral += env.bb.x
        theta = Kp*(env.bb.x) + Kd*(env.bb.v) + ki*(integral)
        action = np.tanh(theta) * 1.5
        ballPosition.append(env.bb.x)
        beamAngle.append(env.bb.theta)
        
        obs, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            env.reset()
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(ballPosition, label = 'Position of Ball')
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(beamAngle, label = 'Angle of Beam')
            plt.legend()
            plt.show()
            break
env.close()        
         
