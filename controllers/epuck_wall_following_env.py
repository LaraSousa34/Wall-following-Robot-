# Imports _____________________________________________________________________________________________________
import gym
import numpy as np
import logging
import matplotlib.pyplot as plt
from controllers.utils import move_robot_to
import math

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



#__________________________________________________________________________________________Let's Calculate the Reward!
def calculate_reward(distMin, touch_detected, time_without_collision):
    '''
    Calculates the reward
    '''
    reward = 0.0
    print(distMin)

    # reward for keeping the ideal distance from the wall
    #if 0.026 <= distMin <= 0.044:
    if 0.021 <= distMin <= 0.056:
        reward += 5.0

    #elif distMin < 0.026:
    elif distMin < 0.021:
        reward -= 7.0

    #elif distMin > 0.044:
    elif distMin > 0.056:
        reward -= 7 # Penalty


    reward += time_without_collision * 0.01  # little reward for time without collision

    print("distMin:", distMin, "reward:", reward, "touch_detected:", touch_detected)
    return reward






# Ambiente Gym personalizado_________________________________________________________________________
class WallFollowingEnv(gym.Env):
    '''
    Environment for controlling the
    Wall Following
    '''
    def __init__(self, robot, axle_length=0.057, wheel_radius=0.0205):
        super(WallFollowingEnv, self).__init__()

        self.robot = robot
        self.axle_length = axle_length
        self.wheel_radius = wheel_radius
        self.timestep = int(self.robot.getBasicTimeStep())

        # Sensores
        self.lidar = self.robot.getDevice('lidar')
        self.lidar.enable(self.timestep)
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.timestep)
        self.touch = self.robot.getDevice('touch sensor')
        self.touch.enable(self.timestep)

        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.robot.step(self.timestep)

        # Normalização dos dados do LIDAR
        self.lidar_max_range = 2.5

        if self.lidar.getRangeImage():
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(len(self.lidar.getRangeImage()),), dtype=np.float32)
        else:
            logging.warning("Lidar data not available at initialization.")

        self.robot.step(self.timestep)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.current_step = 0
        self.max_steps = 1000
        self.current_reward = 0
        self.total_reward = 0
        self.episode_rewards = []
        self.time_without_collision = 0  # Timer para tempo sem colisão
        self.prev_action = np.zeros(2)  # Armazena a ação anterior



    def cmd(self, action):
        if self.left_motor is None or self.right_motor is None:
            logging.error("Error: Could not find the motors. Please check the motor names.")
            return

        max_linear_velocity = 10.0
        max_angular_velocity = 10.0

        linear_velocity = 0.8 * max_linear_velocity
        angular_velocity = action[1] * max_angular_velocity

        left_wheel_velocity = linear_velocity - (angular_velocity * self.axle_length / 2)
        right_wheel_velocity = linear_velocity + (angular_velocity * self.axle_length / 2)

        left_wheel_velocity = self.clamp_velocity(left_wheel_velocity)
        right_wheel_velocity = self.clamp_velocity(right_wheel_velocity)

        print(f"linear_velocity: {linear_velocity}, angular_velocity: {angular_velocity}")
        print(f"left_wheel_velocity: {left_wheel_velocity}, right_wheel_velocity: {right_wheel_velocity}")

        self.left_motor.setVelocity(left_wheel_velocity)
        self.right_motor.setVelocity(right_wheel_velocity)




    def clamp_velocity(self, velocity, max_velocity=10.0):
        return max(min(velocity, max_velocity), -max_velocity)

    def smooth_action(self, action, alpha=0.01):
        #quanto menor for o alpha, mais suave será a transição entre as ações
        return alpha * action + (1 - alpha) * self.prev_action

    def check_termination_conditions(self, distMin):
        '''

        Treino:
            Angulo Reto: 0.50 , 0.76  || x_max=1.55; x_min=0.15; y_max=1.55; y_min=0.15
            Angulo2: 0.56, 0.42 || x_max=1.80; x_min=0.25; y_max=1.35; y_min=0.10
            Curva2: 1.04, 1.62 || x_max=1.80; x_min=0.25; y_max=1.75; y_min=0.15
        '''

        if self.touch.getValue() ==1:
            logging.info("Termination: Collision detected.")
            return True
            #return False

        #change this values for the right environment setup (boundaries)
        x_max = 1.80
        x_min = 0.25
        y_max = 1.75
        y_min = 0.15

        gps_values = self.gps.getValues()
        x = gps_values[0]
        y = gps_values[1]

        #if x >= x_max or x <= x_min or y >= y_max or y <= y_min or distMin>=0.044:
        if x >= x_max or x <= x_min or y >= y_max or y <= y_min or distMin>=0.056:
            logging.info("Termination: Out of bounds.")
            return True

        return False

    def step(self, action):
        print("Entra na Step Function")
        try:
            action = self.smooth_action(action)  # Suaviza a ação
            self.cmd(action)
            self.prev_action = action  # Armazena a ação atual para a próxima iteração
            self.robot.step(self.timestep)

            obs = self._get_obs()
            distMin = obs.min()
            touch_detected = self.touch.getValue() > 0.7

            if not touch_detected:
                self.time_without_collision += 1  # Incrementar o timer se não houver colisão
            else:
                self.time_without_collision = 0  # Resetar o timer em caso de colisão

            reward = calculate_reward(distMin, touch_detected, self.time_without_collision)
            done = self.check_termination_conditions(distMin)

            self.current_step += 1
            self.total_reward += reward

            logging.info(f"Current step: {self.current_step}, Reward: {reward}, Total Reward: {self.total_reward}")

            return obs, reward, done, {}
        except Exception as e:
            logging.error(f"Error in step function: {e}")
            return self._get_obs(), 0, True, {}

    def reset(self):

        logging.info(f"Episode finished with {self.current_step} timesteps and total reward: {self.total_reward}")
        self.robot.step(32)
        logging.info(f"Episode finished with total reward: {self.total_reward}")
        self.episode_rewards.append(self.total_reward)

        self.total_reward = 0
        self.current_step = 0
        self.time_without_collision = 0  # Resetar o timer no início de um novo episódio

        initial_position = (1.04, 1.62) #change this positions for the right initial position to the current environment
        #warp_robot(self.robot, "e-puck", initial_position)
        self.robot.step()

        gps_readings = self.gps.getValues()
        actual_position = (gps_readings[0], gps_readings[1])
        compass_readings = self.compass.getValues()
        actual_orientation = math.atan2(compass_readings[0], compass_readings[1])

        move_robot_to(self.robot, actual_position, actual_orientation, initial_position, 10, 10)

        obs = self._get_obs()
        logging.info("Environment reset.")
        return obs

    def _get_obs(self):
        '''
        Coleta e normaliza os dados do LIDAR para serem usados
        como observação pelo agente.
        '''
        raw_obs = self.lidar.getRangeImage()


        if raw_obs is None:
            obs = np.zeros(self.observation_space.shape[0])
        else:
            obs = np.array(raw_obs)
            obs[np.isinf(obs)] = self.lidar_max_range
            obs = np.nan_to_num(obs, nan=0.0)
            obs /= self.lidar_max_range
            print(obs)
        return obs

    def close(self):
        pass

