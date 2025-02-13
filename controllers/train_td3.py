#importing____________________________________________________________________________________________________
import os
import logging
import numpy as np
import torch
from stable_baselines3 import PPO, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from epuck_wall_following_env import WallFollowingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from controller import Robot
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
import tensorflow as tf
from tensorflow import summary
from tensorflow.summary import create_file_writer
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')




# Set environment variable for Webots home__________________________________________________
if 'WEBOTS_HOME' not in os.environ:
    os.environ['WEBOTS_HOME'] = '/Applications/Webots.app'
    print("WEBOTS_HOME set in script to:", os.environ['WEBOTS_HOME'])
else:
    print("WEBOTS_HOME already set to:", os.environ['WEBOTS_HOME'])




# Check for MPS availability________________________________________________________________
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

print("Importing modules...")


#Hyperparameters ____________________________________________________________________


print("Modules imported successfully.")

hyperparams = {
    'learning_rate': [1e-3],
    'n_layers': [2],
    'layer_size': [128],
    'batch_size': [32],
    'gamma': [0.99],
    'entropy_coefficient': [0.01]
}




# Ensure the TensorBoard log directory exists___________________________________________
tensorboard_log_dir = "/tmp/tensorboard_logs"
os.makedirs(tensorboard_log_dir, exist_ok=True)



#Adding EarlyStoppingCallBack____________________________________________________________________
class EarlyStoppingCallback(BaseCallback):
    '''
    When mean reward is greater than the specified threshold,
    the training is stopped.
    '''

    def __init__(self, reward_threshold, verbose=0):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.reward_threshold = reward_threshold  # reward threshold
        self.best_mean_reward = -np.inf  # upgrade with the best mean reward seen.

    def _on_step(self) -> bool:
        # Get the current reward
        if 'episode_rewards' in self.locals:
            current_rewards = self.locals['episode_rewards'][-1]
        else:
            current_rewards = 0

        # Check if the current mean reward is above the threshold
        if current_rewards > self.reward_threshold:
            if self.verbose > 0:
                print(f"Stopping early: mean reward {current_rewards} exceeds threshold {self.reward_threshold}")
                print(f"Training stopped at timestep {self.num_timesteps}")
            return False  # Returning False stops the training

        return True  # Continue training




#CallBack logic ___________________________________________________________________________________________
class CustomTensorBoardCallback(BaseCallback):
    '''
    Custom TensorBoard
    '''

    def __init__(self, log_dir, verbose=0):
        super(CustomTensorBoardCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.writer = None

    def _on_training_start(self) -> None:
        self.writer = summary.create_file_writer(self.log_dir)

    def _on_step(self) -> bool:
        with self.writer.as_default():
            if 'rollout/ep_rew_mean' in self.locals:
                summary.scalar('rollout/ep_rew_mean', self.locals['rollout/ep_rew_mean'], step=self.num_timesteps)
            if 'rollout/ep_len_mean' in self.locals:
                summary.scalar('rollout/ep_len_mean', self.locals['rollout/ep_len_mean'], step=self.num_timesteps)
            if 'train/loss_policy' in self.locals:
                summary.scalar('train/loss_policy', self.locals['train/loss_policy'], step=self.num_timesteps)
            if 'train/loss_value' in self.locals:
                summary.scalar('train/loss_value', self.locals['train/loss_value'], step=self.num_timesteps)
            if 'train/entropy' in self.locals:
                summary.scalar('train/entropy', self.locals['train/entropy'], step=self.num_timesteps)
            if 'train/learning_rate' in self.locals:
                summary.scalar('train/learning_rate', self.locals['train/learning_rate'], step=self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        self.writer.close()

#Histograms on TensorBoard (CALLBACK)____________________________________________________________________
class HistogramCallback(BaseCallback):
    '''
    Callback to display a histogram of the training
    '''
    def __init__(self, log_dir, verbose=0):
        super(HistogramCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.writer = None

    def _on_training_start(self) -> None:
        self.writer = summary.create_file_writer(self.log_dir)

    def _on_step(self) -> bool:
        with self.writer.as_default():
            for name, param in self.model.policy.named_parameters():
                if param.requires_grad:
                    param = param.detach()
                summary.histogram(name, param.numpy(), step=self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        self.writer.close()



#____________________________________________________________________let's start training, saving or loading!
print("Setting up the environment...")
robot = Robot() #robot instance
env = WallFollowingEnv(robot)   #environment
print("Environment set up successfully.")
print(env)
logging.debug("Starting the training process...")

print("Starting training")
reward_threshold = 50  # Define a reward threshold for early stopping

#____________________________________________________________________
#defining the TD3 as the DRL model for training
#n_actions = env.action_space.shape[-1]
#action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
#model.save("TD3_wallfollowing")  # vai criar a pasta no dir de onde estamos a correr

# loading the saved model
#model = TD3.load(r'/Users/larasousa/Desktop/IRI/controllers/Training/TD3/20000', env)
#model = TD3.load(r'/Users/larasousa/Desktop/IRI/controllers/Training/TD3_2/20000', env)
#model = TD3.load(r'/Users/larasousa/Desktop/IRI/controllers/Training/TD3_3/20000', env)
#model = TD3.load(r'/Users/larasousa/Desktop/IRI/controllers/Training/TD3_4_5/40000', env)
model = TD3.load(r'/Users/larasousa/Desktop/IRI/controllers/Training/TD3_4_5_2/40000', env)


#____________________________________________________________________
max_iters = 2   #max number of times I want my model to train 10k timesteps
#timesteps = 10000
timesteps = 20000
iters = 0

while iters < max_iters:
    iters += 1

    #tensorboard config_____________________________________________________________________
    early_stopping_callback = EarlyStoppingCallback(reward_threshold=reward_threshold, verbose=1)
    custom_tensorboard_callback = CustomTensorBoardCallback(log_dir=tensorboard_log_dir, verbose=1)
    histogram_callback = HistogramCallback(log_dir=tensorboard_log_dir, verbose=1)

    model.learn(total_timesteps=timesteps,
                callback=[early_stopping_callback, custom_tensorboard_callback, histogram_callback])




    #________________________________________________________________________let's save our model!
    # create Trainig/TD3's folders in the same directory where we have the python files
    # inside that the files are saved

    #model.save(f"{'Training/TD3'}/{timesteps * iters}")
    #model.save(f"{'Training/TD3_2'}/{timesteps * iters}")
    #model.save(f"{'Training/TD3_3'}/{timesteps * iters}")
    #model.save(f"{'Training/TD3_4_5'}/{timesteps * iters}")
    #model.save(f"{'Training/TD3_4_5_2'}/{timesteps * iters}")
    model.save(f"{'Training/TD3_4_5_3'}/{timesteps * iters}")