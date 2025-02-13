
import os
import logging

import matplotlib as plt

import sys
sys.path.append(r'C:\Program Files\Webots\lib\controller\python')
# Your existing imports and setup code
print("Importing modules...")
from stable_baselines3 import PPO,DDPG,TD3
from epuck_wall_following_envNOVO import WallFollowingEnv
import warnings
from controller import Robot
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress specific categories of warnings



def plot_rewards(episode_rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


"""
    Testar o modelopara x eps.

    descomentar para cada modelo
    """

print("Setting up the environment...")
robot=Robot()
env = WallFollowingEnv(robot)
print("Environment set up successfully.")
print(env)
logging.debug("Starting the testing process...")

model=PPO.load('Training\Ronda1\PPO_1\\20000', env)
#model=DDPG.load('Training\Ronda1\DDGP_1\\20000', env)
#model=TD3.load('Training\Ronda1\TD3_1\\20000', env)

eps_reward=[] #armazena total rewards em cada ep
count_collisions=0 #vai obter as colisoes totais
tempo_medio_sem_col=0 #vai obter as somas de tempos sem colisao, vai fazer a media
count_outlimits=0 #conta a vezes que saiu dos seus limites
num_episodes=100 #testar e obter resultados ao longo de x eps

for episode in range(num_episodes):
    obs = env.reset() # a cada novo ep, reset do ambiente. mudar cordenadas de pos inicial e limitacoes para cada mapa
    done = False
    total_reward = 0
    while not done:
         # Escolha a ação a partir do modelo treinado
        action, _states = model.predict(obs, deterministic=True) #obtem a acao

        # Execute a ação no ambiente
        obs, reward, done, info = env.step(action)

        #Acumula a recompensa
        total_reward += reward

    eps_reward.append(total_reward)
    count_collisions+=info[3]
    count_outlimits+=info[2]
    tempo_medio_sem_col+=info[0]

    print(f"Episode {episode + 1}: Total Reward: {total_reward}  Total Steps: {info[1]}  Time Without Collision: {info[0]}")

media=tempo_medio_sem_col / num_episodes
print(f"Total Eps {num_episodes}: Total Collisions: {count_collisions}  Total OutOfLimits: {count_outlimits}  Mean Time Without Collision: {media}")

