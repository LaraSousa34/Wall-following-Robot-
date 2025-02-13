

Project: Wall Following Robot with DRL

Overview

This project is part of the university course “Introduction to Intelligent Robotics” in the Bachelor’s degree program in Artificial Intelligence and Data Science. The main goal is to evaluate the performance and robustness of three different Deep Reinforcement Learning (DRL) models—PPO, DDPG, and TD3—in navigating various letter-shaped environments while following walls. The responsible professors are Luís Paulo Reis and Gonçalo Leão.

Directory Structure
entrega/
├── controllers/
│   ├── train_ddpg.py
│   ├── train_td3.py
│   ├── train_ppo.py
│   ├── epuck_wall_following_env.py
│   ├── utils.py
|   |── create_map.py
|   |── Test_Models.py
├── tensorboard/
│   ├── round_1/
|   |  |──screenshots from the train on tensorboard
│   └── round_2/
|   |  |──screenshot from the train on tensorboard
├── training/
│   ├── Mapas_Treino/
│   ├── last_iter/
|   |  |──PPO
|   |  |    |──Ronda_1
|   |  |    |──Ronda_2
|   |  |──DDPG
|   |  |    |──Ronda_1
|   |  |    |──Ronda_2
|   |  |──TD3
|   |  |    |──Ronda_1
|   |  |    |──Ronda_2
│   ├── ddpg_wallfollowing.zip
│   ├── ppo_wallfollowing.zip
│   └── TD3_wallfollowing.zip
├── testing/
│   ├── alphabet_round1/
│   |   ├── same_init_position_part1.jpeg
│   |   ├── same_init_position_part2.jpeg
│   |   ├── diff_init_position_part1.jpeg
│   |   ├── diff_init_position_part2.jpeg
│   ├── alphabet_round2/
│   ├── video/
│   |   ├── Resultados_Video.zip
│   └── mapas/
│   |   ├── Mapas_Teste.zip
└── ReadMe.txt

	•	controllers: Contains the Python scripts used for training the DRL models.
	•	train_ddpg.py: Script to train the DDPG model.
	•	train_td3.py: Script to train the TD3 model.
	•	train_ppo.py: Script to train the PPO model.
	•	epuck_wall_following_env.py: Environment setup script for wall-following using e-puck robot.
	•	utils.py: Utility functions used across different scripts.
	•	tensorboard: Contains TensorBoard logs for visualizing training metrics.
	•	round_1: TensorBoard logs from the first round of training.
	•	round_2: TensorBoard logs from the second round of training.
	•	training: Contains training data and models.
	•	Mapas_Treino: Training maps used in the project.
	•	last_iter: Saved models from the last iteration of training.
	•	ddpg_wallfollowing.zip: Trained DDPG model.
	•	ppo_wallfollowing.zip: Trained PPO model.
	•	TD3_wallfollowing.zip: Trained TD3 model.
	•	testing: Contains testing data and results.
	•	alphabet_round1: Results from testing in the first round.
	•	alphabet_round2: Results from testing in the second round.
	•	video: Video comparisons of different initial positions.
	•	mapas: Maps used during testing.
    •	create_map: Used to create the training and testing environments/maps
    •	Test_Models: Testing the DRL models for the letters


Operating Systems Compatibility

- This project is compatible with Linux, macOS, and Windows operating systems.

Dependencies

The following dependencies need to be installed to run the project:

- Python 3.x
- TensorFlow
- TensorBoard
- OpenAI Gym
- Webots
- NumPy

You can install the dependencies using pip:

pip install tensorflow tensorboard gym webots numpy

Instructions to Run the Code

1. Setup Environment:
   - Ensure Python 3.x and all dependencies are installed.
   - Install Webots and set up the environment as per the Webots installation guide.

2. Training:
   - Navigate to the `controllers` directory.
   - Run the desired training script:

     python train_ddpg.py
     python train_td3.py
     python train_ppo.py

   - Training logs and metrics will be saved in the `tensorboard` directory.

3. Testing:
   - Navigate to the `testing` directory.
   - Run the test scripts or use the provided trained models in the `training` directory.

4. Visualization:
   - Use TensorBoard to visualize training metrics: tensorboard --logdir=tensorboard

   - Open the provided videos in the `video` directory for visual comparisons.



Training and Testing Procedure

1.	Training Round 1:
	•	Each DRL model was trained with 60,000 timesteps across different environments.
	•	Performance metrics such as total collisions, total out-of-limits occurrences, and mean time without collisions were tracked using TensorBoard.
	•	Results indicated that PPO generally outperformed DDPG and TD3 in terms of consistency and fewer collisions.
2.	Testing with Different Initial Positions:
	•	Tested the robustness of the models by varying the initial positions.
	•	The impact of initial positioning on model performance was highlighted, with PPO maintaining consistent results, while DDPG and TD3 showed more variability.
3.	Training Round 2:
	•	Repeated the same procedure as in round 1 but with different distance values and doubling the timesteps per iteration.
	•	A table was created to record the environments, the model names in TensorBoard, the names under which the folders were saved, and the order of training.

Observations

	•	PPO: Showed the most consistent performance across different environments, with fewer collisions and higher mean times without collisions.
	•	DDPG and TD3: Displayed more variability in their performance, particularly in more complex environments.
	•	The results indicate that while the models can handle simpler environments, they struggle with more complex ones.

Contact
larasousapf@gmail.com, up202109782@up.pt, up202107955@fc.up.pt



