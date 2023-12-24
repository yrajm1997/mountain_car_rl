import gymnasium as gym
import random
import numpy as np
np.random.seed(42)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


# Create MountainCar environment
env = gym.make("MountainCar-v0", render_mode="rgb_array")

# Initial state
state_, info = env.reset()
#state_

# Write a funtion for model creation

def create_nn_model():

    # Network
    inputs = layers.Input(shape=(2,))
    layer1 = layers.Dense(10, activation='relu')(inputs)
    layer2 = layers.Dense(5, activation='relu')(layer1)
    output = layers.Dense(3)(layer2)

    return keras.Model(inputs= inputs, outputs= output)


# Create model
model = create_nn_model()
target_model = create_nn_model()
#model.summary()

# define the loss function and optimization algorithm
model.compile(loss='mse', optimizer='adam')

def one_hot_encode(action):
    one_hot = np.zeros(3)
    one_hot[int(action)] = 1
    return one_hot


def value_function_nn(state, model):
    result = model(state.reshape(-1,2))
    return result


############################# TRAINING ALGORITHM ##################################################

# Function Approximation

# Hyperparameters
alpha = 0.05        # Learning rate: Q value update step size
gamma = 0.99        # Discount factor: the factor at which the Q-Value gets decremented after each cycle
epsilon = 0.15       # epsilon to be used by greedy policy
episodes = 2000
max_steps_per_episode = 1000
steps_per_update = 200


# For plotting metrics
all_epochs_nn = []
all_rewards_nn = []


# Function approximation using TD(0)

for i in range(1, episodes + 1):
    # Initial state and reward
    state, info = env.reset()
    reward = 0
    done = False

    # Using epsilon-greedy policy for selecting the initial action to take
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()   # Explore action space
    else:
        action = np.argmax(value_function_nn(state, target_model).numpy()[0])     # Exploit learned values

    epochs, rewards = 0, 0
    count = 0

    while not done:

        # Take action on environment, and get reward and next-state
        next_state, reward, done, truncated, info = env.step(action)

        # Using epsilon-greedy policy for selecting the action to take
        if random.uniform(0, 1) < epsilon:
            next_action = env.action_space.sample()   # Explore action space
        else:
            next_action = np.argmax(value_function_nn(next_state, target_model).numpy()[0])     # Exploit learned values

        #Vs = value_function_nn(state, action, model).numpy()[0,0]
        
        #td_target = value_function_nn(state, model).numpy()
        Vs_ = value_function_nn(next_state, target_model).numpy()[0, next_action]
        td_target = reward + gamma*Vs_
        td_target = one_hot_encode(next_action) * td_target

        #scaled_state = scaler.transform(state.reshape(-1,2))
        model.fit(state.reshape(-1,2), td_target.reshape(-1, 3), verbose=0)

        if done:
            break

        action = next_action
        state = next_state

        epochs += 1
        rewards += reward

        count += 1
        
        if count > steps_per_update:
            target_model.set_weights(model.get_weights())

        if count > max_steps_per_episode:
            break

    if epsilon > 0.08:
        epsilon = epsilon - 2/episodes
    else:
        epsilon = 0.08

    all_epochs_nn.append(epochs)
    all_rewards_nn.append(rewards)

    print(f"Episode: {i}/{episodes}, Epsilon: {np.round(epsilon,2)}, Reward: {np.round(rewards,2)}")

    if i % 100 == 0:
        #clear_output(wait=True)
        print(" ")
        print(f"Episode: {i}/{episodes}, Epsilon: {np.round(epsilon,2)}, Reward: {np.round(rewards,2)}, Avg_score: {np.mean(all_rewards_nn[-100:]).round(2)}")
    
    if i % 1000 == 0:
        target_model.save_weights(f'trained_model/trained_weights_episode_{i}.h5')
        target_model.save_weights(f'trained_model/trained_json_weights_episode_{i}.json')

print("Training finished.\n")

