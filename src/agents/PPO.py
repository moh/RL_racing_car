import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from agents.help_func import *

# Computing the advantage function using generalized advantage estimation
def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids], actions[rand_ids], log_probs[rand_ids], returns[rand_ids], advantage[rand_ids]

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages,model, optimizer, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            logits, value = model(state)
            probs = torch.softmax(logits, dim = -1) # converting the logits to probabilities
            dist = Categorical(probs)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def ppo_train(env, model, optimizer, device, max_frames, num_steps, ppo_epochs, mini_batch_size, threshold_reward, results_dir, model_file_name, save_frequency, data_saver):
    frame_idx = 0
    
    
    early_stop = False
    episode_idx = 0

    while frame_idx < max_frames and not early_stop:
        state, info = env.reset()
        episode_reward = 0 
        
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []
        entropy = 0
        cumulated_rewards_per_episode = []

        for _ in range(num_steps):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            logits, value = model(state)
            probs = torch.softmax(logits, dim = -1)
            dist = Categorical(probs)

            action = dist.sample()
            next_state, reward, terminated, truncated, info = custom_step(env, action) # env.step(action)
            done = terminated or truncated
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob.unsqueeze(0))
            values.append(value)
            rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(device))

            states.append(state)
            actions.append(action)

            state = next_state
            episode_reward += reward
            frame_idx += 1
            cumulated_rewards_per_episode.append(episode_reward)

        max_reward = max(cumulated_rewards_per_episode)
        
        next_value = torch.zeros(1, 1).to(device) if done else model(torch.FloatTensor(next_state).unsqueeze(0).to(device))[1]
        returns = compute_gae(next_value, rewards, masks, values)
        ppo_update(ppo_epochs, mini_batch_size, torch.cat(states), torch.cat(actions), torch.cat(log_probs).detach(), torch.cat(returns).detach(), torch.cat(returns).detach() - torch.cat(values).detach(), model, optimizer)

        if episode_idx % save_frequency == 0 or done:
            data_saver.save_model_data(model, episode_idx, last=done)
            data_saver.save_rewards_data({"total_reward": episode_reward, "max" : max_reward}, episode_idx)
            data_saver.save_gif(max_reward)
            print("reward at the end = ", episode_reward)
            print("reward max        = ", max_reward)
        if episode_reward > threshold_reward:
            early_stop = True
            print("Early stopping, training completed.")
        episode_idx += 1
        

    return cumulated_rewards_per_episode

def ppo_agent_training(env, ppo_model, params):
    num_steps = int(params["num_steps"])
    learning_rate = float(params["learning_rate"])
    ppo_epochs = int(params["ppo_epochs"])
    mini_batch_size = int(params["mini_batch_size"])
    max_frames = int(params["max_frames"])
    threshold_reward = float(params["threshold_reward"])

    results_dir = params["results_dir"]
    model_file_name = params["model_file_name"]
    save_frequency = int(params["save_frequency"])

    data_saver = DataSaver(results_dir, model_file_name, save_frequency)
    data_saver.save_params_to_json(params)
    env.data_saver = data_saver

    ppo_model = ppo_model.to(device)
    optimizer = torch.optim.Adam(ppo_model.parameters(), lr=learning_rate)

    episode_rewards = ppo_train(env=env,
                                               model=ppo_model,
                                               optimizer=optimizer,
                                               device=device,
                                               max_frames=max_frames,
                                               num_steps=num_steps,
                                               ppo_epochs=ppo_epochs,
                                               mini_batch_size=mini_batch_size,
                                               threshold_reward=threshold_reward,
                                               results_dir=results_dir,
                                               model_file_name=model_file_name,
                                               save_frequency=save_frequency,
                                               data_saver=data_saver)
    
    # Save the trained model
    data_saver.save_model_data(ppo_model, num_steps, True)
    env.close()

    return episode_rewards




