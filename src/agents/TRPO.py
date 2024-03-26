import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from agents.help_func import *

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def trpo_iter(states, actions, log_probs, returns, advantage):
    for state, action, old_log_probs, return_, advantage in zip(states, actions, log_probs, returns, advantage):
        yield state, action, old_log_probs, return_, advantage

def conjugate_gradients(Avp, b, nsteps=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()  # Note that PyTorch's operations use clone() instead of copy.
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

def trpo_update(states, actions, log_probs, returns, advantages, model, optimizer, max_kl, damping, cg_iters):
    for state, action, old_log_probs, return_, advantage in trpo_iter(states, actions, log_probs, returns, advantages):
        logits, value = model(state)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        entropy = dist.entropy().mean()
        new_log_probs = dist.log_prob(action)

        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantage

        loss = -torch.min(surr1, torch.clamp(ratio, 1.0 - max_kl, 1.0 + max_kl) * advantage).mean()

        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()

        def Hx(x):
            grads = torch.autograd.grad(torch.dot(loss_grad, x), model.parameters(), retain_graph=True)
            hessian_vector_product = torch.cat([grad.view(-1) for grad in grads]).detach() + damping * x
            return hessian_vector_product + 1e-8 * x

        def fisher_vector_product(p):
            with torch.no_grad():
                Avp = Hx(p)
                return Avp + damping * p

        step_dir = conjugate_gradients(fisher_vector_product, loss_grad, cg_iters)

        shs = 0.5 * torch.dot(step_dir, Hx(step_dir))
        lagrange_multiplier = torch.sqrt(shs / max_kl)
        step = step_dir / lagrange_multiplier

        natural_gradient = step
        grad_norm = torch.norm(natural_gradient)
        if grad_norm > max_kl:
            natural_gradient *= max_kl / grad_norm

        full_step = natural_gradient
        loss_improve = -torch.dot(loss_grad, full_step)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

def trpo_train(env, model, optimizer, device, max_frames, num_steps, threshold_reward, results_dir, model_file_name, save_frequency, data_saver, max_kl=0.01, damping=0.1, cg_iters=10):
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
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)

            action = dist.sample()
            next_state, reward, terminated, truncated, info = custom_step(env, action)
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
        trpo_update(torch.cat(states), torch.cat(actions), torch.cat(log_probs).detach(), torch.cat(returns).detach(), torch.cat(returns).detach() - torch.cat(values).detach(), model, optimizer, max_kl, damping, cg_iters)

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

def trpo_agent_training(env, trpo_model, params):
    num_steps = int(params["num_steps"])
    learning_rate = float(params["learning_rate"])
    max_frames = int(params["max_frames"])
    threshold_reward = float(params["threshold_reward"])
    results_dir = params["results_dir"]
    model_file_name = params["model_file_name"]
    save_frequency = int(params["save_frequency"])
    damping = float(params["damping"])
    cg_iters = int(params["cg_iters"])
    max_kl = float(params["max_kl"])

    data_saver = DataSaver(results_dir, model_file_name, save_frequency)
    data_saver.save_params_to_json(params)
    env.data_saver = data_saver

    trpo_model = trpo_model.to(device)
    optimizer = torch.optim.Adam(trpo_model.parameters(), lr=learning_rate)

    episode_rewards = trpo_train(env=env,
                                model=trpo_model,
                                optimizer=optimizer,
                                device=device,
                                max_frames=max_frames,
                                num_steps=num_steps,
                                threshold_reward=threshold_reward,
                                results_dir=results_dir,
                                model_file_name=model_file_name,
                                save_frequency=save_frequency,
                                data_saver=data_saver,
                                max_kl=max_kl,
                                damping=damping,
                                cg_iters=cg_iters)
    
    # Save the trained model
    data_saver.save_model_data(trpo_model, num_steps, True)
    env.close()

    return episode_rewards
