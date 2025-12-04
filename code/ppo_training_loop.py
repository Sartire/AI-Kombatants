import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class PPOTrainer:
    def __init__(
        self,
        policy,
        env,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_steps=2048,
        n_epochs=10,
        batch_size=64,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        PPO Trainer with clipped objective.
        
        Args:
            policy: PPOPolicyNetwork instance
            env: Gym environment
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Max gradient norm for clipping
            n_steps: Steps to collect before update
            n_epochs: Number of optimization epochs per update
            batch_size: Minibatch size for updates
            device: Device to train on
        """
        self.policy = policy.to(device)
        self.env = env
        self.device = device
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Storage for rollouts
        self.reset_rollout_buffer()
        
    def reset_rollout_buffer(self):
        """Initialize storage for collecting trajectories"""
        self.rollout = {
            'imgs': [],
            'vecs': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
    
    def collect_rollouts(self):
        """Collect n_steps of experience from the environment"""
        obs = self.env.reset()
        
        for step in range(self.n_steps):
            # Split observation into image and vector
            img, vec = self.split_observation(obs)
            
            # Convert to tensors
            img_t = torch.FloatTensor(img).unsqueeze(0).to(self.device)
            vec_t = torch.FloatTensor(vec).unsqueeze(0).to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(img_t, vec_t)
            
            # Convert action to numpy for env
            action_np = action.cpu().numpy()[0]
            
            # Take step in environment
            next_obs, reward, done, info = self.env.step(action_np)
            
            # Store transition
            self.rollout['imgs'].append(img)
            self.rollout['vecs'].append(vec)
            self.rollout['actions'].append(action.cpu().numpy()[0])
            self.rollout['rewards'].append(reward)
            self.rollout['values'].append(value.cpu().numpy()[0, 0])
            self.rollout['log_probs'].append(log_prob.cpu().numpy()[0])
            self.rollout['dones'].append(done)
            
            obs = next_obs
            
            if done:
                obs = self.env.reset()
        
        # Bootstrap value for last state if not done
        img, vec = self.split_observation(obs)
        img_t = torch.FloatTensor(img).unsqueeze(0).to(self.device)
        vec_t = torch.FloatTensor(vec).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, _, last_value = self.policy.get_action(img_t, vec_t)
            last_value = last_value.cpu().numpy()[0, 0]
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            self.rollout['rewards'],
            self.rollout['values'],
            self.rollout['dones'],
            last_value
        )
        
        self.rollout['advantages'] = advantages
        self.rollout['returns'] = returns
    
    def split_observation(self, obs):
        """
        Split observation into image and vector components.
        Adapt this to match your environment's observation space.
        
        Example assumes obs is a dict with 'image' and 'vector' keys.
        """
        if isinstance(obs, dict):
            img = obs['image']
            vec = obs['vector']
        else:
            # If obs is a tuple or you have a different structure, adapt here
            # Example: obs = (img, vec)
            img, vec = obs
        
        # Ensure correct format (C, H, W) for image
        if img.ndim == 3 and img.shape[-1] in [1, 3, 4]:
            img = np.transpose(img, (2, 0, 1))
        
        # Normalize image to [0, 1] if needed
        if img.max() > 1.0:
            img = img / 255.0
        
        return img, vec
    
    def compute_gae(self, rewards, values, dones, last_value):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        advantages = np.zeros(len(rewards))
        last_gae = 0
        
        # Append last value for bootstrapping
        values = values + [last_value]
        
        # Compute GAE backwards
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                last_gae = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae
        
        # Returns are advantages + values
        returns = advantages + values[:-1]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update_policy(self):
        """Perform PPO update using collected rollouts"""
        # Convert rollout data to tensors
        imgs = torch.FloatTensor(np.array(self.rollout['imgs'])).to(self.device)
        vecs = torch.FloatTensor(np.array(self.rollout['vecs'])).to(self.device)
        actions = torch.FloatTensor(np.array(self.rollout['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.rollout['log_probs'])).to(self.device)
        advantages = torch.FloatTensor(self.rollout['advantages']).to(self.device)
        returns = torch.FloatTensor(self.rollout['returns']).to(self.device)
        
        # Create dataset
        dataset_size = len(imgs)
        indices = np.arange(dataset_size)
        
        # Training metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        # Perform multiple epochs of optimization
        for epoch in range(self.n_epochs):
            # Shuffle data
            np.random.shuffle(indices)
            
            # Mini-batch updates
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_imgs = imgs[batch_indices]
                batch_vecs = vecs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions with current policy
                log_probs, entropy, values = self.policy.evaluate_actions(
                    batch_imgs, batch_vecs, batch_actions
                )
                
                # PPO clipped objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                values = values.squeeze()
                value_loss = 0.5 * (values - batch_returns).pow(2).mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.value_loss_coef * value_loss + 
                       self.entropy_coef * entropy_loss)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        # Return average losses
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses)
        }
    
    def train(self, total_timesteps, log_interval=10):
        """
        Main training loop.
        
        Args:
            total_timesteps: Total number of environment steps
            log_interval: How often to log progress (in updates)
        """
        n_updates = total_timesteps // self.n_steps
        episode_rewards = deque(maxlen=100)
        current_episode_reward = 0
        
        print(f"Training for {total_timesteps} timesteps ({n_updates} updates)")
        
        for update in range(1, n_updates + 1):
            # Collect rollouts
            self.collect_rollouts()
            
            # Track episode rewards
            for reward, done in zip(self.rollout['rewards'], self.rollout['dones']):
                current_episode_reward += reward
                if done:
                    episode_rewards.append(current_episode_reward)
                    current_episode_reward = 0
            
            # Update policy
            losses = self.update_policy()
            
            # Reset rollout buffer
            self.reset_rollout_buffer()
            
            # Logging
            if update % log_interval == 0:
                timesteps = update * self.n_steps
                mean_reward = np.mean(episode_rewards) if episode_rewards else 0
                print(f"Update {update}/{n_updates} | Timesteps: {timesteps}")
                print(f"  Mean reward (last 100 eps): {mean_reward:.2f}")
                print(f"  Policy loss: {losses['policy_loss']:.4f}")
                print(f"  Value loss: {losses['value_loss']:.4f}")
                print(f"  Entropy: {losses['entropy_loss']:.4f}")
                print()
        
        print("Training complete!")


# Example usage
if __name__ == "__main__":
    import gym
    from ppo_policy_network import PPOPolicyNetwork  # Import your policy
    
    # Create environment
    # This is a placeholder - replace with your actual environment
    env = gym.make('YourEnv-v0')
    
    # Create policy network
    policy = PPOPolicyNetwork(
        img_channels=3,
        img_height=84,
        img_width=84,
        vec_size=8,
        num_buttons=6,
        hidden_size=512
    )
    
    # Create trainer
    trainer = PPOTrainer(
        policy=policy,
        env=env,
        lr=3e-4,
        n_steps=2048,
        batch_size=64
    )
    
    # Train
    trainer.train(total_timesteps=1_000_000, log_interval=10)
    
    # Save model
    torch.save(policy.state_dict(), 'ppo_policy.pth')