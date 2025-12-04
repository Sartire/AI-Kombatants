import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOPolicyNetwork(nn.Module):
    """
    Policy network for PPO with dual inputs: images and numeric vectors.
    Outputs logits for discrete button actions.
    """
    
    def __init__(
        self,
        img_channels=3,
        img_height=84,
        img_width=84,
        vec_size=8,
        num_buttons=6,
        hidden_size=512
    ):
        """
        Args:
            img_channels: Number of image channels (e.g., 3 for RGB, 1 for grayscale)
            img_height: Height of input image
            img_width: Width of input image
            vec_size: Dimension of numeric vector input
            num_buttons: Number of discrete button actions
            hidden_size: Size of hidden layer before output
        """
        super(PPOPolicyNetwork, self).__init__()
        
        self.img_channels = img_channels
        self.img_height = img_height
        self.img_width = img_width
        self.vec_size = vec_size
        self.num_buttons = num_buttons
        
        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of flattened conv output
        conv_out_size = self._get_conv_output_size()
        
        # Fully connected layers for vector input
        self.fc_vec = nn.Linear(vec_size, 128)
        
        # Combined processing
        self.fc_combined = nn.Linear(conv_out_size + 128, hidden_size)
        
        # Policy head (actor) - outputs logits for each button
        self.policy_head = nn.Linear(hidden_size, num_buttons)
        
        # Value head (critic) - outputs state value estimate
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_conv_output_size(self):
        """Calculate the output size after conv layers"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.img_channels, self.img_height, self.img_width)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return x.view(1, -1).size(1)
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Smaller init for policy head (common in PPO)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0)
        
        # Value head init
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0)
    
    def forward(self, img, vec):
        """
        Forward pass through the network.
        
        Args:
            img: Image tensor of shape (batch, channels, height, width)
            vec: Numeric vector of shape (batch, vec_size)
        
        Returns:
            action_logits: Logits for each button (batch, num_buttons)
            value: State value estimate (batch, 1)
        """
        # Process image through conv layers
        x_img = F.relu(self.conv1(img))
        x_img = F.relu(self.conv2(x_img))
        x_img = F.relu(self.conv3(x_img))
        x_img = x_img.view(x_img.size(0), -1)  # Flatten
        
        # Process vector input
        x_vec = F.relu(self.fc_vec(vec))
        
        # Combine both streams
        x_combined = torch.cat([x_img, x_vec], dim=1)
        x = F.relu(self.fc_combined(x_combined))
        
        # Get action logits and value estimate
        action_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return action_logits, value
    
    def get_action(self, img, vec, deterministic=False):
        """
        Sample an action from the policy.
        
        Args:
            img: Image tensor
            vec: Numeric vector
            deterministic: If True, select argmax action instead of sampling
        
        Returns:
            action: Selected action for each button (batch, num_buttons)
            log_prob: Log probability of the action (batch,)
            value: State value estimate (batch, 1)
        """
        action_logits, value = self.forward(img, vec)
        
        # Create distribution for each button (independent Bernoulli)
        probs = torch.sigmoid(action_logits)
        
        if deterministic:
            action = (probs > 0.5).float()
        else:
            action = torch.bernoulli(probs)
        
        # Calculate log probability
        log_prob = (action * torch.log(probs + 1e-8) + 
                    (1 - action) * torch.log(1 - probs + 1e-8)).sum(dim=-1)
        
        return action, log_prob, value
    
    def evaluate_actions(self, img, vec, actions):
        """
        Evaluate actions for PPO updates.
        
        Args:
            img: Image tensor
            vec: Numeric vector
            actions: Actions taken (batch, num_buttons)
        
        Returns:
            log_prob: Log probability of actions (batch,)
            entropy: Entropy of the policy (batch,)
            value: State value estimate (batch, 1)
        """
        action_logits, value = self.forward(img, vec)
        
        probs = torch.sigmoid(action_logits)
        
        # Log probability of taken actions
        log_prob = (actions * torch.log(probs + 1e-8) + 
                    (1 - actions) * torch.log(1 - probs + 1e-8)).sum(dim=-1)
        
        # Entropy (for exploration bonus)
        entropy = -(probs * torch.log(probs + 1e-8) + 
                    (1 - probs) * torch.log(1 - probs + 1e-8)).sum(dim=-1)
        
        return log_prob, entropy, value


# Example usage
if __name__ == "__main__":
    # Create network
    policy = PPOPolicyNetwork(
        img_channels=3,
        img_height=84,
        img_width=84,
        vec_size=8,
        num_buttons=6,
        hidden_size=512
    )
    
    # Example inputs
    batch_size = 4
    img = torch.randn(batch_size, 3, 84, 84)
    vec = torch.randn(batch_size, 8)
    
    # Forward pass
    action_logits, value = policy(img, vec)
    print(f"Action logits shape: {action_logits.shape}")
    print(f"Value shape: {value.shape}")
    
    # Sample actions
    actions, log_probs, values = policy.get_action(img, vec)
    print(f"Sampled actions: {actions}")
    print(f"Log probs shape: {log_probs.shape}")
    
    # Evaluate actions
    log_probs_eval, entropy, values_eval = policy.evaluate_actions(img, vec, actions)
    print(f"Entropy shape: {entropy.shape}")