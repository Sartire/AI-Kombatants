from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

# Configure.
config = (
    PPOConfig()
    .environment("CartPole-v1")
    .training(
        train_batch_size_per_learner=2000,
        lr=0.0004,
    )
)

# Train through Ray Tune.
results = tune.Tuner(
    "PPO",
    param_space=config,
    # Train for 4000 timesteps (2 iterations).
    run_config=tune.RunConfig(stop={"num_env_steps_sampled_lifetime": 4000}),
).fit()

