from ray.rllib.callbacks.callbacks import RLlibCallback
import numpy as np
class EpisodeReturn(RLlibCallback):
    

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        
        reward_total = np.sum(episode.get_rewards())
        metrics_logger.log_value("episode_return", reward_total, reduce="mean", window=50)

