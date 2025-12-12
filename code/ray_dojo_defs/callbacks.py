from ray.rllib.callbacks.callbacks import RLlibCallback

class EpisodeReturn(RLlibCallback):
    def __init__(self):
        super().__init__()
        # Keep some global state in between individual callback events.
        self.overall_sum_of_rewards = 0.0

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        self.overall_sum_of_rewards += episode.get_return()
        metrics_logger.log_value("episode_return", self.overall_sum_of_rewards)