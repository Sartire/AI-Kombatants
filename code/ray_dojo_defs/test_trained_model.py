import retro
import gymnasium as gym
import numpy as np
import os
import pandas as pd
from env_class import MKII_Single_Env, MKII_obs_space
from single_play_agent import Kombatant


from pathlib import Path
import torch

from ray.rllib.core.distribution.torch.torch_distribution import TorchCategorical
from ray.rllib.core.columns import Columns


skip1_path = Path('/kombat_artifacts/checkpoints/skip_1/PPO_MKII_Single_Env_47196_00001_1_2025-12-12_19-40-09/checkpoint_000000/learner_group/learner/rl_module/default_policy')

skip2_path = Path('/kombat_artifacts/checkpoints/skip_2/PPO_MKII_Single_Env_ae94a_00000_0_2025-12-12_20-04-31/checkpoint_000000/learner_group/learner/rl_module/default_policy')

recording_base = Path('/kombat_artifacts/recordings')

#model_check_path = Path('/kombat_artifacts/checkpoints/learner_group/learner/rl_module/default_policy')

def get_action(logits):
    dist = TorchCategorical(logits=logits)
    action = dist.sample().item()
    return action

def test_model(model_path, alias, num_runs=20):

    trained = Kombatant.from_checkpoint(model_path)
    trained.eval()
    # make record dir
    record_path = recording_base / alias
    record_path.mkdir(exist_ok=True, parents=True)


    env_config={
                     'state': 'Level1.JaxVsBaraka',
                     'record_dir': record_path,
                     'n_skip_steps': 0,
                     'skip_repeat': True,
                     'reset_delay': 174,
                     'health_weights': (-1,1)
                 }

    test_env = MKII_Single_Env(config=env_config)


    trials = {'total_reward': [],
              'agent_win': [],
              'num_match_steps': [],
              'fin_player_health': [],
              'fin_opponent_health': []
              }
    
    for match in range(num_runs):

        obs, info = test_env.reset()
        
        # get the initial data for the agent
        img = torch.tensor(obs['image']).unsqueeze(0)
        additional = torch.tensor(obs['additional_data']).unsqueeze(0)


        data = {'obs':{'image': img, 'additional_data': additional}}

        with torch.no_grad():
            output = trained(data)


        action = get_action(output[Columns.ACTION_DIST_INPUTS])

        truncated = False
        terminated = False

        reward_total = 0
        steps = 0
        while not terminated and not truncated:
            obs, reward, truncated, terminated, info = test_env.step(action)

            # convert obs to tensor
            img = torch.tensor(obs['image']).unsqueeze(0)
            additional = torch.tensor(obs['additional_data']).unsqueeze(0)
            data = {Columns.OBS:{'image': img, 'additional_data': additional},
                    Columns.STATE_IN: output[Columns.STATE_OUT]} # need hidden state from previous step
            # feed to network
            with torch.no_grad():
                output = trained(data)

            # Feed logits to distribution
            action = get_action(output[Columns.ACTION_DIST_INPUTS])

            reward_total += reward
            steps += 1

        if info['enemy_health'] <= 0:
            agent_win = 1
        else:
            agent_win = 0
            
        trials['total_reward'].append(reward_total)
        trials['agent_win'].append(agent_win)
        trials['num_match_steps'].append(steps)
        trials['fin_player_health'].append(info['health'])
        trials['fin_opponent_health'].append(info['enemy_health'])
    # end of matches
    trial_df = pd.DataFrame(trials, columns = ['total_reward', 'agent_win', 'num_match_steps', 'fin_player_health', 'fin_opponent_health'])
    trial_df['agent'] = alias
    test_env.stop_record()
    test_env.close()
    return trial_df



skip1_df = test_model(skip1_path, 'skip_1')
skip2_df = test_model(skip2_path, 'skip_2')

df = pd.concat([skip1_df, skip2_df])

df.to_csv(recording_base / 'test_results.csv')

print(df.query('agent_win == 1'))



