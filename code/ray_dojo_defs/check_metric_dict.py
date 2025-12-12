import pickle
from pprint import pprint

from env_class import MKII_Single_Env, MKII_obs_space
from single_play_agent import Kombatant
from callbacks import EpisodeReturn

dict = pickle.load(open('/kombat_artifacts/debug_metrics.p', 'rb'))
dict['config'] = None

#pprint(dict.keys())
