import pickle
from pprint import pprint

dict = pickle.load(open('/kombat_artifacts/debug_metrics.p', 'rb'))

pprint(dict.keys())
