import json
import pickle
import collections

import numpy as np
import scipy.stats as st

from recommendation import RecommendationEngine


total_runtimes_sa = collections.defaultdict(list)
total_runtimes_tb = collections.defaultdict(list)

total_energy_advantage = collections.defaultdict(int)

eng = RecommendationEngine()
db = eng.get_database()
i = 0
for _, metadata in db.iter_metadata():
    i += 1
    print('Approx : ' + str(metadata.approx))
    if i % 10 == 0:
        print(i)
        print(metadata)


print("Number of QUBOs:", i)

