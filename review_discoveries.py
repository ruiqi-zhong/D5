import json
import pandas as pd

all_discoveries = json.load(open('discoveries.json', 'r'))

for discovery_set in all_discoveries:
    print('=========')
    print('problem_id', discovery_set['problem_id'])
    print('described corpus', discovery_set['described corpus'])
    print('research goal')
    print(discovery_set['research goal'])
    df = pd.DataFrame(discovery_set['rows'])
    print(df)
