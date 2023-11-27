

import numpy as np
import pandas as pd
all_games = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQMaaNO_0JU-A2gdSyJpF-WEjJGqWqZdIIp9g9gHGpTdJ3G8l6BvV1PvtmrB3nUTHxnDC_zbiAp3sJx/pub?gid=0&single=true&output=csv')

all_games['timeDiff'] = all_games.groupby('game')['curr'].diff(1)

event_pairs = all_games[(all_games['timeDiff'] >= 1) & (all_games['timeDiff'] <= 7)
  & (all_games['replay'].shift(1) == 0) &(all_games['replay'] == 0)]
all_games['Pair'] = all_games.index.isin(event_pairs.index-1)
all_games['PairX'] = all_games.index.isin(event_pairs.index)

all_games['Pair'] = all_games.apply(lambda row: -2 if row['Pair'] and row['PairX'] else -1 if row['Pair'] and not row['PairX'] else 1 if row['PairX'] and not row['Pair']  else 0, axis=1)
all_games.drop(columns=['PairX'], inplace=True)

all_games['Difference'] = (all_games['end'].shift(-1) - all_games['start']).where(all_games['Pair'] < 0)

for index, row in all_games.iterrows():
    if not row['Pair']<0:
        continue
    next_row = all_games.iloc[index + 1]
    if row['Difference'] > 10:
      average_curr = (row['curr'] + next_row['curr']) / 2
      total_time_diff= (10 - (next_row['curr']-row['curr']))/2
      crop_start = round( max(row['start'], row['curr'] - total_time_diff), 2)
      crop_end = round( min(next_row['end'], next_row['curr'] + total_time_diff), 2)
      all_games.at[index, 'pair_start'] = crop_start
      all_games.at[index, 'pair_end'] =  crop_end
    else:
      all_games.at[index, 'pair_start'] = row['start']
      all_games.at[index, 'pair_end'] =  next_row['end']
    all_games.at[index, 'pair_diff'] = next_row['curr'] - row['curr']

all_games

from collections import Counter
pair_counter = Counter()

for index, row in all_games.iterrows():
    if row['Pair'] < 0:
        next_row = all_games.iloc[index + 1]
        next_label = next_row['label']
        pair = f"{row['label']}->{next_label}"
        pair_counter[pair] += 1
        all_games.at[index, 'Pair-label'] = pair

for pair, count in pair_counter.most_common(999):
    print(f"{pair}: {count}")

all_games[all_games['pair_end'] - all_games['pair_start'] > 8] # all clips longer than 8 seconds

all_games[(all_games['replay'] == 0) & all_games['game'].str.contains('europe_uefa-champions-league/2014-2015/2015-0')]

# all_games.to_csv('all_games.csv', index=False)