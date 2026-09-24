"""One-off: fill all_club_games.csv `stage` (ESPN season slug) for 2019+.

ESPN's scoreboard only takes single dates, so this walks every date that has
a stored 2019+ game, maps event_id -> slug, and writes the column back.
Idempotent; rerun only if older rows are missing stage.
"""
import time
import pandas as pd
from cobi import espn_fetch_day

FIRST_SEASON = 2019

df = pd.read_csv('all_club_games.csv', dtype={'event_id': str})
if 'stage' not in df.columns:
    df['stage'] = ''
df['stage'] = df['stage'].fillna('')
need = df[(df['season'] >= FIRST_SEASON) & (df['stage'] == '')]
dates = sorted(set(pd.to_datetime(need['date']).dt.date))
# ESPN files late kickoffs under the next UTC day in some cases, so also
# query the day after each match date.
qdates = sorted(set(dates) | {d + pd.Timedelta(days=1) for d in dates})
print(f"{len(need):,} rows missing stage across {len(dates)} dates ({len(qdates)} queries)")
slug_by_event = {}
for i, d in enumerate(qdates):
    for ev in espn_fetch_day('usa.1', d).get('events', []) or []:
        slug_by_event[str(ev.get('id'))] = (ev.get('season') or {}).get('slug') or ''
    if i % 100 == 0:
        print(f"  {i}/{len(qdates)}")
mask = (df['season'] >= FIRST_SEASON) & (df['stage'] == '')
df.loc[mask, 'stage'] = df.loc[mask, 'event_id'].map(slug_by_event).fillna('')
left = ((df['season'] >= FIRST_SEASON) & (df['stage'] == '')).sum()
print(f"filled; {left} rows still missing stage")
df.to_csv('all_club_games.csv', index=False)
