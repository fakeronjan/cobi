"""
generate_data.py — reads cobi_ratings_final.csv and writes JSON files for the web frontend.
Run after cobi.py. Outputs to docs/data/.
"""

import pandas as pd
import numpy as np
import json
import os
import re
from bisect import bisect_right
from datetime import date, datetime, timezone

os.makedirs('docs/data/teams', exist_ok=True)
os.makedirs('docs/data/seasons', exist_ok=True)

print("Reading ratings...")
df = pd.read_csv('cobi_ratings_final.csv')
df['date'] = pd.to_datetime(df['date']).dt.date
df['last_match_date'] = pd.to_datetime(df['last_match_date'], errors='coerce').dt.date
df['season'] = df['season'].astype(str)

# Snapshot season is calendar year (string). 'Y is complete' once today is past
# Dec 31 of that year — but we want intra-season views to show 1st/2nd labels
# rather than blank, while only awarding "Champion" after the trophies are
# decided. cobi.py's trophy detection is already gated on 7-day-post-final,
# so by the time a finish column says 'Champion', the trophy is real.

def clean(val):
    if pd.isna(val):
        return ''
    return str(val)


def slug(name):
    return re.sub(r'[^\w]', '_', name).strip('_')


# ── Cumulative records ───────────────────────────────────────────────────────
# Cumulative W-D-L per (team, season) for the LEAGUE (MLS or Liga MX) games.
# Cup/cross-league competitions don't count toward record. Shootout winner
# counts as W (mirrors how the ratings treat it).
print("Computing season records...")
LEAGUE_COMPETITIONS = {'MLS', 'Liga MX'}

games_raw = pd.read_csv('all_club_games.csv', parse_dates=['date'])
games_raw['home_score'] = pd.to_numeric(games_raw['home_score'], errors='coerce')
games_raw['away_score'] = pd.to_numeric(games_raw['away_score'], errors='coerce')
games_raw = games_raw.dropna(subset=['home_score', 'away_score']).copy()
# COBI uses calendar year as snapshot season label
games_raw['snap_season'] = games_raw['date'].dt.year.astype(str)

games_lg = games_raw[games_raw['competition'].isin(LEAGUE_COMPETITIONS)].copy()


def _result(home_team, home_score, away_score, shootout_winner, perspective_team):
    if home_score > away_score:
        return 'W' if perspective_team == home_team else 'L'
    if away_score > home_score:
        return 'L' if perspective_team == home_team else 'W'
    if pd.notna(shootout_winner) and shootout_winner:
        return 'W' if shootout_winner == perspective_team else 'L'
    return 'D'


# Build per-game team-perspective rows (one per team per game)
home_persp = games_lg.copy()
home_persp['team'] = home_persp['home_team']
home_persp['result'] = home_persp.apply(
    lambda r: _result(r['home_team'], r['home_score'], r['away_score'],
                      r.get('shootout_winner'), r['home_team']), axis=1)
away_persp = games_lg.copy()
away_persp['team'] = away_persp['away_team']
away_persp['result'] = away_persp.apply(
    lambda r: _result(r['home_team'], r['home_score'], r['away_score'],
                      r.get('shootout_winner'), r['away_team']), axis=1)
team_persp = pd.concat([home_persp, away_persp], ignore_index=True, sort=False)
team_persp = team_persp.sort_values(['team', 'snap_season', 'date'])

# Cumulative W-D-L string at each (team, snap_season, date)
team_persp['w'] = (team_persp['result'] == 'W').astype(int)
team_persp['d'] = (team_persp['result'] == 'D').astype(int)
team_persp['l'] = (team_persp['result'] == 'L').astype(int)
team_persp['cum_w'] = team_persp.groupby(['team', 'snap_season'])['w'].cumsum()
team_persp['cum_d'] = team_persp.groupby(['team', 'snap_season'])['d'].cumsum()
team_persp['cum_l'] = team_persp.groupby(['team', 'snap_season'])['l'].cumsum()
team_persp['record'] = (
    team_persp['cum_w'].astype(str) + '-' +
    team_persp['cum_d'].astype(str) + '-' +
    team_persp['cum_l'].astype(str)
)

# Per-(team, season) sorted history for as-of lookups
_rec_hist = {}
for (team, season), grp in team_persp.groupby(['team', 'snap_season']):
    grp = grp.sort_values('date')
    dates = [str(d.date()) for d in grp['date']]
    recs  = list(grp['record'])
    _rec_hist[(team, season)] = (dates, recs)


def record_as_of(team, season, snap_date_str):
    entry = _rec_hist.get((team, season))
    if not entry:
        return '0-0-0'
    dates, recs = entry
    idx = bisect_right(dates, snap_date_str) - 1
    return recs[idx] if idx >= 0 else '0-0-0'


# ── End-of-season detection ──────────────────────────────────────────────────
# A snapshot is the end-of-season for calendar year Y if it's the last snapshot
# whose date falls in year Y AND that year has a "complete" trophy decided
# (i.e., MLS Cup awarded for Y, or Liga MX Apertura champion of Y, etc.).
# For COBI we use a simpler signal: the last snapshot in calendar year Y is
# the EOS snapshot. Years still in progress get no EOS row.
print("Tagging end-of-season snapshots...")
df['date_dt'] = pd.to_datetime(df['date'])
season_last_snap = (
    df.groupby('season')['ranking_id']
    .max()
    .to_dict()
)
df['is_end_of_season'] = df.apply(
    lambda r: 1 if r['ranking_id'] == season_last_snap.get(r['season']) else 0,
    axis=1
)
df.drop(columns=['date_dt'], inplace=True)

# Per-team league-rank within each snapshot (rank within MLS, rank within Liga MX)
df['lg_rank'] = (
    df.groupby(['ranking_id', 'league'])['rating']
    .rank(ascending=False, method='min')
    .astype(int)
)

# Current snapshot + records (after lg_rank exists)
latest_id = int(df['ranking_id'].max())
latest = df[df['ranking_id'] == latest_id].sort_values('rank').copy()
latest_date_str = str(latest['date'].iloc[0])
cur_records = {
    r['team']: record_as_of(r['team'], r['season'], latest_date_str)
    for _, r in latest.iterrows()
}


# ── 1. Current standings ─────────────────────────────────────────────────────
print("Writing current_standings.json...")
standings_data = {
    'updated': latest_date_str,
    'teams': [
        {
            'rank':                int(r['rank']),
            'lg_rank':             int(r['lg_rank']),
            'team':                r['team'],
            'league':              clean(r['league']),
            'rating':              round(float(r['rating']), 3),
            'record':              cur_records.get(r['team'], '0-0-0'),
            'games_played':        int(r['games_played']),
            'last_match':          clean(r['last_match']),
            'last_match_date':     clean(r['last_match_date']),
            'mls_cup_finish':      clean(r.get('mls_cup_finish', '')),
            'apertura_finish':     clean(r.get('liga_mx_apertura_finish', '')),
            'clausura_finish':     clean(r.get('liga_mx_clausura_finish', '')),
            'ccl_finish':          clean(r.get('ccl_finish', '')),
            'leagues_cup_finish':  clean(r.get('leagues_cup_finish', '')),
        }
        for _, r in latest.iterrows()
    ],
}
with open('docs/data/current_standings.json', 'w') as f:
    json.dump(standings_data, f, separators=(',', ':'))


# ── 2. Champions table ───────────────────────────────────────────────────────
# Rich per-trophy view modeled on ZIDANE's champions.json: dict keyed by
# trophy label, each value a list of {season, champion: {…full team data…},
# runner_up: {…}, final_score} entries. Frontend renders this as a head-to-
# head table comparing champion vs runner-up at end of season.
print("Writing champions.json...")
trophies = pd.read_csv('cobi_trophies.csv')

# End-of-year snapshot per (team, season) — supplies rating/rank/lg_rank/record
eoy_lookup = {}
for (team, season), grp in df.groupby(['team', 'season']):
    last = grp.sort_values('date').iloc[-1]
    eoy_lookup[(team, str(season))] = {
        'team':                 team,
        'league':               clean(last['league']),
        'rating':               round(float(last['rating']), 3),
        'rank':                 int(last['rank']),
        'lg_rank':              int(last['lg_rank']),
        'record':               record_as_of(team, str(season), str(last['date'])),
        'mls_cup_finish':       clean(last.get('mls_cup_finish', '')),
        'apertura_finish':      clean(last.get('liga_mx_apertura_finish', '')),
        'clausura_finish':      clean(last.get('liga_mx_clausura_finish', '')),
        'ccl_finish':           clean(last.get('ccl_finish', '')),
        'leagues_cup_finish':   clean(last.get('leagues_cup_finish', '')),
    }

# Final-score lookup — per (year, trophy_label, champ, runner_up) figures
# the score of the actual championship match. Single-game trophies use the
# raw score of the final; 2-leg trophies use the aggregate.
def _score_for_trophy(year, trophy_label, champ, runner_up):
    y = int(year)
    # Filter master games to the relevant competition + year + (if Liga MX) half
    g = games_raw.copy()
    g['year'] = g['date'].dt.year
    if trophy_label == 'MLS Cup':
        sub = g[(g['competition'] == 'MLS') & (g['year'] == y)]
    elif trophy_label == 'Liga MX Apertura':
        sub = g[(g['competition'] == 'Liga MX') & (g['year'] == y) & (g['date'].dt.month >= 7)]
    elif trophy_label == 'Liga MX Clausura':
        sub = g[(g['competition'] == 'Liga MX') & (g['year'] == y) & (g['date'].dt.month < 7)]
    elif trophy_label == 'CONCACAF CL':
        sub = g[(g['competition'] == 'CONCACAF CL') & (g['year'] == y)]
    elif trophy_label == 'Leagues Cup':
        sub = g[(g['competition'] == 'Leagues Cup') & (g['year'] == y)]
    else:
        return ''
    pair = sub[
        ((sub['home_team'] == champ) & (sub['away_team'] == runner_up)) |
        ((sub['home_team'] == runner_up) & (sub['away_team'] == champ))
    ].sort_values('date')
    if pair.empty:
        return ''
    is_2leg = trophy_label in ('Liga MX Apertura', 'Liga MX Clausura') or \
              (trophy_label == 'CONCACAF CL' and y < 2024)
    if is_2leg and len(pair) >= 2:
        legs = pair.tail(2)
        c_goals = sum(int(r['home_score']) if r['home_team'] == champ else int(r['away_score'])
                      for _, r in legs.iterrows())
        r_goals = sum(int(r['home_score']) if r['home_team'] == runner_up else int(r['away_score'])
                      for _, r in legs.iterrows())
        score = f'{c_goals}-{r_goals}'
        if c_goals == r_goals:
            sw = legs.iloc[-1].get('shootout_winner')
            if pd.notna(sw) and str(sw).strip():
                score += ' (pen)'
        return score
    last = pair.iloc[-1]
    if last['home_team'] == champ:
        c_score, r_score = int(last['home_score']), int(last['away_score'])
    else:
        c_score, r_score = int(last['away_score']), int(last['home_score'])
    score = f'{c_score}-{r_score}'
    if c_score == r_score:
        sw = last.get('shootout_winner')
        if pd.notna(sw) and str(sw).strip():
            score += ' (pen)'
    return score

HONOR_KEY = {
    'MLS Cup':           'MLS Cup',
    'Liga MX Apertura':  'Liga MX Apertura',
    'Liga MX Clausura':  'Liga MX Clausura',
    'CONCACAF CL':       'CONCACAF CL',
    'Leagues Cup':       'Leagues Cup',
}
champions_by_trophy = {label: [] for label in HONOR_KEY}

for (year, label), grp in trophies.groupby(['year', trophies['honor'].str.replace(' Champion', '', regex=False).str.replace(' Runner-Up', '', regex=False)]):
    champ_row = grp[grp['honor'].str.endswith('Champion')]
    ru_row    = grp[grp['honor'].str.endswith('Runner-Up')]
    if champ_row.empty or ru_row.empty:
        continue
    champ_team = champ_row.iloc[0]['team']
    ru_team    = ru_row.iloc[0]['team']
    champion  = eoy_lookup.get((champ_team, str(year)))
    runner_up = eoy_lookup.get((ru_team, str(year)))
    if champion is None or runner_up is None:
        # Team didn't pass min_games filter — fall back to a minimal record
        if champion is None:
            champion = {'team': champ_team, 'league': '', 'rating': None, 'rank': None,
                        'lg_rank': None, 'record': '—', 'mls_cup_finish': '',
                        'apertura_finish': '', 'clausura_finish': '',
                        'ccl_finish': '', 'leagues_cup_finish': ''}
        if runner_up is None:
            runner_up = {'team': ru_team, 'league': '', 'rating': None, 'rank': None,
                         'lg_rank': None, 'record': '—', 'mls_cup_finish': '',
                         'apertura_finish': '', 'clausura_finish': '',
                         'ccl_finish': '', 'leagues_cup_finish': ''}
    champions_by_trophy[label].append({
        'season':      str(year),
        'champion':    champion,
        'runner_up':   runner_up,
        'final_score': _score_for_trophy(year, label, champ_team, ru_team),
    })

# Sort each trophy's entries by season (newest first, like ZIDANE)
for label in champions_by_trophy:
    champions_by_trophy[label].sort(key=lambda e: e['season'], reverse=True)

with open('docs/data/champions.json', 'w') as f:
    json.dump(champions_by_trophy, f, separators=(',', ':'))


# ── 3. GOAT table ────────────────────────────────────────────────────────────
# End-of-season top-rated teams that won a trophy that year (MLS Cup champion,
# Liga MX Apertura/Clausura champion, CCL champion, or Leagues Cup champion).
# Drops in-progress current year automatically since uncrowned years won't be
# included.
print("Writing goat_teams.json...")
eos = df[df['is_end_of_season'] == 1].copy()
won_a_trophy = (
    (eos.get('mls_cup_finish', '') == 'Champion') |
    (eos.get('liga_mx_apertura_finish', '') == 'Champion') |
    (eos.get('liga_mx_clausura_finish', '') == 'Champion') |
    (eos.get('ccl_finish', '') == 'Champion') |
    (eos.get('leagues_cup_finish', '') == 'Champion')
)
eos = eos[won_a_trophy]
eos = eos.sort_values('rating', ascending=False).head(50).reset_index(drop=True)

# End-of-season record per (team, season)
final_record_lookup = {}
for (t, s), grp in team_persp.groupby(['team', 'snap_season']):
    final_record_lookup[(t, s)] = grp.sort_values('date').iloc[-1]['record']

goat_data = [
    {
        'rank':                i + 1,
        'team':                r['team'],
        'season':              r['season'],
        'league':              clean(r['league']),
        'rating':              round(float(r['rating']), 3),
        'record':              final_record_lookup.get((r['team'], r['season']), '—'),
        'mls_cup_finish':      clean(r.get('mls_cup_finish', '')),
        'apertura_finish':     clean(r.get('liga_mx_apertura_finish', '')),
        'clausura_finish':     clean(r.get('liga_mx_clausura_finish', '')),
        'ccl_finish':          clean(r.get('ccl_finish', '')),
        'leagues_cup_finish':  clean(r.get('leagues_cup_finish', '')),
    }
    for i, (_, r) in enumerate(eos.iterrows())
]
with open('docs/data/goat_teams.json', 'w') as f:
    json.dump(goat_data, f, separators=(',', ':'))


# ── 4. Per-team JSON files ───────────────────────────────────────────────────
print("Writing per-team JSON files...")
game_days = df[(df['is_game_day'] == 1) | (df['is_end_of_season'] == 1)].copy()
game_days = game_days.sort_values(['team', 'season', 'date'])

all_teams = sorted(df['team'].unique())
teams_index = []

for team in all_teams:
    tdf = game_days[game_days['team'] == team]
    if len(tdf) == 0:
        continue
    league = clean(df[df['team'] == team]['league'].iloc[-1])
    team_slug = slug(team)
    teams_index.append({'name': team, 'league': league, 'slug': team_slug})

    seasons = {}
    for season, sdf in tdf.groupby('season'):
        seasons[str(season)] = [
            {
                'date':              str(r['date']),
                'rating':            round(float(r['rating']), 3),
                'rank':              int(r['rank']),
                'lg_rank':           int(r['lg_rank']),
                'is_end_of_season':  int(r['is_end_of_season']),
                'record':            record_as_of(team, str(season), str(r['date'])),
                'last_match':        clean(r['last_match']),
                'mls_cup_finish':    clean(r.get('mls_cup_finish', '')),
                'apertura_finish':   clean(r.get('liga_mx_apertura_finish', '')),
                'clausura_finish':   clean(r.get('liga_mx_clausura_finish', '')),
                'ccl_finish':        clean(r.get('ccl_finish', '')),
                'leagues_cup_finish': clean(r.get('leagues_cup_finish', '')),
            }
            for _, r in sdf.sort_values('date').iterrows()
        ]

    with open(f'docs/data/teams/{team_slug}.json', 'w') as f:
        json.dump({'team': team, 'league': league, 'seasons': seasons},
                  f, separators=(',', ':'))

teams_index.sort(key=lambda x: (x['league'], x['name']))
with open('docs/data/teams_index.json', 'w') as f:
    json.dump(teams_index, f, separators=(',', ':'))


# ── 5. Per-season JSON files ─────────────────────────────────────────────────
print("Writing per-season JSON files...")
all_seasons = sorted(df['season'].unique(), reverse=True)  # latest first

for season in all_seasons:
    sdf = df[df['season'] == season].copy()
    sdf = sdf.sort_values(['date', 'rank'])

    snaps = []
    for snap_date, gdf in sdf.groupby('date'):
        gdf = gdf.sort_values('rank')
        teams = []
        for _, r in gdf.iterrows():
            teams.append({
                'rank':              int(r['rank']),
                'lg_rank':           int(r['lg_rank']),
                'team':              r['team'],
                'league':            clean(r['league']),
                'rating':            round(float(r['rating']), 3),
                'record':            record_as_of(r['team'], season, str(snap_date)),
                'last_match':        clean(r['last_match']),
                'last_match_date':   clean(r['last_match_date']),
                'mls_cup_finish':    clean(r.get('mls_cup_finish', '')),
                'apertura_finish':   clean(r.get('liga_mx_apertura_finish', '')),
                'clausura_finish':   clean(r.get('liga_mx_clausura_finish', '')),
                'ccl_finish':        clean(r.get('ccl_finish', '')),
                'leagues_cup_finish': clean(r.get('leagues_cup_finish', '')),
            })
        snaps.append({
            'date':              str(snap_date),
            'label':             None,
            'is_end_of_season':  int(gdf['is_end_of_season'].max()),
            'teams':             teams,
        })

    with open(f'docs/data/seasons/{season}.json', 'w') as f:
        json.dump({'season': season, 'snapshots': snaps},
                  f, separators=(',', ':'))

# seasons_index.json — format expected by frontend
df_dates = pd.to_datetime(df['date'])
seasons_index = {
    'seasons':      all_seasons,
    'first_date':   str(df_dates.min().date()),
    'last_date':    str(df_dates.max().date()),
    'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
}
with open('docs/data/seasons_index.json', 'w') as f:
    json.dump(seasons_index, f, separators=(',', ':'))


# ── 6. Updated metadata ──────────────────────────────────────────────────────
meta = {
    'last_refreshed_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    'latest_snapshot_date': latest_date_str,
    'n_teams':   len(all_teams),
    'n_seasons': len(all_seasons),
}
with open('docs/data/meta.json', 'w') as f:
    json.dump(meta, f, separators=(',', ':'))

print(f"\n[done] {len(all_teams)} teams, {len(all_seasons)} seasons. JSON written to docs/data/")
