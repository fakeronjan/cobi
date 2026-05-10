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

# ── MLS Eastern/Western Conference history ──────────────────────────────────
# Per-(team, year) conference assignment. Each entry is a list of inclusive
# (start_year, end_year, conf) ranges; 9999 means "ongoing". Historical aliases
# (Chicago Fire/Chicago Fire FC, Columbus Crew/Columbus Crew SC) are mapped to
# the same franchise lineage.
MLS_CONFERENCE_HISTORY = {
    'Atlanta United FC':         [(2017, 9999, 'East')],
    'Austin FC':                 [(2021, 9999, 'West')],
    'CF Montréal':               [(2012, 9999, 'East')],
    'Charlotte FC':              [(2022, 9999, 'East')],
    'Chicago Fire':              [(1998, 2001, 'West'), (2002, 9999, 'East')],
    'Chicago Fire FC':           [(1998, 2001, 'West'), (2002, 9999, 'East')],
    'Chivas USA':                [(2005, 2014, 'West')],
    'Colorado Rapids':           [(1996, 9999, 'West')],
    'Columbus Crew':             [(1996, 9999, 'East')],
    'Columbus Crew SC':          [(1996, 9999, 'East')],
    'D.C. United':               [(1996, 9999, 'East')],
    'FC Cincinnati':             [(2019, 9999, 'East')],
    'FC Dallas':                 [(1996, 9999, 'West')],
    'Houston Dynamo FC':         [(2006, 9999, 'West')],
    'Inter Miami CF':            [(2020, 9999, 'East')],
    'LA Galaxy':                 [(1996, 9999, 'West')],
    'LAFC':                      [(2018, 9999, 'West')],
    'Miami Fusion':              [(1998, 2001, 'East')],
    'Minnesota United FC':       [(2017, 9999, 'West')],
    'Nashville SC':              [(2020, 2020, 'East'), (2021, 2023, 'West'), (2024, 9999, 'East')],
    'New England Revolution':    [(1996, 9999, 'East')],
    'New York City FC':          [(2015, 9999, 'East')],
    'Orlando City SC':           [(2015, 9999, 'East')],
    'Philadelphia Union':        [(2010, 9999, 'East')],
    'Portland Timbers':          [(2011, 9999, 'West')],
    'Real Salt Lake':            [(2005, 9999, 'West')],
    'Red Bull New York':         [(1996, 9999, 'East')],
    'San Diego FC':              [(2025, 9999, 'West')],
    'San Jose Earthquakes':      [(1996, 9999, 'West')],
    'Seattle Sounders FC':       [(2009, 9999, 'West')],
    'Sporting Kansas City':      [(1996, 2010, 'East'), (2011, 9999, 'West')],
    'St. Louis CITY SC':         [(2023, 9999, 'West')],
    'Tampa Bay Mutiny':          [(1996, 2001, 'East')],
    'Toronto FC':                [(2007, 9999, 'East')],
    'Vancouver Whitecaps':       [(2011, 9999, 'West')],
}


def conference_for(team, year):
    """Conference as of the given calendar year. Returns '' if unmapped."""
    history = MLS_CONFERENCE_HISTORY.get(team)
    if not history:
        return ''
    y = int(year)
    for start, end, conf in history:
        if start <= y <= end:
            return conf
    return ''


def current_conference(team):
    """Most recent conference assignment (used for team-index grouping)."""
    history = MLS_CONFERENCE_HISTORY.get(team)
    if not history:
        return ''
    return history[-1][2]


print("Reading ratings...")
df = pd.read_csv('cobi_ratings_final.csv')
df['date'] = pd.to_datetime(df['date']).dt.date
df['last_match_date'] = pd.to_datetime(df['last_match_date'], errors='coerce').dt.date
df['season'] = df['season'].astype(str)
df['conference'] = df.apply(lambda r: conference_for(r['team'], r['season']), axis=1)


def clean(val):
    if pd.isna(val):
        return ''
    return str(val)


def slug(name):
    return re.sub(r'[^\w]', '_', name).strip('_')


# ── Cumulative records ───────────────────────────────────────────────────────
# Cumulative W-D-L per (team, season) for MLS games. Shootout winner counts
# as W (mirrors how the ratings treat it).
print("Computing season records...")

games_raw = pd.read_csv('all_club_games.csv', parse_dates=['date'])
games_raw['home_score'] = pd.to_numeric(games_raw['home_score'], errors='coerce')
games_raw['away_score'] = pd.to_numeric(games_raw['away_score'], errors='coerce')
games_raw = games_raw.dropna(subset=['home_score', 'away_score']).copy()
games_raw['snap_season'] = games_raw['date'].dt.year.astype(str)

games_lg = games_raw[games_raw['competition'] == 'MLS'].copy()


# ── Regular season vs playoff partition ─────────────────────────────────────
# Decision Day = last date in Sept-Nov of each calendar year with >= 6 MLS
# games (the simultaneous final-weekend pattern). Anything after = playoffs.
# Mirrors cobi.py's Shield-detection heuristic.
def _decision_day(year_games):
    if year_games.empty:
        return None
    late = year_games[
        (year_games['date'].dt.month >= 9) &
        (year_games['date'].dt.month <= 11)
    ]
    if late.empty:
        return year_games['date'].dt.date.max()
    daily = late.groupby(late['date'].dt.date).size()
    big = daily[daily >= 6]
    return big.index.max() if not big.empty else daily.idxmax()


_decision_day_by_year = {}
for y, sg in games_lg.groupby(games_lg['date'].dt.year):
    dd = _decision_day(sg)
    if dd is not None:
        _decision_day_by_year[int(y)] = dd

# is_playoff flag per game (date > Decision Day)
def _is_playoff_row(row):
    dd = _decision_day_by_year.get(row['date'].year)
    return dd is not None and row['date'].date() > dd

games_lg['is_playoff'] = games_lg.apply(_is_playoff_row, axis=1)


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

# Regular-season-only counts (for the W-D-L / Pts column)
reg = team_persp[~team_persp['is_playoff']].copy()
reg['w'] = (reg['result'] == 'W').astype(int)
reg['d'] = (reg['result'] == 'D').astype(int)
reg['l'] = (reg['result'] == 'L').astype(int)
reg['cum_w'] = reg.groupby(['team', 'snap_season'])['w'].cumsum()
reg['cum_d'] = reg.groupby(['team', 'snap_season'])['d'].cumsum()
reg['cum_l'] = reg.groupby(['team', 'snap_season'])['l'].cumsum()
reg['record'] = (
    reg['cum_w'].astype(str) + '-' +
    reg['cum_d'].astype(str) + '-' +
    reg['cum_l'].astype(str)
)

# Playoff-only counts (W-L; draws shouldn't appear since shootouts decide
# every MLS playoff game, but kept robust in case of historical edge cases).
po = team_persp[team_persp['is_playoff']].copy()
po['w'] = (po['result'] == 'W').astype(int)
po['l'] = (po['result'] == 'L').astype(int)
po['cum_w'] = po.groupby(['team', 'snap_season'])['w'].cumsum()
po['cum_l'] = po.groupby(['team', 'snap_season'])['l'].cumsum()
po['record'] = po['cum_w'].astype(str) + '-' + po['cum_l'].astype(str)

_reg_hist = {}
for (team, season), grp in reg.groupby(['team', 'snap_season']):
    grp = grp.sort_values('date')
    dates = [str(d.date()) for d in grp['date']]
    recs  = list(grp['record'])
    _reg_hist[(team, season)] = (dates, recs)

_po_hist = {}
for (team, season), grp in po.groupby(['team', 'snap_season']):
    grp = grp.sort_values('date')
    dates = [str(d.date()) for d in grp['date']]
    recs  = list(grp['record'])
    _po_hist[(team, season)] = (dates, recs)


def regular_record_as_of(team, season, snap_date_str):
    entry = _reg_hist.get((team, season))
    if not entry:
        return '0-0-0'
    dates, recs = entry
    idx = bisect_right(dates, snap_date_str) - 1
    return recs[idx] if idx >= 0 else '0-0-0'


def playoff_record_as_of(team, season, snap_date_str):
    entry = _po_hist.get((team, season))
    if not entry:
        return ''
    dates, recs = entry
    idx = bisect_right(dates, snap_date_str) - 1
    return recs[idx] if idx >= 0 else ''


# ── End-of-season detection ──────────────────────────────────────────────────
# A snapshot is the EOS for calendar year Y if it's the last ranking_id whose
# date falls in year Y. cobi.py's trophy detection is gated on 7-day-post-final
# so by the time mls_cup_finish says 'Champion', the trophy is real.
print("Tagging end-of-season snapshots...")
season_last_snap = (
    df.groupby('season')['ranking_id']
    .max()
    .to_dict()
)
df['is_end_of_season'] = df.apply(
    lambda r: 1 if r['ranking_id'] == season_last_snap.get(r['season']) else 0,
    axis=1
)

# Per-team conference rank within each snapshot (rank within East / West)
df['conf_rank'] = (
    df.groupby(['ranking_id', 'conference'])['rating']
    .rank(ascending=False, method='min')
    .astype(int)
)

latest_id = int(df['ranking_id'].max())
latest = df[df['ranking_id'] == latest_id].sort_values('rank').copy()
latest_date_str = str(latest['date'].iloc[0])
cur_reg = {
    r['team']: regular_record_as_of(r['team'], r['season'], latest_date_str)
    for _, r in latest.iterrows()
}
cur_po = {
    r['team']: playoff_record_as_of(r['team'], r['season'], latest_date_str)
    for _, r in latest.iterrows()
}


# ── 1. Current standings ─────────────────────────────────────────────────────
print("Writing current_standings.json...")
standings_data = {
    'updated': latest_date_str,
    'teams': [
        {
            'rank':                int(r['rank']),
            'conf_rank':           int(r['conf_rank']),
            'team':                r['team'],
            'conference':          clean(r['conference']),
            'rating':              round(float(r['rating']), 3),
            'regular_record':      cur_reg.get(r['team'], '0-0-0'),
            'playoff_record':      cur_po.get(r['team'], ''),
            'games_played':        int(r['games_played']),
            'last_match':          clean(r['last_match']),
            'last_match_date':     clean(r['last_match_date']),
            'mls_cup_finish':            clean(r.get('mls_cup_finish', '')),
            'supporters_shield_finish':  clean(r.get('supporters_shield_finish', '')),
        }
        for _, r in latest.iterrows()
    ],
}
with open('docs/data/current_standings.json', 'w') as f:
    json.dump(standings_data, f, separators=(',', ':'))


# ── 2. Champions table ───────────────────────────────────────────────────────
# MLS Cup head-to-head: champion vs runner-up at end of season, with the
# final's score. Schema mirrors ZIDANE's champions.json but only one trophy.
print("Writing champions.json...")
trophies = pd.read_csv('cobi_trophies.csv')

eoy_lookup = {}
for (team, season), grp in df.groupby(['team', 'season']):
    last = grp.sort_values('date').iloc[-1]
    eoy_lookup[(team, str(season))] = {
        'team':                       team,
        'conference':                 clean(last['conference']),
        'rating':                     round(float(last['rating']), 3),
        'rank':                       int(last['rank']),
        'conf_rank':                  int(last['conf_rank']),
        'regular_record':             regular_record_as_of(team, str(season), str(last['date'])),
        'playoff_record':             playoff_record_as_of(team, str(season), str(last['date'])),
        'mls_cup_finish':             clean(last.get('mls_cup_finish', '')),
        'supporters_shield_finish':   clean(last.get('supporters_shield_finish', '')),
    }


def _mls_cup_score(year, champ, runner_up):
    """MLS Cup is a single-game final played late Nov/Dec. Return 'C-R' or
    'C-R (pen)' if it went to a shootout."""
    y = int(year)
    g = games_raw.copy()
    g['year'] = g['date'].dt.year
    sub = g[(g['competition'] == 'MLS') & (g['year'] == y)]
    pair = sub[
        ((sub['home_team'] == champ) & (sub['away_team'] == runner_up)) |
        ((sub['home_team'] == runner_up) & (sub['away_team'] == champ))
    ].sort_values('date')
    if pair.empty:
        return ''
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


def _shield_score(year, champ, runner_up):
    """Shield 'score' = champion's regular-season points vs runner-up's.
    Mirrors cobi.py's Shield detection logic — Decision Day filter, MLS games
    only, 3 for win incl. shootout, 1 for draw."""
    y = int(year)
    g = games_raw.copy()
    g['year'] = g['date'].dt.year
    mls_y = g[(g['competition'] == 'MLS') & (g['year'] == y)]
    if mls_y.empty:
        return ''
    late = mls_y[(mls_y['date'].dt.month >= 9) & (mls_y['date'].dt.month <= 11)]
    if late.empty:
        ds_date = mls_y['date'].dt.date.max()
    else:
        daily = late.groupby(late['date'].dt.date).size()
        big = daily[daily >= 6]
        ds_date = big.index.max() if not big.empty else daily.idxmax()
    reg = mls_y[mls_y['date'].dt.date <= ds_date]

    def _team_pts(t):
        pts = 0
        for _, r in reg.iterrows():
            if r['home_team'] == t:
                gf, ga = int(r['home_score']), int(r['away_score'])
            elif r['away_team'] == t:
                gf, ga = int(r['away_score']), int(r['home_score'])
            else:
                continue
            sw = r.get('shootout_winner')
            if gf > ga or (gf == ga and pd.notna(sw) and sw == t):
                pts += 3
            elif gf == ga and not (pd.notna(sw) and sw):
                pts += 1
        return pts

    return f'{_team_pts(champ)}-{_team_pts(runner_up)} pts'


CHAMPIONS_TROPHIES = [
    ('MLS Cup',           _mls_cup_score),
    ('Supporters Shield', _shield_score),
]

champions_by_trophy = {label: [] for label, _ in CHAMPIONS_TROPHIES}

for (year, label), grp in trophies.groupby(['year', trophies['honor'].str.replace(' Champion', '', regex=False).str.replace(' Runner-Up', '', regex=False)]):
    if label not in champions_by_trophy:
        continue
    champ_row = grp[grp['honor'].str.endswith('Champion')]
    ru_row    = grp[grp['honor'].str.endswith('Runner-Up')]
    if champ_row.empty or ru_row.empty:
        continue
    champ_team = champ_row.iloc[0]['team']
    ru_team    = ru_row.iloc[0]['team']
    champion  = eoy_lookup.get((champ_team, str(year)))
    runner_up = eoy_lookup.get((ru_team, str(year)))
    if champion is None:
        champion = {'team': champ_team, 'conference': '', 'rating': None,
                    'rank': None, 'conf_rank': None, 'regular_record': '0-0-0', 'playoff_record': '',
                    'mls_cup_finish': '', 'supporters_shield_finish': ''}
    if runner_up is None:
        runner_up = {'team': ru_team, 'conference': '', 'rating': None,
                     'rank': None, 'conf_rank': None, 'regular_record': '0-0-0', 'playoff_record': '',
                     'mls_cup_finish': '', 'supporters_shield_finish': ''}
    score_fn = dict(CHAMPIONS_TROPHIES)[label]
    champions_by_trophy[label].append({
        'season':      str(year),
        'champion':    champion,
        'runner_up':   runner_up,
        'final_score': score_fn(year, champ_team, ru_team),
    })

for label in champions_by_trophy:
    champions_by_trophy[label].sort(key=lambda e: e['season'], reverse=True)

with open('docs/data/champions.json', 'w') as f:
    json.dump(champions_by_trophy, f, separators=(',', ':'))


# ── 3. GOAT table ────────────────────────────────────────────────────────────
# End-of-season top-rated MLS Cup winners. Drops in-progress current year
# automatically since uncrowned years won't be included.
print("Writing goat_teams.json...")
eos = df[df['is_end_of_season'] == 1].copy()
won_a_trophy = (
    (eos.get('mls_cup_finish', '') == 'Champion') |
    (eos.get('supporters_shield_finish', '') == 'Champion')
)
eos = eos[won_a_trophy]
eos = eos.sort_values('rating', ascending=False).head(50).reset_index(drop=True)

final_reg_lookup = {(t, s): grp.sort_values('date').iloc[-1]['record']
                    for (t, s), grp in reg.groupby(['team', 'snap_season'])}
final_po_lookup  = {(t, s): grp.sort_values('date').iloc[-1]['record']
                    for (t, s), grp in po.groupby(['team', 'snap_season'])}

goat_data = [
    {
        'rank':                       i + 1,
        'team':                       r['team'],
        'season':                     r['season'],
        'conference':                 clean(r['conference']),
        'rating':                     round(float(r['rating']), 3),
        'regular_record':             final_reg_lookup.get((r['team'], r['season']), '0-0-0'),
        'playoff_record':             final_po_lookup.get((r['team'], r['season']), ''),
        'mls_cup_finish':             clean(r.get('mls_cup_finish', '')),
        'supporters_shield_finish':   clean(r.get('supporters_shield_finish', '')),
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
    conference = current_conference(team)
    team_slug = slug(team)
    teams_index.append({'name': team, 'conference': conference, 'slug': team_slug})

    seasons = {}
    for season, sdf in tdf.groupby('season'):
        seasons[str(season)] = [
            {
                'date':              str(r['date']),
                'rating':            round(float(r['rating']), 3),
                'rank':              int(r['rank']),
                'conf_rank':         int(r['conf_rank']),
                'is_end_of_season':  int(r['is_end_of_season']),
                'regular_record':    regular_record_as_of(team, str(season), str(r['date'])),
                'playoff_record':    playoff_record_as_of(team, str(season), str(r['date'])),
                'last_match':        clean(r['last_match']),
                'mls_cup_finish':            clean(r.get('mls_cup_finish', '')),
                'supporters_shield_finish':  clean(r.get('supporters_shield_finish', '')),
            }
            for _, r in sdf.sort_values('date').iterrows()
        ]

    with open(f'docs/data/teams/{team_slug}.json', 'w') as f:
        json.dump({'team': team, 'conference': conference, 'seasons': seasons},
                  f, separators=(',', ':'))

teams_index.sort(key=lambda x: (x['conference'], x['name']))
with open('docs/data/teams_index.json', 'w') as f:
    json.dump(teams_index, f, separators=(',', ':'))


# ── 5. Per-season JSON files ─────────────────────────────────────────────────
print("Writing per-season JSON files...")
all_seasons = sorted(df['season'].unique(), reverse=True)

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
                'conf_rank':         int(r['conf_rank']),
                'team':              r['team'],
                'conference':        clean(r['conference']),
                'rating':            round(float(r['rating']), 3),
                'regular_record':    regular_record_as_of(r['team'], season, str(snap_date)),
                'playoff_record':    playoff_record_as_of(r['team'], season, str(snap_date)),
                'last_match':        clean(r['last_match']),
                'last_match_date':   clean(r['last_match_date']),
                'mls_cup_finish':            clean(r.get('mls_cup_finish', '')),
                'supporters_shield_finish':  clean(r.get('supporters_shield_finish', '')),
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
