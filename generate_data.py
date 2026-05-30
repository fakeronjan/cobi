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

# ── Same-franchise rebrand aliases (must match cobi.py) ─────────────────────
# Defensive layer in case generate_data.py runs against a CSV that didn't
# go through cobi.py's normalization. Keep in sync with cobi.py's map.
MLS_TEAM_ALIASES = {
    'Chicago Fire':                   'Chicago Fire FC',
    'Columbus Crew SC':               'Columbus Crew',
    'NY/NJ MetroStars':               'Red Bull New York',
    'MetroStars':                     'Red Bull New York',
    'New York Red Bulls':             'Red Bull New York',
    'Kansas City Wiz':                'Sporting Kansas City',
    'Kansas City Wizards':            'Sporting Kansas City',
    'Dallas Burn':                    'FC Dallas',
    'Los Angeles Galaxy':             'LA Galaxy',
    'San Jose Clash':                 'San Jose Earthquakes',
    'Montreal Impact':                'CF Montréal',
}


# ── Shootout-era (1996-1999) regular-season final standings ─────────────────
# MLS used a no-draws shootout format from 1996-1999: regulation W = 3 pts,
# shootout W = 1 pt, L = 0 pts. Our gap-fill source (footballcsv) folded
# shootout outcomes into regulation results — meaning we cannot recover the
# era-accurate W vs SOW split from the games CSV. Hard-code season-final
# standings from Wikipedia so end-of-season records and Shield winners are
# era-correct. Verified Pts = 3*W + SOW for every entry.
# Source: https://en.wikipedia.org/wiki/<year>_Major_League_Soccer_season
# Format: (year, canonical_team) → {w, sow, l, gf, ga, pts}
# Keep in sync with cobi.py.
MLS_EARLY_STANDINGS = {
    # 1996 ─────────────────────────────────────────────────────────────
    (1996, 'Tampa Bay Mutiny'):       {'w': 19, 'sow': 1, 'l': 12, 'gf': 66, 'ga': 51, 'pts': 58},
    (1996, 'D.C. United'):            {'w': 15, 'sow': 1, 'l': 16, 'gf': 62, 'ga': 56, 'pts': 46},
    (1996, 'Red Bull New York'):      {'w': 12, 'sow': 3, 'l': 17, 'gf': 45, 'ga': 47, 'pts': 39},
    (1996, 'Columbus Crew'):          {'w': 11, 'sow': 4, 'l': 17, 'gf': 59, 'ga': 60, 'pts': 37},
    (1996, 'New England Revolution'): {'w':  9, 'sow': 6, 'l': 17, 'gf': 43, 'ga': 56, 'pts': 33},
    (1996, 'LA Galaxy'):              {'w': 15, 'sow': 4, 'l': 13, 'gf': 59, 'ga': 49, 'pts': 49},
    (1996, 'FC Dallas'):              {'w': 12, 'sow': 5, 'l': 15, 'gf': 50, 'ga': 48, 'pts': 41},
    (1996, 'Sporting Kansas City'):   {'w': 12, 'sow': 5, 'l': 15, 'gf': 61, 'ga': 63, 'pts': 41},
    (1996, 'San Jose Earthquakes'):   {'w': 12, 'sow': 3, 'l': 17, 'gf': 50, 'ga': 50, 'pts': 39},
    (1996, 'Colorado Rapids'):        {'w':  9, 'sow': 2, 'l': 21, 'gf': 44, 'ga': 59, 'pts': 29},
    # 1997 ─────────────────────────────────────────────────────────────
    (1997, 'D.C. United'):            {'w': 17, 'sow': 4, 'l': 11, 'gf': 70, 'ga': 53, 'pts': 55},
    (1997, 'Tampa Bay Mutiny'):       {'w': 14, 'sow': 3, 'l': 15, 'gf': 55, 'ga': 60, 'pts': 45},
    (1997, 'Columbus Crew'):          {'w': 12, 'sow': 3, 'l': 17, 'gf': 42, 'ga': 41, 'pts': 39},
    (1997, 'New England Revolution'): {'w': 11, 'sow': 4, 'l': 17, 'gf': 40, 'ga': 53, 'pts': 37},
    (1997, 'Red Bull New York'):      {'w': 11, 'sow': 2, 'l': 19, 'gf': 43, 'ga': 53, 'pts': 35},
    (1997, 'Sporting Kansas City'):   {'w': 14, 'sow': 7, 'l': 11, 'gf': 57, 'ga': 51, 'pts': 49},
    (1997, 'LA Galaxy'):              {'w': 14, 'sow': 2, 'l': 16, 'gf': 55, 'ga': 44, 'pts': 44},
    (1997, 'FC Dallas'):              {'w': 13, 'sow': 3, 'l': 16, 'gf': 55, 'ga': 49, 'pts': 42},
    (1997, 'Colorado Rapids'):        {'w': 12, 'sow': 2, 'l': 18, 'gf': 50, 'ga': 59, 'pts': 38},
    (1997, 'San Jose Earthquakes'):   {'w':  9, 'sow': 3, 'l': 20, 'gf': 55, 'ga': 59, 'pts': 30},
    # 1998 ─────────────────────────────────────────────────────────────
    (1998, 'D.C. United'):            {'w': 17, 'sow': 7, 'l':  8, 'gf': 74, 'ga': 48, 'pts': 58},
    (1998, 'Columbus Crew'):          {'w': 15, 'sow': 0, 'l': 17, 'gf': 67, 'ga': 56, 'pts': 45},
    (1998, 'Red Bull New York'):      {'w': 12, 'sow': 3, 'l': 17, 'gf': 54, 'ga': 63, 'pts': 39},
    (1998, 'Miami Fusion'):           {'w': 10, 'sow': 5, 'l': 17, 'gf': 46, 'ga': 68, 'pts': 35},
    (1998, 'Tampa Bay Mutiny'):       {'w': 11, 'sow': 1, 'l': 20, 'gf': 46, 'ga': 57, 'pts': 34},
    (1998, 'New England Revolution'): {'w':  9, 'sow': 2, 'l': 21, 'gf': 53, 'ga': 66, 'pts': 29},
    (1998, 'LA Galaxy'):              {'w': 22, 'sow': 2, 'l':  8, 'gf': 85, 'ga': 44, 'pts': 68},
    (1998, 'Chicago Fire FC'):        {'w': 18, 'sow': 2, 'l': 12, 'gf': 62, 'ga': 45, 'pts': 56},
    (1998, 'Colorado Rapids'):        {'w': 14, 'sow': 2, 'l': 16, 'gf': 62, 'ga': 69, 'pts': 44},
    (1998, 'FC Dallas'):              {'w': 11, 'sow': 4, 'l': 17, 'gf': 43, 'ga': 59, 'pts': 37},
    (1998, 'San Jose Earthquakes'):   {'w': 10, 'sow': 3, 'l': 19, 'gf': 48, 'ga': 60, 'pts': 33},
    (1998, 'Sporting Kansas City'):   {'w': 10, 'sow': 2, 'l': 20, 'gf': 45, 'ga': 50, 'pts': 32},
    # 1999 ─────────────────────────────────────────────────────────────
    (1999, 'D.C. United'):            {'w': 17, 'sow': 6, 'l':  9, 'gf': 65, 'ga': 43, 'pts': 57},
    (1999, 'Columbus Crew'):          {'w': 13, 'sow': 6, 'l': 13, 'gf': 48, 'ga': 39, 'pts': 45},
    (1999, 'Tampa Bay Mutiny'):       {'w':  9, 'sow': 5, 'l': 18, 'gf': 51, 'ga': 50, 'pts': 32},
    (1999, 'Miami Fusion'):           {'w':  8, 'sow': 5, 'l': 19, 'gf': 42, 'ga': 59, 'pts': 29},
    (1999, 'New England Revolution'): {'w':  7, 'sow': 5, 'l': 20, 'gf': 38, 'ga': 53, 'pts': 26},
    (1999, 'Red Bull New York'):      {'w':  4, 'sow': 3, 'l': 25, 'gf': 32, 'ga': 64, 'pts': 15},
    (1999, 'LA Galaxy'):              {'w': 17, 'sow': 3, 'l': 12, 'gf': 49, 'ga': 29, 'pts': 54},
    (1999, 'FC Dallas'):              {'w': 16, 'sow': 3, 'l': 13, 'gf': 54, 'ga': 35, 'pts': 51},
    (1999, 'Chicago Fire FC'):        {'w': 15, 'sow': 3, 'l': 14, 'gf': 51, 'ga': 36, 'pts': 48},
    (1999, 'Colorado Rapids'):        {'w': 14, 'sow': 6, 'l': 12, 'gf': 38, 'ga': 39, 'pts': 48},
    (1999, 'San Jose Earthquakes'):   {'w':  9, 'sow':10, 'l': 13, 'gf': 48, 'ga': 49, 'pts': 37},
    (1999, 'Sporting Kansas City'):   {'w':  6, 'sow': 2, 'l': 24, 'gf': 33, 'ga': 53, 'pts': 20},
}

SHOOTOUT_ERA_YEARS = {1996, 1997, 1998, 1999}


def early_record(team, season):
    """Return W-SOW-L string for a (team, season) in the shootout era, or None."""
    s = MLS_EARLY_STANDINGS.get((int(season), team))
    if not s:
        return None
    return f"{s['w']}-{s['sow']}-{s['l']}"


def canonical_team(name):
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return name
    return MLS_TEAM_ALIASES.get(name, name)


# ── Era-aware display names ─────────────────────────────────────────────────
# The rating system uses canonical team names for franchise continuity, but
# historical UI views should show what the team was actually called at the
# time. Maps canonical → list of (start_year, end_year_inclusive, display_name)
# ranges. 9999 = ongoing. Includes corrections where ESPN's scrape uses the
# modern name anachronistically (Houston Dynamo FC was just "Houston Dynamo"
# until 2020; San Jose Earthquakes were "San Jose Clash" 1996-1999, etc.).
MLS_TEAM_DISPLAY_HISTORY = {
    'Chicago Fire FC':      [(1998, 2019, 'Chicago Fire'),
                             (2020, 9999, 'Chicago Fire FC')],
    'Columbus Crew':        [(1996, 2014, 'Columbus Crew'),
                             (2015, 2020, 'Columbus Crew SC'),
                             (2021, 9999, 'Columbus Crew')],
    'CF Montréal':          [(2012, 2020, 'Montreal Impact'),
                             (2021, 9999, 'CF Montréal')],
    'FC Dallas':            [(1996, 2004, 'Dallas Burn'),
                             (2005, 9999, 'FC Dallas')],
    'Houston Dynamo FC':    [(2006, 2019, 'Houston Dynamo'),
                             (2020, 9999, 'Houston Dynamo FC')],
    'Red Bull New York':    [(1996, 1997, 'NY/NJ MetroStars'),
                             (1998, 2005, 'MetroStars'),
                             (2006, 9999, 'Red Bull New York')],
    'San Jose Earthquakes': [(1996, 1999, 'San Jose Clash'),
                             (2000, 9999, 'San Jose Earthquakes')],
    'Sporting Kansas City': [(1996, 1996, 'Kansas City Wiz'),
                             (1997, 2010, 'Kansas City Wizards'),
                             (2011, 9999, 'Sporting Kansas City')],
}


def display_name(canonical, year):
    """Era-appropriate display name for the given canonical team and year."""
    history = MLS_TEAM_DISPLAY_HISTORY.get(canonical)
    if not history:
        return canonical
    y = int(year)
    for start, end, name in history:
        if start <= y <= end:
            return name
    return canonical


def current_display_name(canonical):
    """The team's most recent display name (used for dropdowns / latest snapshot)."""
    history = MLS_TEAM_DISPLAY_HISTORY.get(canonical)
    if not history:
        return canonical
    return history[-1][2]


def historical_display_names(canonical):
    """List of distinct prior display names (most recent first), excluding
    the current name. Used to show "(MetroStars / NY/NJ MetroStars)" hint
    in the Team Summary dropdown."""
    history = MLS_TEAM_DISPLAY_HISTORY.get(canonical)
    if not history:
        return []
    current = history[-1][2]
    seen = {current}
    out = []
    # Walk newest → oldest (excluding current which is the last entry)
    for _, _, name in reversed(history[:-1]):
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


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
    'Chicago Fire FC':           [(1998, 2001, 'West'), (2002, 9999, 'East')],
    'Chivas USA':                [(2005, 2014, 'West')],
    'Colorado Rapids':           [(1996, 9999, 'West')],
    'Columbus Crew':             [(1996, 9999, 'East')],
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
    'Sporting Kansas City':      [(1996, 1999, 'West'), (2000, 2010, 'East'), (2011, 9999, 'West')],
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
# Defensive normalization (cobi.py should already have done this)
df['team'] = df['team'].map(canonical_team)
df['conference'] = df.apply(lambda r: conference_for(r['team'], r['season']), axis=1)


# cobi.py constructs last_match strings using the canonical franchise name
# (e.g. "L vs. Chicago Fire FC 0-1 (MLS)" for a 2010 game when the team was
# actually called the Chicago Fire). Rewrite the opponent portion with the
# era-appropriate display name. Liga MX / foreign opponents in CCL games
# don't match any canonical so they pass through unchanged.
_LAST_MATCH_RE = re.compile(r'^([WLD])\s+(vs\.?|@)\s+(.+?)\s+(\d+\s*-\s*\d+)\s*(\([^)]+\))?\s*$')

def era_aware_last_match(raw, season):
    if not raw:
        return raw
    m = _LAST_MATCH_RE.match(str(raw))
    if not m:
        return raw
    letter, venue, opponent, score, comp = m.groups()
    new_opp = display_name(opponent.strip(), str(season))
    suffix = f' {comp}' if comp else ''
    return f"{letter} {venue} {new_opp} {score}{suffix}"


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
# Defensive normalization (cobi.py should already have done this)
games_raw['home_team'] = games_raw['home_team'].map(canonical_team)
games_raw['away_team'] = games_raw['away_team'].map(canonical_team)
if 'shootout_winner' in games_raw.columns:
    games_raw['shootout_winner'] = games_raw['shootout_winner'].map(canonical_team)
games_raw['snap_season'] = games_raw['date'].dt.year.astype(str)

games_lg = games_raw[games_raw['competition'] == 'MLS'].copy()

# Teams that played at least one MLS league game in each calendar year.
# Used to filter snapshots so a defunct team's stale rating doesn't linger
# (e.g., Miami Fusion folded after 2001 → no 2002 games → drop from 2002
# snapshots even if their 2001 games are still in the rolling window).
teams_by_season = {}
for s, sg in games_lg.groupby('snap_season'):
    teams_by_season[str(s)] = set(sg['home_team']).union(set(sg['away_team']))


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
# Gate both EOS and EORS on the trophy-awarded signal from cobi.py — Shield
# fires 7 days after Decision Day, Cup fires 7 days after MLS Cup, so
# either field == 'Champion' is a clean "this milestone has happened"
# proxy. Without these gates, in-progress seasons (current year before
# Decision Day) would falsely tag their most-recent snapshot as
# "End of Postseason", since season_last_snap is just whatever's latest.
seasons_with_cup_winner    = set(
    df[df.get('mls_cup_finish', '') == 'Champion']['season'].unique()
)
seasons_with_shield_winner = set(
    df[df.get('supporters_shield_finish', '') == 'Champion']['season'].unique()
)

season_last_snap = {
    s: rid for s, rid in
    df.groupby('season')['ranking_id'].max().to_dict().items()
    if s in seasons_with_cup_winner
}
df['is_end_of_season'] = df.apply(
    lambda r: 1 if r['ranking_id'] == season_last_snap.get(r['season']) else 0,
    axis=1
)

# End-of-regular-season snapshot per season: last ranking_id whose date is
# on or before that year's Decision Day. Used by the GOAT table and the
# Shield row of the Champions tab so Shield form isn't dragged down by a
# team's playoff exit.
df['_date_only'] = pd.to_datetime(df['date']).dt.date
season_last_reg_snap = {}
for season, sub in df.groupby('season'):
    if season not in seasons_with_shield_winner:
        continue
    dd = _decision_day_by_year.get(int(season))
    if dd is None:
        continue
    eligible = sub[sub['_date_only'] <= dd]
    if not eligible.empty:
        season_last_reg_snap[season] = eligible['ranking_id'].max()
df['is_end_of_regular_season'] = df.apply(
    lambda r: 1 if r['ranking_id'] == season_last_reg_snap.get(r['season']) else 0,
    axis=1
)
eors_rating_lookup = {
    (r['team'], r['season']): float(r['rating'])
    for _, r in df[df['is_end_of_regular_season'] == 1].iterrows()
}
df.drop(columns=['_date_only'], inplace=True)

# Per-team conference rank within each snapshot (rank within East / West)
df['conf_rank'] = (
    df.groupby(['ranking_id', 'conference'])['rating']
    .rank(ascending=False, method='min')
    .astype(int)
)

# EORS rank/conf_rank lookups (now that conf_rank exists)
eors_rank_lookup = {
    (r['team'], r['season']): (int(r['rank']), int(r['conf_rank']))
    for _, r in df[df['is_end_of_regular_season'] == 1].iterrows()
}

latest_id = int(df['ranking_id'].max())
latest = df[df['ranking_id'] == latest_id].sort_values('rank').copy()
latest_date_str = str(latest['date'].iloc[0])


# ── East-vs-West MLS Cup format eras ────────────────────────────────────────
# Conference badges go gold ("East 🏆") only when the year's MLS Cup playoff
# format was DESIGNED to guarantee East-vs-West finalists (separate conf
# brackets → conf finals → MLS Cup). The 2001-2012 wild-card / reseeding
# era allowed same-conference finals by design (and several happened: 2001
# SJ-LA both W; 2004 DC-KC both E; 2008 Columbus-NY both E; 2009/10/11/12
# all both W) — so no conference-champion badges for those years even when
# the final happened to be cross-conf. Mirrors LOBO's pattern: only show
# the conference winner badge when the league formally crowned conference
# champions en route to the title.
def is_east_west_format_year(year):
    y = int(year)
    return y <= 2000 or y >= 2013


# Pull MLS Cup finalists from trophies CSV so we know who to flag with the
# conf-champion badge in the East-vs-West-design years.
_trophies_for_conf = pd.read_csv('cobi_trophies.csv')
_cup_pairs = {}  # year (int) → {champ_team, runner_up_team}
for (yr, label), grp in _trophies_for_conf.groupby([
    'year',
    _trophies_for_conf['honor'].str.replace(' Champion', '', regex=False)
                                .str.replace(' Runner-Up', '', regex=False)
]):
    if label != 'MLS Cup':
        continue
    champ_row = grp[grp['honor'].str.endswith('Champion')]
    ru_row    = grp[grp['honor'].str.endswith('Runner-Up')]
    if champ_row.empty or ru_row.empty:
        continue
    _cup_pairs[int(yr)] = {champ_row.iloc[0]['team'], ru_row.iloc[0]['team']}


def is_cup_conf_finalist(team, year):
    """True iff team made MLS Cup that year AND the year's playoff format
    was structured around conference brackets (1996-2000, 2013+)."""
    yr = int(year)
    return is_east_west_format_year(yr) and team in _cup_pairs.get(yr, set())
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
            'display_name':        display_name(r['team'], r['season']),
            'conference':          clean(r['conference']),
            'rating':              round(float(r['rating']), 3),
            'regular_record':      cur_reg.get(r['team'], '0-0-0'),
            'playoff_record':      cur_po.get(r['team'], ''),
            'games_played':        int(r['games_played']),
            'last_match':          era_aware_last_match(clean(r['last_match']), r['season']),
            'last_match_date':     clean(r['last_match_date']),
            'mls_cup_finish':            clean(r.get('mls_cup_finish', '')),
            'supporters_shield_finish':  clean(r.get('supporters_shield_finish', '')),
            'mls_cup_conf_finalist':     is_cup_conf_finalist(r['team'], r['season']),
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
    # Shootout-era (1996-1999) override: substitute era-accurate W-SOW-L
    # from Wikipedia season totals so end-of-season records and pts come
    # out right. Computed cumulative records would over-credit shootout
    # wins (treating them all as regulation Ws → 3 pts each).
    reg_rec = early_record(team, str(season)) or \
              regular_record_as_of(team, str(season), str(last['date']))
    eors_rating = eors_rating_lookup.get((team, str(season)))
    eors_ranks  = eors_rank_lookup.get((team, str(season)))
    eoy_lookup[(team, str(season))] = {
        'team':                       team,
        'display_name':               display_name(team, str(season)),
        'conference':                 clean(last['conference']),
        'rating':                     round(float(last['rating']), 3),
        'rating_postseason':          round(float(last['rating']), 3),
        'rating_regular_season':      round(float(eors_rating), 3) if eors_rating is not None else None,
        'rank':                       int(last['rank']),
        'conf_rank':                  int(last['conf_rank']),
        'rank_regular_season':        eors_ranks[0] if eors_ranks else None,
        'conf_rank_regular_season':   eors_ranks[1] if eors_ranks else None,
        'regular_record':             reg_rec,
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
        champion = {'team': champ_team, 'display_name': display_name(champ_team, str(year)),
                    'conference': '', 'rating': None,
                    'rank': None, 'conf_rank': None,
                    'regular_record': '0-0-0', 'playoff_record': '',
                    'mls_cup_finish': '', 'supporters_shield_finish': ''}
    if runner_up is None:
        runner_up = {'team': ru_team, 'display_name': display_name(ru_team, str(year)),
                     'conference': '', 'rating': None,
                     'rank': None, 'conf_rank': None,
                    'regular_record': '0-0-0', 'playoff_record': '',
                     'mls_cup_finish': '', 'supporters_shield_finish': ''}
    score_fn = dict(CHAMPIONS_TROPHIES)[label]
    # Copy the champion/runner_up dicts so each trophy's entry has its OWN
    # mutable copy. Without this, a team that wins both MLS Cup and Shield
    # in the same year (a Double) shares the same eoy_lookup dict between
    # both trophy entries — and the second trophy's title_count walk
    # overwrites the first's. Bit LA Galaxy 2011 (MLS Cup #3 instead of #2).
    champions_by_trophy[label].append({
        'season':      str(year),
        'champion':    dict(champion),
        'runner_up':   dict(runner_up),
        'final_score': score_fn(year, champ_team, ru_team),
    })

for label in champions_by_trophy:
    champions_by_trophy[label].sort(key=lambda e: e['season'], reverse=True)

# Cumulative title counts per (team, trophy). Walks each trophy's entries
# chronologically (oldest first), tally champion + runner-up running totals,
# and attach to each entry for display as "(N 🏆)" / "(N 🛡️)" / "(N 🥈)" in
# the UI. Mirrors DUNCAN / GRIFFEY / SAKIC pattern.
for label, entries in champions_by_trophy.items():
    champ_count = {}
    ru_count    = {}
    for entry in reversed(entries):  # reversed = chronological (oldest first)
        ct = entry['champion']['team']
        rt = entry['runner_up']['team']
        champ_count[ct] = champ_count.get(ct, 0) + 1
        ru_count[rt]    = ru_count.get(rt, 0) + 1
        entry['champion']['title_count']        = champ_count[ct]
        entry['runner_up']['runner_up_count']   = ru_count[rt]

with open('docs/data/champions.json', 'w') as f:
    json.dump(champions_by_trophy, f, separators=(',', ':'))


# ── 3. GOAT tables (RS + PS) ─────────────────────────────────────────────────
# Two lists matching the fleet pattern (DILLON / GRIFFEY / SAKIC):
#   - goat_rs.json: top 50 by end-of-regular-season rating, all teams eligible.
#   - goat_ps.json: top 50 by end-of-postseason rating, restricted to MLS Cup
#     participants (final-game contenders) so the list shows actual
#     championship-level teams, not playoff flameouts.
GOAT_TOP_N = 50
print("Writing goat_rs.json + goat_ps.json...")

final_reg_lookup = {(t, s): grp.sort_values('date').iloc[-1]['record']
                    for (t, s), grp in reg.groupby(['team', 'snap_season'])}
final_po_lookup  = {(t, s): grp.sort_values('date').iloc[-1]['record']
                    for (t, s), grp in po.groupby(['team', 'snap_season'])}


# Short / disrupted seasons — flagged on GOAT/Champions/Standings/TeamSummary
# rows so the UI can tag them inline + footnote.
SHORT_SEASONS = {
    2020: {
        'tag': 'COVID',
        'category': 'covid',
        'note': 'The 2020 season was suspended in March, resumed with the "MLS Is Back" tournament in an Orlando bubble (July-August), then continued with a regional regular season averaging ~23 games per team (vs typical 34).',
    },
}


def _goat_row(r, rating_to_show, rank):
    s = int(r['season']) if not pd.isna(r['season']) else 0
    return {
        'rank':                       rank,
        'team':                       r['team'],
        'display_name':               display_name(r['team'], r['season']),
        'season':                     r['season'],
        'short_season':               s in SHORT_SEASONS,
        'short_season_tag':           SHORT_SEASONS.get(s, {}).get('tag', '')      if s in SHORT_SEASONS else '',
        'short_season_category':      SHORT_SEASONS.get(s, {}).get('category', '') if s in SHORT_SEASONS else '',
        'short_season_note':          SHORT_SEASONS.get(s, {}).get('note', '')     if s in SHORT_SEASONS else '',
        'conference':                 clean(r['conference']),
        'rating':                     round(float(rating_to_show), 3),
        'regular_record':             early_record(r['team'], r['season']) or
                                      final_reg_lookup.get((r['team'], r['season']), '0-0-0'),
        'playoff_record':             final_po_lookup.get((r['team'], r['season']), ''),
        'mls_cup_finish':             clean(r.get('mls_cup_finish', '')),
        'supporters_shield_finish':   clean(r.get('supporters_shield_finish', '')),
        'mls_cup_conf_finalist':      is_cup_conf_finalist(r['team'], r['season']),
    }


# GOAT-RS: end-of-regular-season snapshots, ALL teams eligible.
eors_rows = (
    df[df['is_end_of_regular_season'] == 1]
    .sort_values('rating', ascending=False)
    .head(GOAT_TOP_N)
    .reset_index(drop=True)
)
goat_rs = [_goat_row(r, r['rating'], i + 1) for i, (_, r) in enumerate(eors_rows.iterrows())]

# GOAT-PS: end-of-season snapshots, MLS Cup participants only
# (champion OR runner-up of the MLS Cup final). Same fleet pattern as
# DILLON (SB participants), GRIFFEY (WS), SAKIC (Stanley Cup Final).
eos = df[df['is_end_of_season'] == 1].copy()
played_mls_cup = eos.get('mls_cup_finish', '').isin(['Champion', 'Runner-Up'])
eos = (
    eos[played_mls_cup]
    .sort_values('rating', ascending=False)
    .head(GOAT_TOP_N)
    .reset_index(drop=True)
)
goat_ps = [_goat_row(r, r['rating'], i + 1) for i, (_, r) in enumerate(eos.iterrows())]

with open('docs/data/goat_rs.json', 'w') as f:
    json.dump(goat_rs, f, separators=(',', ':'))
with open('docs/data/goat_ps.json', 'w') as f:
    json.dump(goat_ps, f, separators=(',', ':'))


# ── 4. Per-team JSON files ───────────────────────────────────────────────────
print("Writing per-team JSON files...")
game_days = df[
    (df['is_game_day'] == 1) |
    (df['is_end_of_season'] == 1) |
    (df['is_end_of_regular_season'] == 1)
].copy()
game_days = game_days.sort_values(['team', 'season', 'date'])

all_teams = sorted(df['team'].unique())
teams_index = []

for team in all_teams:
    tdf = game_days[game_days['team'] == team]
    if len(tdf) == 0:
        continue
    conference = current_conference(team)
    team_slug = slug(team)
    teams_index.append({'name': team, 'display_name': current_display_name(team),
                        'historical_names': historical_display_names(team),
                        'conference': conference, 'slug': team_slug})

    seasons = {}
    for season, sdf in tdf.groupby('season'):
        # Skip seasons where the team didn't actually play any MLS league games.
        # Without this, a defunct team gets a "ghost" season the year after they
        # fold (Miami Fusion / Tampa Bay Mutiny in 2002) because the solver
        # still publishes EOS / EORS snapshots for them while their 2001 games
        # are in the rolling window. Same `teams_by_season` filter the per-season
        # JSON loop already uses below.
        if team not in teams_by_season.get(str(season), set()):
            continue
        seasons[str(season)] = [
            {
                'date':              str(r['date']),
                'display_name':      display_name(team, str(season)),
                'conference':        conference_for(team, str(season)),
                'rating':            round(float(r['rating']), 3),
                'rank':              int(r['rank']),
                'conf_rank':         int(r['conf_rank']),
                'is_end_of_season':         int(r['is_end_of_season']),
                'is_end_of_regular_season': int(r['is_end_of_regular_season']),
                'regular_record':    (early_record(team, str(season)) if int(r['is_end_of_season']) == 1 else None)
                                     or regular_record_as_of(team, str(season), str(r['date'])),
                'playoff_record':    playoff_record_as_of(team, str(season), str(r['date'])),
                'last_match':        era_aware_last_match(clean(r['last_match']), season),
                'mls_cup_finish':            clean(r.get('mls_cup_finish', '')),
                'supporters_shield_finish':  clean(r.get('supporters_shield_finish', '')),
                'mls_cup_conf_finalist':     is_cup_conf_finalist(team, str(season)),
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
    # Filter to teams that actually played in this season — drops defunct
    # franchises whose stale rating is still in the rolling window (e.g.,
    # Miami Fusion folded after 2001, would otherwise appear in early-2002
    # snapshots until their 2001 games age out of the 200-game-day window).
    active = teams_by_season.get(str(season), set())
    if active:
        sdf = sdf[sdf['team'].isin(active)]
    sdf = sdf.sort_values(['date', 'rank'])

    snaps = []
    for snap_date, gdf in sdf.groupby('date'):
        gdf = gdf.sort_values('rank')
        # Re-rank within the snapshot since filtering may have left gaps
        gdf = gdf.reset_index(drop=True)
        gdf['snap_rank'] = gdf['rating'].rank(ascending=False, method='min').astype(int)
        gdf['snap_conf_rank'] = (
            gdf.groupby('conference')['rating']
            .rank(ascending=False, method='min')
            .astype(int)
        )
        snap_is_eos  = int(gdf['is_end_of_season'].max()) == 1
        snap_is_eors = int(gdf['is_end_of_regular_season'].max()) == 1
        teams = []
        for _, r in gdf.iterrows():
            reg_rec = (early_record(r['team'], season) if snap_is_eos else None) \
                      or regular_record_as_of(r['team'], season, str(snap_date))
            teams.append({
                'rank':              int(r['snap_rank']),
                'conf_rank':         int(r['snap_conf_rank']),
                'team':              r['team'],
                'display_name':      display_name(r['team'], season),
                'conference':        clean(r['conference']),
                'rating':            round(float(r['rating']), 3),
                'regular_record':    reg_rec,
                'playoff_record':    playoff_record_as_of(r['team'], season, str(snap_date)),
                'last_match':        era_aware_last_match(clean(r['last_match']), season),
                'last_match_date':   clean(r['last_match_date']),
                'mls_cup_finish':            clean(r.get('mls_cup_finish', '')),
                'supporters_shield_finish':  clean(r.get('supporters_shield_finish', '')),
                'mls_cup_conf_finalist':     is_cup_conf_finalist(r['team'], season),
            })
        # If the snapshot is BOTH EORS and EOS (rare — mid-1990s seasons
        # that ended on Decision Day with no playoff round in our data, or
        # in-progress seasons before playoffs start), prefer the EOS label
        # since it reflects the actual final state for the season.
        if snap_is_eos:
            snap_label = 'End of playoffs'
        elif snap_is_eors:
            snap_label = 'End of regular season'
        else:
            snap_label = None
        snaps.append({
            'date':                     str(snap_date),
            'label':                    snap_label,
            'is_end_of_season':         int(gdf['is_end_of_season'].max()),
            'is_end_of_regular_season': int(gdf['is_end_of_regular_season'].max()),
            'teams':                    teams,
        })

    with open(f'docs/data/seasons/{season}.json', 'w') as f:
        json.dump({'season': season, 'snapshots': snaps},
                  f, separators=(',', ':'))

df_dates = pd.to_datetime(df['date'])
# first_date / last_date describe the underlying GAME data, not surviving
# rating snapshots — the min_games filter shaves the first few weeks of
# each team's history off the rated output, but those games still happened.
seasons_index = {
    'seasons':      all_seasons,
    'first_date':   str(games_lg['date'].min().date()),
    'last_date':    str(games_lg['date'].max().date()),
    'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    'disrupted_seasons': {
        str(year): {'tag': info['tag'], 'category': info['category'], 'note': info['note']}
        for year, info in SHORT_SEASONS.items()
    },
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
