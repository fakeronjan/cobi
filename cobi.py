# ============================================================
# COBI - MLS Power Rankings
# Named after Cobi Jones (LA Galaxy 1996-2007, USMNT)
# Based on ZIDANE / MESSI / LOGAN architecture
# ============================================================

import sys
import time
from datetime import date, datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import requests

# ============================================================
# PARAMETERS
# ============================================================

window_game_days = 100   # ~1 MLS season; tighter than ZIDANE/MESSI on
                          # purpose — MLS has heavy player turnover (European
                          # stars arriving post-prime, MLS young stars
                          # leaving for Europe), so we want the ratings to
                          # respond to current rosters quickly.
margin_cap       = 4
shootout_margin  = 0.5
home_field_adv   = 0.5
min_games        = 5    # Threshold for inclusion in final standings.
                          # Set to 5 because Massey ratings are stable enough
                          # by then; legacy 15 nuked the first ~6 weeks of
                          # 1996 (only KC had 15+ games until late June) and
                          # Inter Miami's entire 2020 pre-October.

# WLS: weights affect observation influence, not margin magnitude.
# Margin transform (cap=4) + per-game HCA are applied in prepare_game_data
# upstream — solver just takes the pre-prepped adj_margin_home as input.
WEIGHTING_MODE = "wls"

# Re-process the most recent N ranking_ids (game-days) on every run so late-
# arriving ESPN data is absorbed. Without this, a cron firing mid-day caches
# that day's snapshot and never re-ranks it even when more games for the
# same day land hours later.
RECOMPUTE_TAIL_DAYS = 7

# ============================================================
# ERA + DATA SOURCE CONFIG
# ============================================================

FIRST_SEASON_YEAR = 1996  # MLS inaugural season

ESPN_BASE = 'https://site.api.espn.com/apis/site/v2/sports/soccer'

# (competition_label, espn_slug, first_year_in_espn, league_match_flag)
ESPN_COMPETITIONS = [
    ('MLS',          'usa.1',                  2002, True),
]

LEAGUE_COMPETITIONS = {'MLS'}  # the only rated competition

# Same-franchise rebrands → consolidate to one canonical name so a team's
# rating is continuous across name changes. Relocations are NOT aliased
# (San Jose → Houston was a move, not a rename — they remain distinct).
# Forward-looking entries cover names that don't currently appear in our
# scrape window (2002+) but would matter if older data is added.
MLS_TEAM_ALIASES = {
    # Chicago franchise (1998+)
    'Chicago Fire':                   'Chicago Fire FC',
    # Columbus franchise (1996+)
    'Columbus Crew SC':               'Columbus Crew',
    # NY/NJ franchise (1996+) — no-ops on current data, future-proofing
    'NY/NJ MetroStars':               'Red Bull New York',
    'MetroStars':                     'Red Bull New York',
    'New York Red Bulls':             'Red Bull New York',
    # Kansas City franchise (1996+)
    'Kansas City Wiz':                'Sporting Kansas City',
    'Kansas City Wizards':            'Sporting Kansas City',
    # Dallas franchise (1996+)
    'Dallas Burn':                    'FC Dallas',
    # LA Galaxy (full name from Wikipedia's standings)
    'Los Angeles Galaxy':             'LA Galaxy',
    # San Jose franchise (1996+)
    'San Jose Clash':                 'San Jose Earthquakes',
    # Montreal franchise (2012+)
    'Montreal Impact':                'CF Montréal',
}


def _solve_massey(window_df, weighting_mode):
    """
    Homebrew weighted-least-squares Massey solver. Replaces rankit.

    Takes a window df with home_team, away_team, adj_margin_home (the
    HCA + cap pre-applied margin from home perspective), and date_weight.
    Returns DataFrame with columns: name, rating, rank.

    Margin transform + per-game HCA are applied UPSTREAM in
    prepare_game_data (since COBI has soccer-specific quirks like
    shootout-margin handling that need to happen before solver). The
    solver just does WLS on the pre-prepped response variable.
    """
    teams = sorted(set(window_df["home_team"]) | set(window_df["away_team"]))
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    n_games = len(window_df)

    X = np.zeros((n_games + 1, n_teams))
    y = np.zeros(n_games + 1)
    w = np.zeros(n_games + 1)

    adj_margin = window_df["adj_margin_home"].to_numpy(dtype=float)
    weights    = window_df["date_weight"].to_numpy(dtype=float)
    home_names = window_df["home_team"].to_numpy()
    away_names = window_df["away_team"].to_numpy()

    for i in range(n_games):
        X[i, team_idx[home_names[i]]] = 1.0
        X[i, team_idx[away_names[i]]] = -1.0

    if weighting_mode == "wls":
        y[:n_games] = adj_margin
        w[:n_games] = weights
    elif weighting_mode == "margin_scale":
        y[:n_games] = adj_margin * weights
        w[:n_games] = 1.0
    else:
        raise ValueError(f"Unknown WEIGHTING_MODE: {weighting_mode}")

    # Zero-sum constraint via high-weight extra row.
    X[-1, :] = 1.0
    y[-1] = 0.0
    w[-1] = 1.0e8

    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    r, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    out = pd.DataFrame({"name": teams, "rating": r})
    out["rank"] = out["rating"].rank(ascending=False, method="min").astype(int)
    return out


def canonical_team(name):
    if name is None:
        return name
    return MLS_TEAM_ALIASES.get(name, name)


# Shootout-era (1996-1999) regular-season final standings from Wikipedia.
# MLS used a no-draws shootout format then (regulation W = 3 pts, shootout
# W = 1 pt, L = 0). Our gap-fill source folded shootout outcomes into
# regulation results so we can't recover the W vs SOW split from the games
# CSV. Hard-code season-final per-team standings so the Shield winner +
# end-of-season records are era-correct. Verified Pts = 3*W + SOW.
# Keep in sync with generate_data.py.
MLS_EARLY_STANDINGS = {
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


_today = date.today()
CURRENT_YEAR = _today.year

# ============================================================
# ESPN FETCH HELPERS
# ============================================================

def espn_fetch_year(slug, year, max_retries=3, sleep=0.5):
    """Pull a full calendar year of events from ESPN's scoreboard endpoint."""
    url = f"{ESPN_BASE}/{slug}/scoreboard"
    params = {'dates': f'{year}0101-{year}1231', 'limit': 1000}
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            time.sleep(sleep)
            return r.json()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [warn] {slug} {year}: {e}")
                return {}
            time.sleep(2 ** attempt)
    return {}


def espn_extract_matches(payload, competition_label, league_match_flag):
    """Convert ESPN scoreboard payload into unified match rows."""
    rows = []
    for ev in payload.get('events', []) or []:
        status = (ev.get('status') or {}).get('type', {}) or {}
        if status.get('state') != 'post' or not status.get('completed', False):
            continue

        comps = ev.get('competitions') or []
        if not comps:
            continue
        comp = comps[0]
        teams = comp.get('competitors') or []
        if len(teams) != 2:
            continue

        home = next((t for t in teams if t.get('homeAway') == 'home'), None)
        away = next((t for t in teams if t.get('homeAway') == 'away'), None)
        if not home or not away:
            continue

        try:
            home_score = int(home.get('score', 0) or 0)
            away_score = int(away.get('score', 0) or 0)
        except (TypeError, ValueError):
            continue

        desc = (status.get('description') or '').lower()
        penalties = ('penalt' in desc) or ('shootout' in desc)
        home_pen = away_pen = None
        shootout_winner = None
        if penalties:
            try:
                hp = home.get('shootoutScore')
                ap = away.get('shootoutScore')
                home_pen = int(hp) if hp is not None else None
                away_pen = int(ap) if ap is not None else None
                if home_pen is not None and away_pen is not None:
                    if home_pen > away_pen:
                        shootout_winner = (home.get('team') or {}).get('displayName')
                    elif away_pen > home_pen:
                        shootout_winner = (away.get('team') or {}).get('displayName')
            except (TypeError, ValueError):
                pass
            # ESPN sometimes flags the winner directly via competitor.winner=true
            # even when shootoutScore isn't populated.
            if shootout_winner is None and home_score == away_score:
                hw = home.get('winner')
                aw = away.get('winner')
                if hw is True:
                    shootout_winner = (home.get('team') or {}).get('displayName')
                elif aw is True:
                    shootout_winner = (away.get('team') or {}).get('displayName')

        raw_date = ev.get('date') or comp.get('date') or ''
        try:
            # ESPN's `date` field is the kickoff in UTC. Naively .date()-ing
            # it puts any 7pm-or-later US-local kickoff on the next UTC day
            # (a 7:30pm PT game = 02:30Z next day, etc.) — which mis-attributes
            # the game and breaks date filtering / standings. All current MLS
            # venues are UTC-4 (Eastern DST) through UTC-7 (Pacific DST), so
            # subtracting 7h before .date() always lands on the correct
            # local schedule day regardless of venue, since MLS only schedules
            # evening kickoffs.
            kickoff_utc = datetime.fromisoformat(raw_date.replace('Z', '+00:00'))
            match_date = (kickoff_utc - timedelta(hours=7)).date()
        except ValueError:
            continue

        venue_obj = comp.get('venue') or {}
        neutral = bool(comp.get('neutralSite', False))

        rows.append({
            'date':            match_date.isoformat(),
            'season':          _season_for_match(competition_label, match_date),
            'competition':     competition_label,
            'league_match':    league_match_flag,
            'home_team':       canonical_team((home.get('team') or {}).get('displayName') or ''),
            'away_team':       canonical_team((away.get('team') or {}).get('displayName') or ''),
            'home_score':      home_score,
            'away_score':      away_score,
            'home_pen':        home_pen,
            'away_pen':        away_pen,
            'penalties':       penalties,
            'shootout_winner': shootout_winner,
            'neutral':         neutral,
            'venue':           venue_obj.get('fullName') or '',
            'event_id':        ev.get('id') or '',
        })
    return rows


def _season_for_match(competition, match_date):
    """Per-match season label. MLS uses calendar year (Feb-Dec)."""
    return str(match_date.year)


def scrape_espn_all(start_year=None, end_year=None):
    end_year = end_year or CURRENT_YEAR
    all_rows = []
    for label, slug, first_year, league_flag in ESPN_COMPETITIONS:
        fy = max(first_year, start_year or first_year)
        print(f"[ESPN] {label} ({slug}) — {fy}-{end_year}")
        for year in range(fy, end_year + 1):
            payload = espn_fetch_year(slug, year)
            rows = espn_extract_matches(payload, label, league_flag)
            print(f"  {year}: {len(rows)} matches")
            all_rows.extend(rows)
    return pd.DataFrame(all_rows)


# ============================================================
# HISTORICAL GAP-FILL LOADERS
# ============================================================
# Static CSV produced by the dedicated scraper:
#   - mls_early_historical.csv     (footballcsv 1996-2001)
# Run scrape_mls_early.py separately to refresh; cobi.py just merges it in.

GAPFILL_PATHS = [
    'mls_early_historical.csv',
]


def load_gapfill():
    """Concatenate any present gap-fill CSVs into a single DataFrame.
    Returns an empty DataFrame if none exist."""
    import os
    frames = []
    for p in GAPFILL_PATHS:
        if not os.path.exists(p):
            print(f"  [gapfill] {p} not found — skipping")
            continue
        df = pd.read_csv(p)
        print(f"  [gapfill] {p}: {len(df):,} rows")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def merge_match_sources(espn_df, gapfill_df):
    """Combine ESPN + gap-fill data, dedupe by (date, home_team, away_team)
    keeping the ESPN row when both exist (richer metadata)."""
    if gapfill_df.empty:
        return espn_df.copy()
    if espn_df.empty:
        return gapfill_df.copy()

    espn = espn_df.copy()
    espn['_src_priority'] = 0  # ESPN preferred
    gap = gapfill_df.copy()
    gap['_src_priority'] = 1

    combined = pd.concat([espn, gap], ignore_index=True, sort=False)
    combined = combined.sort_values(['_src_priority'])
    combined = combined.drop_duplicates(
        subset=['date', 'home_team', 'away_team'],
        keep='first',
    )
    combined = combined.drop(columns=['_src_priority']).reset_index(drop=True)
    return combined


# ============================================================
# RATING PIPELINE
# ============================================================

def date_to_snapshot_season(d):
    """Snapshot season label = calendar year. MLS runs Feb-Dec, fits cleanly."""
    return str(d.year)


def run_pipeline(scrape=True):
    # ---- 1. Load match data ----
    if scrape:
        espn_df = scrape_espn_all()
        print("\n[gapfill] Loading historical CSVs...")
        gap_df = load_gapfill()
        merged = merge_match_sources(espn_df, gap_df)
        merged.to_csv('all_club_games.csv', index=False)
        print(f"[merge] {len(merged):,} matches "
              f"(ESPN {len(espn_df):,} + gap {len(gap_df):,} after dedupe) → all_club_games.csv\n")

    df = pd.read_csv('all_club_games.csv')
    # Patches for known data gaps in the merged source. ESPN sometimes leaves
    # shootout_winner empty even when penalties=True (the 2006 MLS Cup is the
    # canonical example: NE 1-1 Houston, Houston won 4-3 on PKs). Apply
    # known overrides keyed by (date, home_team, away_team).
    SHOOTOUT_OVERRIDES = {
        ('2006-11-12', 'New England Revolution', 'Houston Dynamo FC'): 'Houston Dynamo FC',
    }
    for (d, h, a), sw in SHOOTOUT_OVERRIDES.items():
        mask = (df['date'].astype(str).str[:10] == d) & \
               (df['home_team'] == h) & (df['away_team'] == a)
        if mask.any():
            df.loc[mask, 'shootout_winner'] = sw
            df.loc[mask, 'penalties'] = True

    # Defensive: drop any non-MLS rows lingering from pre-refactor data so the
    # on-disk CSV ends up MLS-only regardless of whether we re-scraped.
    pre_count = len(df)
    df = df[df['competition'] == 'MLS'].copy()
    # Apply same-franchise alias map so rebrands (Chicago Fire → Chicago Fire
    # FC, Columbus Crew SC → Columbus Crew) become one continuous team in the
    # ratings. Both load-time AND scrape-time normalize, so the on-disk CSV
    # converges to canonical names.
    pre_home = df['home_team'].copy()
    df['home_team'] = df['home_team'].map(lambda n: canonical_team(n))
    df['away_team'] = df['away_team'].map(lambda n: canonical_team(n))
    n_renamed = (pre_home != df['home_team']).sum()
    if len(df) != pre_count or n_renamed > 0:
        if len(df) != pre_count:
            print(f"  filtered {pre_count - len(df):,} non-MLS rows from on-disk data")
        if n_renamed > 0:
            print(f"  normalized {n_renamed:,} home_team names via MLS_TEAM_ALIASES")
        df.to_csv('all_club_games.csv', index=False)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True, kind='stable')
    df.reset_index(drop=True, inplace=True)

    # Drop any malformed rows
    df = df.dropna(subset=['home_score', 'away_score', 'home_team', 'away_team']).copy()
    df['home_score_int'] = df['home_score'].astype(int)
    df['away_score_int'] = df['away_score'].astype(int)

    print(f"Loaded {len(df):,} matches, {df['date'].min().date()} → {df['date'].max().date()}")

    # ---- 2. Margins (incl. shootout treatment) ----
    df['raw_margin_home'] = df['home_score_int'] - df['away_score_int']
    df['margin_home'] = df['raw_margin_home'].astype(float)

    shootout_mask = df['shootout_winner'].notna() & (df['raw_margin_home'] == 0)
    df.loc[shootout_mask & (df['shootout_winner'] == df['home_team']), 'margin_home'] =  shootout_margin
    df.loc[shootout_mask & (df['shootout_winner'] == df['away_team']), 'margin_home'] = -shootout_margin

    df['margin_home'] = df['margin_home'].clip(-margin_cap, margin_cap)

    df['hfa'] = np.where(df['neutral'].fillna(False).astype(bool), 0.0, home_field_adv)
    df['adj_margin_home'] = df['margin_home'] - df['hfa']

    # ---- 3. Result flags + last_match strings ----
    df['home_win'] = np.where(df['raw_margin_home'] > 0, 1.0,
                       np.where(df['raw_margin_home'] == 0, 0.5, 0.0))
    df.loc[shootout_mask & (df['shootout_winner'] == df['home_team']), 'home_win'] = 1.0
    df.loc[shootout_mask & (df['shootout_winner'] == df['away_team']), 'home_win'] = 0.0
    df['away_win'] = 1.0 - df['home_win']

    def _flag(w):
        if w == 1.0: return 'W'
        if w == 0.0: return 'L'
        return 'D'

    df['home_result_flag'] = df['home_win'].map(_flag)
    df['away_result_flag'] = df['away_win'].map(_flag)

    df['home_last_match'] = (
        df['home_result_flag'] + ' ' +
        df['home_score_int'].map(str) + '-' + df['away_score_int'].map(str) +
        ' vs. ' + df['away_team'] +
        ' (' + df['competition'] + ')'
    )
    df['away_last_match'] = (
        df['away_result_flag'] + ' ' +
        df['away_score_int'].map(str) + '-' + df['home_score_int'].map(str) +
        ' @ ' + df['home_team'] +
        ' (' + df['competition'] + ')'
    )

    lastmatch_home = df[['date', 'home_team', 'home_last_match']].copy()
    lastmatch_home.columns = ['date', 'name', 'last_match']
    lastmatch_away = df[['date', 'away_team', 'away_last_match']].copy()
    lastmatch_away.columns = ['date', 'name', 'last_match']
    lastmatch_df = pd.concat([lastmatch_home, lastmatch_away]).reset_index(drop=True)
    lastmatch_df['date'] = pd.to_datetime(lastmatch_df['date'])
    # Snapshot season label (calendar year string) — matches the snapshot's
    # `season` column so the downstream merge_asof can scope carry-forward to
    # within the same calendar year. Without this, a team's Dec 2024 match
    # would still show as their last_match in Mar 2025 snapshots.
    lastmatch_df['season'] = lastmatch_df['date'].dt.year.astype(str)

    # ---- 4. Snapshot season + grouped_date_id ----
    df['snapshot_season']  = df['date'].apply(date_to_snapshot_season)
    df['grouped_date_id']  = df.groupby('date').ngroup() + 1

    # ---- 5. Rated-team set per calendar year ----
    # MLS-only universe — every game is rated-vs-rated by construction.
    league_games = df[df['competition'].isin(LEAGUE_COMPETITIONS)]
    df['_rated'] = True
    print(f"  {len(df):,} MLS games (all rated-vs-rated)")

    # ---- 6. Rolling Massey rating loop ----
    print("\nRunning rolling Massey ratings...")
    max_date_id = int(df['grouped_date_id'].max())

    try:
        cobi_df = pd.read_csv('cobi_ratings.csv.gz')
        all_ids = sorted(cobi_df['ranking_id'].unique())
        if len(all_ids) > RECOMPUTE_TAIL_DAYS:
            tail_threshold = all_ids[-RECOMPUTE_TAIL_DAYS]
            n_dropped = int((cobi_df['ranking_id'] >= tail_threshold).sum())
            cobi_df = cobi_df[cobi_df['ranking_id'] < tail_threshold].copy()
            print(f"  Re-processing tail {RECOMPUTE_TAIL_DAYS} game-days "
                  f"({n_dropped:,} rows dropped from cache for late-arriving-data refresh)")
        max_ranked = int(cobi_df['ranking_id'].max()) if not cobi_df.empty else -1
        min_ranked = int(cobi_df['ranking_id'].min()) if not cobi_df.empty else -1
        if max_ranked >= 0:
            print(f"Existing ratings: ranking_ids {min_ranked} → {max_ranked}")
    except FileNotFoundError:
        cobi_df = pd.DataFrame(columns=[
            'ranking_id', 'ranking_date', 'season', 'name', 'rating', 'rank', 'games_played'
        ])
        max_ranked = -1
        min_ranked = -1
        print("No existing ratings — running full history from scratch.")

    last_printed_ym = None

    for i in range(1, max_date_id + 1):
        if min_ranked <= i <= max_ranked:
            continue

        current_date = df.loc[df['grouped_date_id'] == i, 'date'].max()

        working = df.loc[
            (df['grouped_date_id'] >= i - window_game_days + 1) &
            (df['grouped_date_id'] <= i) &
            (df['_rated'])
        ].copy()

        if len(working) < 10:
            continue

        working['game_days_ago'] = i - working['grouped_date_id']
        working['date_weight']   = 1 - (working['game_days_ago'] / window_game_days)

        # Drop draws (adj_margin_home == 0) from the solve to match the
        # prior rankit behavior — rankit ignored zero-margin rows by
        # default. Note: 0-0 regulation draws are dropped, but shootout-
        # decided 0-0 games have non-zero adj_margin (±shootout_margin)
        # so they participate.
        working = working[working['adj_margin_home'] != 0]
        if len(working) < 10:
            continue

        ym = current_date.strftime('%Y-%m')
        if ym != last_printed_ym:
            pct = round(100 * i / max_date_id)
            print(f"  Ratings: {current_date.strftime('%B %Y')} ({pct}%)")
            last_printed_ym = ym

        try:
            ranked = _solve_massey(working, WEIGHTING_MODE)

            if ranked['rating'].isna().any() or np.isinf(ranked['rating']).any():
                continue

            ranked['ranking_id']   = i
            ranked['ranking_date'] = current_date.date()
            ranked['season']       = date_to_snapshot_season(current_date)

            home_gp = working.groupby('home_team').size().reset_index(name='gp_home')
            away_gp = working.groupby('away_team').size().reset_index(name='gp_away')
            home_gp.columns = ['name', 'gp_home']
            away_gp.columns = ['name', 'gp_away']
            gp = pd.merge(home_gp, away_gp, on='name', how='outer').fillna(0)
            gp['games_played'] = (gp['gp_home'] + gp['gp_away']).astype(int)
            ranked = pd.merge(ranked, gp[['name', 'games_played']], on='name', how='left')
            ranked['games_played'] = ranked['games_played'].fillna(0).astype(int)

            cobi_df = pd.concat([cobi_df, ranked], axis=0, sort=False).reset_index(drop=True)
        except Exception as e:
            print(f"  [skip] date_id {i}: {e}")
            continue

    cobi_df.sort_values(['ranking_id', 'name'], inplace=True)
    cobi_df.drop_duplicates(keep='first', inplace=True)
    cobi_df['ranking_date'] = pd.to_datetime(cobi_df['ranking_date']).dt.date
    cobi_df.to_csv('cobi_ratings.csv.gz', index=False, compression='gzip')
    print(f"cobi_ratings.csv.gz saved ({len(cobi_df):,} rows)")

    # ---- 7. Standings ----
    print("\nComputing standings...")
    home_view = df[['season', 'competition', 'home_team', 'away_team',
                    'home_score_int', 'away_score_int', 'home_win']].copy()
    home_view.columns = ['season', 'competition', 'team', 'opponent', 'gf', 'ga', 'result']
    away_view = df[['season', 'competition', 'away_team', 'home_team',
                    'away_score_int', 'home_score_int', 'away_win']].copy()
    away_view.columns = ['season', 'competition', 'team', 'opponent', 'gf', 'ga', 'result']
    team_view = pd.concat([home_view, away_view], ignore_index=True)
    team_view['w'] = (team_view['result'] == 1.0 ).astype(int)
    team_view['d'] = (team_view['result'] == 0.5).astype(int)
    team_view['l'] = (team_view['result'] == 0.0 ).astype(int)

    standings = (
        team_view.groupby(['season', 'competition', 'team'])
        .agg(gp=('gf', 'count'), w=('w', 'sum'), d=('d', 'sum'),
             l=('l', 'sum'), gf=('gf', 'sum'), ga=('ga', 'sum'))
        .reset_index()
    )
    standings['gd']  = standings['gf'] - standings['ga']
    standings['pts'] = standings['w'] * 3 + standings['d']
    standings = standings.sort_values(
        ['season', 'competition', 'pts', 'gd', 'gf'],
        ascending=[True, True, False, False, False]
    )
    standings.to_csv('cobi_standings.csv', index=False)
    print(f"cobi_standings.csv saved ({len(standings):,} rows)")

    # ---- 7b. TROPHY DETECTION ----
    # MLS Cup: single game, played Nov-Dec. Last MLS match of each calendar
    # year IS MLS Cup. Gated on (today > last match + 7 days) so the trophy
    # only appears once the result is conclusive.
    print("\nDetecting trophies...")
    trophy_records = []  # rows of {year, team, honor}

    # No more MLS_CUP_OVERRIDES — 1996 + 2001 finals are now in
    # mls_early_historical.csv, and 2006's shootout_winner is patched at
    # CSV-load time via SHOOTOUT_OVERRIDES above.

    def _winner_loser_single(game):
        """Return (winner, loser) from a single-game final, honoring shootouts."""
        if game['home_score'] > game['away_score']:
            return game['home_team'], game['away_team']
        if game['away_score'] > game['home_score']:
            return game['away_team'], game['home_team']
        sw = game.get('shootout_winner')
        if pd.notna(sw) and sw:
            return sw, (game['away_team'] if sw == game['home_team'] else game['home_team'])
        return None, None

    mls_df = df[df['competition'] == 'MLS']
    if not mls_df.empty:
        for y, g in mls_df.groupby(mls_df['date'].dt.year):
            g = g.sort_values('date')
            last = g.iloc[-1]
            last_date = pd.to_datetime(last['date']).date()
            if (date.today() - last_date).days < 7:
                continue
            champ, ru = _winner_loser_single(last)
            if champ is None:
                continue
            trophy_records.append({'year': str(y), 'team': champ, 'honor': 'MLS Cup Champion'})
            trophy_records.append({'year': str(y), 'team': ru,    'honor': 'MLS Cup Runner-Up'})

    # Supporters' Shield: best regular-season point total. Decision Day =
    # the last date in Sept-Nov with >= 6 MLS games (the simultaneous
    # final-weekend pattern); anything after is playoffs. Gated 7 days
    # past Decision Day so an in-progress final weekend doesn't crown early.
    #
    # 1996-1999: Shield winner is derived from MLS_EARLY_STANDINGS (Wikipedia
    # season totals) since our gap-fill source doesn't preserve the SOW
    # signal needed for era-accurate pts.
    def _regular_season_end_date(season_games):
        """Return the date of Decision Day (regular-season finale) or None."""
        if season_games.empty:
            return None
        # Restrict to Sept-Nov to skip mid-season busy days
        late = season_games[
            (season_games['date'].dt.month >= 9) &
            (season_games['date'].dt.month <= 11)
        ]
        if late.empty:
            return season_games['date'].dt.date.max()
        daily = late.groupby(late['date'].dt.date).size()
        big_days = daily[daily >= 6]
        return big_days.index.max() if not big_days.empty else daily.idxmax()

    def _points(row, team):
        """3 for win (incl. shootout), 1 for draw, 0 for loss."""
        h, a = row['home_team'], row['away_team']
        hs, as_ = row['home_score'], row['away_score']
        if hs > as_:
            return 3 if team == h else 0
        if as_ > hs:
            return 3 if team == a else 0
        sw = row.get('shootout_winner')
        if pd.notna(sw) and sw:
            return 3 if sw == team else 0
        return 1

    if not mls_df.empty:
        for y, g in mls_df.groupby(mls_df['date'].dt.year):
            year_int = int(y)
            # Shootout era: derive Shield from static Wikipedia standings.
            era_rows = [(t, s) for (yr, t), s in MLS_EARLY_STANDINGS.items() if yr == year_int]
            if era_rows:
                # Tiebreakers: pts → total wins (reg + SOW) → GD → GF
                era_rows.sort(key=lambda kv: (kv[1]['pts'], kv[1]['w'] + kv[1]['sow'],
                                              kv[1]['gf'] - kv[1]['ga'], kv[1]['gf']),
                              reverse=True)
                champ, _ = era_rows[0]
                ru,    _ = era_rows[1]
                trophy_records.append({'year': str(y), 'team': champ, 'honor': "Supporters Shield Champion"})
                trophy_records.append({'year': str(y), 'team': ru,    'honor': "Supporters Shield Runner-Up"})
                continue
            ds_date = _regular_season_end_date(g)
            if ds_date is None:
                continue
            if (date.today() - ds_date).days < 7:
                continue
            reg = g[g['date'].dt.date <= ds_date]
            if reg.empty:
                continue
            # Per-team running totals: points, wins, goal differential, goals for.
            # MLS Shield tiebreakers: pts → wins → GD → GF.
            stats = {}  # team → [pts, wins, gd, gf]
            for _, r in reg.iterrows():
                h, a, hs, as_ = r['home_team'], r['away_team'], r['home_score'], r['away_score']
                sw = r.get('shootout_winner')
                for team, gf, ga in ((h, hs, as_), (a, as_, hs)):
                    s = stats.setdefault(team, [0, 0, 0, 0])
                    s[2] += gf - ga
                    s[3] += gf
                    if gf > ga:
                        s[0] += 3; s[1] += 1
                    elif gf < ga:
                        pass
                    elif pd.notna(sw) and sw == team:
                        s[0] += 3; s[1] += 1
                    elif pd.notna(sw) and sw:
                        pass
                    else:
                        s[0] += 1
            ranked = sorted(stats.items(),
                            key=lambda kv: (kv[1][0], kv[1][1], kv[1][2], kv[1][3]),
                            reverse=True)
            if len(ranked) < 2:
                continue
            champ, _ = ranked[0]
            ru,    _ = ranked[1]
            trophy_records.append({'year': str(y), 'team': champ, 'honor': "Supporters Shield Champion"})
            trophy_records.append({'year': str(y), 'team': ru,    'honor': "Supporters Shield Runner-Up"})

    trophy_df = pd.DataFrame(trophy_records) if trophy_records else \
                pd.DataFrame(columns=['year', 'team', 'honor'])
    trophy_df.to_csv('cobi_trophies.csv', index=False)
    print(f"cobi_trophies.csv saved ({len(trophy_df):,} rows)")
    if not trophy_df.empty:
        # Quick summary by honor
        print(trophy_df[trophy_df['honor'].str.endswith('Champion')]
              .groupby('honor').size().sort_index().to_string())

    # ---- 8. Final output ----
    print("\nBuilding final output...")
    final_df = cobi_df.copy()
    final_df.rename(columns={'ranking_date': 'date'}, inplace=True)
    final_df['date'] = pd.to_datetime(final_df['date'])

    latest_id = final_df['ranking_id'].max()
    final_df['most_recent'] = np.where(final_df['ranking_id'] == latest_id, 1, 0)

    # Last match string via merge_asof — scoped by (name, season) so a team's
    # last_match resets to empty at the start of each new calendar year.
    # Cast both date columns to datetime64[ns] explicitly: newer pandas
    # picks resolution per-source (sometimes [s], sometimes [us]) and
    # merge_asof refuses mismatched resolutions even though both are dt64.
    final_df['season'] = final_df['season'].astype(str)
    final_df['date']   = pd.to_datetime(final_df['date']).astype('datetime64[ns]')
    lastmatch_sorted = lastmatch_df.sort_values('date').copy()
    lastmatch_sorted['season'] = lastmatch_sorted['season'].astype(str)
    lastmatch_sorted['date']   = pd.to_datetime(lastmatch_sorted['date']).astype('datetime64[ns]')
    final_df = final_df.sort_values('date')
    final_df = pd.merge_asof(
        final_df, lastmatch_sorted.rename(columns={'date': 'match_date'}),
        left_on='date', right_on='match_date',
        by=['name', 'season'], direction='backward'
    )
    final_df['last_match']      = final_df['last_match'].fillna('')
    final_df['last_match_date'] = final_df['match_date'].dt.date
    final_df.drop(columns=['match_date'], inplace=True)
    final_df['date'] = final_df['date'].dt.date
    final_df['is_game_day'] = np.where(final_df['date'] == final_df['last_match_date'], 1, 0)
    final_df.rename(columns={'name': 'team'}, inplace=True)

    # Honor columns ('Champion'/'Runner-Up'/'').
    HONOR_COLS = {
        'MLS Cup':           'mls_cup_finish',
        'Supporters Shield': 'supporters_shield_finish',
    }
    for honor_label, col in HONOR_COLS.items():
        sub = trophy_df[trophy_df['honor'].str.startswith(honor_label)].copy()
        if sub.empty:
            final_df[col] = ''
            continue
        sub[col] = sub['honor'].str.replace(f'{honor_label} ', '', regex=False)
        sub = sub[['year', 'team', col]].rename(columns={'year': 'season'})
        final_df = pd.merge(final_df, sub, on=['season', 'team'], how='left')
        final_df[col] = final_df[col].fillna('')

    final_df = final_df[[
        'ranking_id', 'date', 'season', 'team',
        'rating', 'rank', 'games_played',
        'last_match_date', 'last_match', 'is_game_day', 'most_recent',
        'mls_cup_finish', 'supporters_shield_finish',
    ]]
    final_df.sort_values(['ranking_id', 'rank'], inplace=True)
    final_df.drop_duplicates(keep='first', inplace=True)
    final_df = final_df[final_df['games_played'] >= min_games]
    final_df['rank'] = (
        final_df.groupby('ranking_id')['rating']
        .rank(ascending=False, method='min')
        .astype(int)
    )
    final_df.to_csv('cobi_ratings_final.csv', index=False)
    print(f"cobi_ratings_final.csv saved ({len(final_df):,} rows)")

    # Spot check
    print("\nMost recent COBI ratings (top 20):")
    latest = final_df[final_df['most_recent'] == 1].head(20)
    print(latest[['rank', 'team', 'rating', 'games_played',
                  'last_match_date', 'last_match']].to_string(index=False))


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    scrape = '--no-scrape' not in sys.argv
    run_pipeline(scrape=scrape)
