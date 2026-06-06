# ============================================================
# COBI - Early MLS gap-fill (1996-2001)
# Source: github.com/footballcsv/major-league-soccer
# Produces: mls_early_historical.csv
# ============================================================
#
# ESPN coverage starts 2002. footballcsv/major-league-soccer covers
# 1996-2016 in CSV format. This script downloads & parses the 1996-2001
# slice and writes a unified-schema CSV that cobi.py merges into the
# master match table.

import csv
import io
import re
from datetime import datetime

import pandas as pd
import requests

REPO_BASE = 'https://raw.githubusercontent.com/footballcsv/major-league-soccer/master'
EARLY_YEARS = list(range(1996, 2002))  # ESPN takes over from 2002

OUT_PATH = 'mls_early_historical.csv'


def _strip_team(raw):
    """Strip trailing ordinal counter like 'San Jose Earthquakes (1)' → 'San Jose Earthquakes'."""
    return re.sub(r'\s*\(\d+\)\s*$', '', raw or '').strip()


def _parse_date(raw):
    """'(Sat) 6 Apr 1996 (W14)' → date(1996, 4, 6). Returns None on failure."""
    if not raw:
        return None
    # Strip the leading day-of-week parens and trailing week-number parens
    s = re.sub(r'^\([A-Za-z]+\)\s*', '', raw.strip())
    s = re.sub(r'\s*\(W\d+\)\s*$', '', s).strip()
    for fmt in ('%d %b %Y', '%d %B %Y'):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _parse_score(raw):
    """'1-0' → (1, 0). Returns (None, None) on missing/malformed."""
    if not raw or '-' not in raw:
        return None, None
    parts = raw.replace('–', '-').split('-')
    if len(parts) != 2:
        return None, None
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except ValueError:
        return None, None


# Historical MLS clubs that need name normalization to align with ESPN.
# footballcsv uses modern-canonical names already (e.g., "Sporting Kansas City"
# even for the 1996 season when the club was "Kansas City Wiz"). This is fine -
# we want a single canonical name per franchise across eras anyway. ESPN-side
# fixups happen in cobi.py via a shared TEAM_NAME_MAP.
TEAM_FIXUPS = {
    # footballcsv → canonical (matches ESPN where possible)
    'New York Red Bulls':      'Red Bull New York',  # ESPN uses Red Bull New York
    'Tampa Bay Mutiny':        'Tampa Bay Mutiny',   # defunct, ESPN N/A
    'Miami Fusion':            'Miami Fusion',       # defunct, ESPN N/A
    'Chivas USA':              'Chivas USA',         # defunct, but ESPN had 'em
}


def fetch_year(year):
    url = f'{REPO_BASE}/{year}/1-mls.csv'
    print(f"  [mls-early] {year}: {url}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    reader = csv.DictReader(io.StringIO(r.text))
    rows = []
    for r_ in reader:
        date_ = _parse_date(r_.get('Date'))
        if not date_:
            continue
        ft = r_.get('FT') or ''
        hs, as_ = _parse_score(ft)
        if hs is None:
            continue

        team1 = _strip_team(r_.get('Team 1'))
        team2 = _strip_team(r_.get('Team 2'))
        team1 = TEAM_FIXUPS.get(team1, team1)
        team2 = TEAM_FIXUPS.get(team2, team2)

        # Penalty/shootout - footballcsv 'P' column carries penalty score like '4-3'
        p = r_.get('P') or ''
        ps_h, ps_a = _parse_score(p)
        penalties = ps_h is not None and ps_a is not None
        shootout_winner = None
        if penalties and hs == as_:
            if ps_h > ps_a:
                shootout_winner = team1
            elif ps_a > ps_h:
                shootout_winner = team2

        # Extra-time score (ET column) - for our purposes treat ET final
        # as the FT for margin (same as how ZIDANE handles AET in CL).
        # If ET present and different from FT, prefer ET-final.
        et = r_.get('ET') or ''
        et_h, et_a = _parse_score(et)
        if et_h is not None and et_a is not None:
            hs, as_ = et_h, et_a

        # Stage: footballcsv uses 'Regular' / 'Playoffs' / 'Final' - flag
        # all non-Regular as playoff matches (still league_match=True since
        # it's all the MLS season for our purposes).
        stage = (r_.get('Stage') or '').strip()

        rows.append({
            'date':            date_.isoformat(),
            'season':          str(year),
            'competition':     'MLS',
            'league_match':    True,
            'home_team':       team1,
            'away_team':       team2,
            'home_score':      hs,
            'away_score':      as_,
            'home_pen':        ps_h,
            'away_pen':        ps_a,
            'penalties':       penalties,
            'shootout_winner': shootout_winner,
            'neutral':         False,
            'venue':           '',
            'event_id':        f'fbcsv-mls-{year}-{date_.isoformat()}-{team1}-{team2}',
            'stage':           stage,
        })
    return rows


def main():
    all_rows = []
    for year in EARLY_YEARS:
        all_rows.extend(fetch_year(year))
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n[done] {len(df):,} matches → {OUT_PATH}")
    print(df.groupby('season').size().to_string())


if __name__ == '__main__':
    main()
