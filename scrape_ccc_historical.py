# ============================================================
# COBI - Pre-2008 CONCACAF Champions Cup gap-fill (1996-2007)
# Source: Wikipedia per-season pages
# Produces: ccc_historical.csv
# ============================================================
#
# ESPN coverage of CONCACAF Champions Cup/League begins in 2008.
# This script scrapes the per-season Wikipedia pages for 1996-2007 to
# capture the cross-league bridge (MLS vs Liga MX, etc.) in the early
# era.
#
# Best-effort: the hCalendar `<div class="footballbox">` pattern is
# consistent across years on Wikipedia, but per-match dates are missing
# in several early-season pages where editors didn't fill in dated
# match boxes. We drop undated matches rather than guess. Result is
# strongest from 2003 onward; 1996-2002 captures fewer matches but the
# knockout/final stage is typically dated.

import csv
import re
import time
from datetime import datetime
from urllib.parse import unquote
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

UA = 'cobi-data/1.0 (https://github.com/fakeronjan/cobi; one-time CCC historical backfill)'
WIKI_BASE = 'https://en.wikipedia.org'

SEASONS = {y: f"/wiki/{y}_CONCACAF_Champions%27_Cup" for y in range(1996, 2008)}

OUT_PATH = 'ccc_historical.csv'

# Team-name fixups: Wikipedia → ESPN/canonical. Built iteratively after
# inspecting parser output. The important ones are MLS clubs (so they
# merge with the rest of the COBI data) and the major Liga MX clubs.
TEAM_FIXUPS = {
    # MLS — Wikipedia uses full names that mostly match ESPN
    'D.C. United':                      'D.C. United',
    'New York Red Bulls':               'Red Bull New York',
    'Los Angeles Galaxy':               'LA Galaxy',
    'LA Galaxy':                        'LA Galaxy',
    'Houston Dynamo':                   'Houston Dynamo FC',
    'Columbus Crew':                    'Columbus Crew',
    'Chicago Fire':                     'Chicago Fire FC',
    'New England Revolution':           'New England Revolution',
    'Sporting Kansas City':             'Sporting Kansas City',
    'Kansas City Wizards':              'Sporting Kansas City',  # franchise continuity
    'Kansas City Wiz':                  'Sporting Kansas City',
    'Real Salt Lake':                   'Real Salt Lake',
    'San Jose Earthquakes':             'San Jose Earthquakes',
    'FC Dallas':                        'FC Dallas',
    'Dallas Burn':                      'FC Dallas',  # rebrand
    'Tampa Bay Mutiny':                 'Tampa Bay Mutiny',
    'Miami Fusion':                     'Miami Fusion',
    'Toronto FC':                       'Toronto FC',
    # Liga MX — Wikipedia tends to use full club names with prefixes
    'Club América':                     'América',
    'Club Deportivo Guadalajara':       'Guadalajara',
    'C.D. Guadalajara':                 'Guadalajara',
    'Guadalajara':                      'Guadalajara',
    'Cruz Azul':                        'Cruz Azul',
    'Club Necaxa':                      'Necaxa',
    'Necaxa':                           'Necaxa',
    'Pachuca':                          'Pachuca',
    'C.F. Pachuca':                     'Pachuca',
    'Toluca':                           'Toluca',
    'Deportivo Toluca F.C.':            'Toluca',
    'Tigres UANL':                      'Tigres UANL',
    'C.F. Monterrey':                   'Monterrey',
    'Monterrey':                        'Monterrey',
    'Atlante':                          'Atlante',
    'Atlante F.C.':                     'Atlante',
    'Pumas UNAM':                       'Pumas UNAM',
    'Club Universidad Nacional':        'Pumas UNAM',
    'Santos Laguna':                    'Santos Laguna',
    'Club Santos Laguna':               'Santos Laguna',
    'Atlas':                            'Atlas',
    'Club León':                        'León',
}


def fetch(url):
    req = Request(url, headers={'User-Agent': UA})
    with urlopen(req, timeout=30) as r:
        return r.read().decode('utf-8', errors='replace')


RE_DATETIME = re.compile(r'(\d{4}-\d{2}-\d{2})')
RE_SCORE    = re.compile(r'(\d+)\s*[–\-]\s*(\d+)')
# Penalty shootout: "(X-Y p)" / "X-Y on penalties" patterns
RE_PEN_KEYWORD = re.compile(r'(?:Penalties|penalt|shootout).*?(\d+)\s*[–\-]\s*(\d+)', re.IGNORECASE | re.DOTALL)
RE_PEN_SUFFIX  = re.compile(r'(?:^|\s|\()(\d+)\s*[–\-]\s*(\d+)\s*(?:p|pen|pens)\b', re.IGNORECASE)


def _team_text(th):
    """Pull team name from an fhome/faway TH cell, skipping flag links."""
    name_span = th.find(itemprop='name')
    scope = name_span if name_span else th
    for a in scope.find_all('a'):
        if a.find_parent(class_='flagicon'):
            continue
        title = a.get('title', '').strip()
        if title:
            return title
        return a.get_text(strip=True)
    return scope.get_text(' ', strip=True)


def _parse_fdate_text(s, year_hint):
    """Parse a Wikipedia fdate string. Tries multiple formats:
    'March 13, 2002', '13 March 2002', '21 February 2007'. Returns
    YYYY-MM-DD or None on failure."""
    s = s.strip()
    if not s:
        return None
    for fmt in ('%B %d, %Y', '%d %B %Y', '%B %Y'):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    # Some pages omit year — append year_hint and retry
    if not re.search(r'\b\d{4}\b', s):
        for fmt in ('%B %d', '%d %B'):
            try:
                d = datetime.strptime(s, fmt).date().replace(year=year_hint)
                return d.isoformat()
            except ValueError:
                continue
    return None


def parse_matches(html, season_year):
    soup = BeautifulSoup(html, 'html.parser')
    out = []
    for box in soup.find_all('div', class_='footballbox'):
        match_date = None
        # Preferred: itemprop="startDate" datetime attr (post-2010 pages)
        time_el = box.find('time', attrs={'itemprop': 'startDate'})
        if time_el and time_el.get('datetime'):
            m = RE_DATETIME.search(time_el['datetime'])
            if m:
                match_date = m.group(1)
        # Fallback: <div class="fdate"> plain text (older pages)
        if not match_date:
            fdate_div = box.find('div', class_='fdate')
            if fdate_div:
                match_date = _parse_fdate_text(fdate_div.get_text(' ', strip=True), season_year)
        if not match_date:
            continue

        fevent = box.find('table', class_='fevent')
        if not fevent:
            continue
        home_th = fevent.find('th', class_='fhome')
        away_th = fevent.find('th', class_='faway')
        score_th = fevent.find('th', class_='fscore')
        if not (home_th and away_th and score_th):
            continue

        home_raw = _team_text(home_th)
        away_raw = _team_text(away_th)
        score_text = score_th.get_text(' ', strip=True)
        sm = RE_SCORE.search(score_text)
        if not sm:
            continue
        hs, as_ = int(sm.group(1)), int(sm.group(2))

        # Penalty detection — only meaningful when score is tied
        shootout_winner = None
        ps_h = ps_a = None
        penalties = False
        if hs == as_:
            full_text = fevent.get_text(' ', strip=True)
            pm = RE_PEN_KEYWORD.search(full_text) or RE_PEN_SUFFIX.search(full_text)
            if pm:
                ph, pa = int(pm.group(1)), int(pm.group(2))
                if ph != pa:
                    penalties = True
                    ps_h, ps_a = ph, pa
                    shootout_winner = home_raw if ph > pa else away_raw

        home = TEAM_FIXUPS.get(home_raw, home_raw)
        away = TEAM_FIXUPS.get(away_raw, away_raw)

        out.append({
            'date':            match_date,
            'season':          str(season_year),
            'competition':     'CONCACAF CL',
            'league_match':    False,
            'home_team':       home,
            'away_team':       away,
            'home_score':      hs,
            'away_score':      as_,
            'home_pen':        ps_h,
            'away_pen':        ps_a,
            'penalties':       penalties,
            'shootout_winner': TEAM_FIXUPS.get(shootout_winner, shootout_winner) if shootout_winner else None,
            'neutral':         False,
            'venue':           '',
            'event_id':        f'wiki-ccc-{season_year}-{match_date}-{home}-{away}',
            'stage':           '',
        })
    return out


def main():
    all_rows = []
    for year, slug in SEASONS.items():
        url = WIKI_BASE + slug
        print(f"  [ccc-wiki] {year}: {url}")
        try:
            html = fetch(url)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
        rows = parse_matches(html, year)
        print(f"    {len(rows)} dated matches")
        all_rows.extend(rows)
        time.sleep(1.0)  # polite

    # Dedupe by (date, home_team, away_team)
    seen = set()
    unique = []
    for r in all_rows:
        k = (r['date'], r['home_team'], r['away_team'])
        if k in seen:
            continue
        seen.add(k)
        unique.append(r)

    fieldnames = list(unique[0].keys()) if unique else [
        'date', 'season', 'competition', 'league_match',
        'home_team', 'away_team', 'home_score', 'away_score',
        'home_pen', 'away_pen', 'penalties', 'shootout_winner',
        'neutral', 'venue', 'event_id', 'stage',
    ]
    with open(OUT_PATH, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(unique)
    print(f"\n[done] {len(unique):,} matches → {OUT_PATH}")


if __name__ == '__main__':
    main()
