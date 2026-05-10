# ============================================================
# COBI - Pre-2002 Liga MX gap-fill (1996-97 through 2001-02)
# Source: Wikipedia per-season pages
# Produces: ligamx_early_historical.csv
# ============================================================
#
# ESPN Liga MX coverage starts in 2002. Wikipedia per-season pages
# (e.g., "1996-97 Mexican Primera División season") expose dated match
# boxes for the Liguilla (playoff) rounds via hCalendar microformat.
# Regular-season matches typically live in unstructured score grids
# that are too brittle to scrape; we accept that as a known thinness.
#
# Result: high-value playoff matches (quarterfinals, semifinals, finals)
# captured for 1996-97 through 2000-01. Partially overlaps with ESPN's
# 2001-02 Clausura coverage; cobi.py dedupes on (date, home, away).

import csv
import re
import time
from datetime import datetime
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

UA = 'cobi-data/1.0 (https://github.com/fakeronjan/cobi; one-time Liga MX historical backfill)'
WIKI_BASE = 'https://en.wikipedia.org'

# Liga MX restructured its season format mid-1990s. URL patterns:
# - 1996-97, 1997-98: combined YYYY-YY page documenting both Invierno+Verano
# - 1998 onwards: separate short-tournament pages (Invierno YYYY / Verano YYYY)
#   Eventually renamed Apertura/Clausura, but ESPN takes over before that.
# Each entry is (label_for_csv, wiki_slug).
SEASONS = [
    ('1996-97', '/wiki/1996%E2%80%9397_Mexican_Primera_Divisi%C3%B3n_season'),
    ('1997-98', '/wiki/1997%E2%80%9398_Mexican_Primera_Divisi%C3%B3n_season'),
    # Short-tournament pages — each tagged to the YYYY-YY season it belongs to.
    # Invierno YYYY (autumn) starts season Y-(Y+1); Verano YYYY (spring) ends it.
    ('1998-99', '/wiki/Primera_Divisi%C3%B3n_de_M%C3%A9xico_Invierno_1998'),
    ('1998-99', '/wiki/Primera_Divisi%C3%B3n_de_M%C3%A9xico_Verano_1999'),
    ('1999-00', '/wiki/Primera_Divisi%C3%B3n_de_M%C3%A9xico_Invierno_1999'),
    ('1999-00', '/wiki/Primera_Divisi%C3%B3n_de_M%C3%A9xico_Verano_2000'),
    ('2000-01', '/wiki/Primera_Divisi%C3%B3n_de_M%C3%A9xico_Invierno_2000'),
    ('2000-01', '/wiki/Primera_Divisi%C3%B3n_de_M%C3%A9xico_Verano_2001'),
    ('2001-02', '/wiki/Primera_Divisi%C3%B3n_de_M%C3%A9xico_Invierno_2001'),
    # Verano 2002 omitted: ESPN coverage starts Jan 2002 and covers it directly.
]

OUT_PATH = 'ligamx_early_historical.csv'

TEAM_FIXUPS = {
    # Wikipedia → canonical (matches ESPN where possible)
    'Club América':                     'América',
    'América':                          'América',
    'Club Deportivo Guadalajara':       'Guadalajara',
    'C.D. Guadalajara':                 'Guadalajara',
    'Guadalajara':                      'Guadalajara',
    'Cruz Azul':                        'Cruz Azul',
    'Necaxa':                           'Necaxa',
    'Club Necaxa':                      'Necaxa',
    'Pachuca':                          'Pachuca',
    'C.F. Pachuca':                     'Pachuca',
    'Toluca':                           'Toluca',
    'Deportivo Toluca':                 'Toluca',
    'Deportivo Toluca F.C.':            'Toluca',
    'Tigres UANL':                      'Tigres UANL',
    'Monterrey':                        'Monterrey',
    'C.F. Monterrey':                   'Monterrey',
    'Atlante':                          'Atlante',
    'Atlante F.C.':                     'Atlante',
    'Pumas UNAM':                       'Pumas UNAM',
    'Club Universidad Nacional':        'Pumas UNAM',
    'U.N.A.M.':                         'Pumas UNAM',
    'Santos Laguna':                    'Santos Laguna',
    'Club Santos Laguna':               'Santos Laguna',
    'Atlas':                            'Atlas',
    'Club Atlas':                       'Atlas',
    'León':                             'León',
    'Club León':                        'León',
    'Tijuana':                          'Tijuana',
    'Querétaro':                        'Querétaro',
    'Puebla':                           'Puebla',
    'Veracruz':                         'Veracruz',
    'Morelia':                          'Atlético Morelia',
    'Monarcas Morelia':                 'Atlético Morelia',
    'Atlético Morelia':                 'Atlético Morelia',
}


def fetch(url):
    req = Request(url, headers={'User-Agent': UA})
    with urlopen(req, timeout=30) as r:
        return r.read().decode('utf-8', errors='replace')


RE_DATETIME = re.compile(r'(\d{4}-\d{2}-\d{2})')
RE_SCORE    = re.compile(r'(\d+)\s*[–\-]\s*(\d+)')
RE_PEN_KEYWORD = re.compile(r'(?:Penalties|penalt|shootout).*?(\d+)\s*[–\-]\s*(\d+)', re.IGNORECASE | re.DOTALL)
RE_PEN_SUFFIX  = re.compile(r'(?:^|\s|\()(\d+)\s*[–\-]\s*(\d+)\s*(?:p|pen|pens)\b', re.IGNORECASE)


def _team_text(th):
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
    s = s.strip()
    if not s:
        return None
    for fmt in ('%B %d, %Y', '%d %B %Y', '%B %Y'):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    if not re.search(r'\b\d{4}\b', s):
        for fmt in ('%B %d', '%d %B'):
            try:
                d = datetime.strptime(s, fmt).date().replace(year=year_hint)
                return d.isoformat()
            except ValueError:
                continue
    return None


def parse_matches(html, season_label):
    soup = BeautifulSoup(html, 'html.parser')
    season_year = int(season_label[:4])
    out = []
    for box in soup.find_all('div', class_='footballbox'):
        match_date = None
        time_el = box.find('time', attrs={'itemprop': 'startDate'})
        if time_el and time_el.get('datetime'):
            m = RE_DATETIME.search(time_el['datetime'])
            if m:
                match_date = m.group(1)
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
            'season':          season_label,
            'competition':     'Liga MX',
            'league_match':    True,
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
            'event_id':        f'wiki-mx-{season_label}-{match_date}-{home}-{away}',
            'stage':           'Liguilla',
        })
    return out


def main():
    all_rows = []
    for season_label, slug in SEASONS:
        url = WIKI_BASE + slug
        print(f"  [ligamx-wiki] {season_label}: {url}")
        try:
            html = fetch(url)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
        rows = parse_matches(html, season_label)
        print(f"    {len(rows)} matches")
        all_rows.extend(rows)
        time.sleep(1.0)

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
