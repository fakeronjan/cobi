"""Real MLS playoff seeds for the current season -> mls_playoff_seeds.json
(read by title_odds.REAL_SEEDS). Runs daily; does nothing until the regular
season is over.

Source: the season's Wikipedia playoff bracket (MLS's last tiebreaker,
disciplinary points, isn't in any feed we use), checked against the
standings and the real playoff matches before anything is written.
"""
# ── Shared core (same in every fleet site's playoff_seeds.py) ─────────────────
# Once a regular season is over, the playoff sims seed from the real seeds
# (playoff_sim.REAL_SEEDS), never their own tiebreak estimate. run() fetches
# them, checks them and writes them; seeds missing or failing a check give a
# warning for GRACE_DAYS after the regular season, then the run fails.
#
# Checks: every seed filled once and every team known; within each seeding
# tier, seeds follow the standings (a source can only differ from our own
# order where records are level); and once real playoff games exist, every
# series between two seeded teams of the same group was opened at the
# better seed.
import json
import os
import re
import urllib.parse
import urllib.request

import pandas as pd

GRACE_DAYS = 2
# Wikipedia asks for a descriptive agent with contact details; ESPN and the
# league APIs refuse agents with an email in them, so they get a plain one.
_UA_WIKI = {'User-Agent': 'fakeronjan-sports/1.0 (rjsikdar@gmail.com)'}
_UA = {'User-Agent': 'Mozilla/5.0'}


def get_json(url):
    ua = _UA_WIKI if 'wikipedia.org' in url else _UA
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=ua), timeout=30))


def wikitext(title):
    d = get_json('https://en.wikipedia.org/w/api.php?' + urllib.parse.urlencode(
        {'action': 'parse', 'page': title, 'prop': 'wikitext', 'format': 'json', 'formatversion': 2,
         'redirects': 1}))
    return d['parse']['wikitext'] if 'parse' in d else None


def bracket_links(text):
    """{linked page name: seed label} from a Wikipedia bracket template's
    RDn-seedXX / RDn-teamXX pairs (first label seen per team)."""
    seeds = {}
    sd = {(m.group(1), int(m.group(2))): m.group(3)
          for m in re.finditer(r'\|\s*RD(\d+)-seed0*(\d+)\s*=\s*([^\n|]*)', text)}
    for m in re.finditer(r'\|\s*RD(\d+)-team0*(\d+)\s*=\s*([^\n]*)', text):
        lab = re.sub(r'[^\w]', '', sd.get((m.group(1), int(m.group(2))), ''))
        link = re.search(r'\[\[([^\]|]+)', m.group(3))
        if not lab or not link:
            continue
        name = re.sub(r'^\d{4}(?:[–-]\d{2,4})?\s+', '', link.group(1).strip())
        seeds.setdefault(re.sub(r'\s+season$', '', name), lab)
    return seeds


def _problem(seeds, teams, tiers, rec, ps_games):
    """seeds: {group: [team, ...]} best first. tiers: [(group, [seed numbers])].
    rec: {team: standings value, higher = better}. ps_games: [(home, away,
    ...)] in date order. Returns a description of the first failed check, or None."""
    seen = [t for lst in seeds.values() for t in lst]
    if any(t not in teams for t in seen) or len(set(seen)) != len(seen):
        return f"unknown or repeated teams: {seen}"
    for g, nums in tiers:
        lst = seeds.get(g, [])
        if len(lst) < max(nums):
            return f"{g} has {len(lst)} seeds, expected {max(nums)}"
        vals = [rec.get(lst[k - 1], 0) for k in nums]
        if any(a < b - 1e-9 for a, b in zip(vals, vals[1:])):
            return f"{g} seeds {nums} don't follow the standings: {[lst[k - 1] for k in nums]} {vals}"
    rank = {t: (g, k) for g, lst in seeds.items() for k, t in enumerate(lst)}
    first = {}
    for x in ps_games:
        first.setdefault(frozenset(x[:2]), x[0])
    for pair, host in first.items():
        x, y = sorted(pair)
        if x in rank and y in rank and rank[x][0] == rank[y][0]:
            if host != min(pair, key=lambda t: rank[t][1]):
                return f"{' vs '.join(pair)} opened at {host}, not the better seed"
    return None


def run(path, season, rs_end, fetch, teams, tiers, rec, ps_games, today=None, label=''):
    """rs_end: last regular-season date (None while it's still going).
    fetch(stored) -> seeds dict (may refine stored ones, e.g. from play-in
    games) or None when the source doesn't have them yet."""
    if rs_end is None:
        return None
    data = json.load(open(path)) if os.path.exists(path) else {}
    stored = data.get(str(season))
    problem = None
    try:
        seeds = fetch(stored)
        if seeds is None:
            problem = 'the source has no seeds yet'
        else:
            problem = _problem(seeds, teams, tiers, rec, ps_games)
    except Exception as e:                       # network, parsing
        seeds, problem = None, f'{type(e).__name__}: {e}'
    if problem is None:
        if seeds != stored:
            data[str(season)] = seeds
            json.dump(dict(sorted(data.items())), open(path, 'w'), indent=1)
            print(f"  {season} {label}playoff seeds -> {path}: {seeds}")
        return seeds
    if stored is not None and _problem(stored, teams, tiers, rec, ps_games) is None:
        print(f"::warning::{season} {label}seed refresh failed ({problem}); keeping the stored seeds")
        return stored
    days = ((today or pd.Timestamp.now()).normalize() - pd.Timestamp(rs_end).normalize()).days
    msg = f"{season} {label}playoff seeds not usable yet: {problem}"
    if days > GRACE_DAYS:
        raise RuntimeError(msg + f" ({days} days after the regular season)")
    print(f"::warning::{msg}")
    return None


# ── MLS ──────────────────────────────────────────────────────────────────────
import unicodedata

SEEDS_JSON = 'mls_playoff_seeds.json'
N_SEEDS = 9                       # per conference since 2023
ALIASES = {'New York Red Bulls': 'Red Bull New York', 'Los Angeles FC': 'LAFC'}


def season_state(games_csv='all_club_games.csv', sched_csv='mls_schedule.csv'):
    """(season, rs_end or None, teams, points by team, playoff matches as
    (home, away) in date order)."""
    g = pd.read_csv(games_csv, low_memory=False)
    g = g[g['competition'] == 'MLS'].assign(date=lambda x: pd.to_datetime(x['date']))
    season = int(g['date'].dt.year.max())
    g = g[(g['date'].dt.year == season) & g['home_score'].notna()].sort_values('date', kind='stable')
    rs = g[g['stage'] == 'regular-season']
    sched = pd.read_csv(sched_csv)
    left = sched[(sched['stage'] == 'regular-season') & (pd.to_datetime(sched['date']).dt.year == season)]
    teams = set(rs['home_team']) | set(rs['away_team'])
    rs_end = rs['date'].max() if len(rs) and not len(left) else None
    pts = {}
    for h, a, hs, as_ in rs[['home_team', 'away_team', 'home_score', 'away_score']].itertuples(index=False):
        pts[h] = pts.get(h, 0) + (3 if hs > as_ else 1 if hs == as_ else 0)
        pts[a] = pts.get(a, 0) + (3 if as_ > hs else 1 if hs == as_ else 0)
    po = g[g['stage'] != 'regular-season']
    return season, rs_end, teams, pts, list(zip(po['home_team'], po['away_team']))


def _norm(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
    return ' '.join(w for w in re.sub(r'[^a-z ]', ' ', s).split() if w not in ('fc', 'sc', 'cf'))


def _match(name, teams):
    name = ALIASES.get(name, name)
    hit = [t for t in teams if _norm(t) == _norm(name)]
    if len(hit) != 1:
        hit = [t for t in teams if _norm(t).startswith(_norm(name)) or _norm(name).startswith(_norm(t))]
    if len(hit) != 1:
        raise ValueError(f"can't match {name!r} to one team: {hit}")
    return hit[0]


def wiki_seeds(season, teams):
    text = wikitext(f'{season} MLS Cup playoffs')
    if not text:
        return None
    by = {'East': {}, 'West': {}}
    for t, lab in bracket_links(text).items():
        if lab[:1] in 'EW' and lab[1:].isdigit():
            by['East' if lab[0] == 'E' else 'West'][int(lab[1:])] = _match(t, teams)
    if any(sorted(v) != list(range(1, N_SEEDS + 1)) for v in by.values()):
        return None                                     # bracket not filled in yet
    return {c: [v[k] for k in sorted(v)] for c, v in by.items()}


def main(today=None):
    season, rs_end, teams, pts, ps = season_state()
    run(SEEDS_JSON, season, rs_end, lambda stored: wiki_seeds(season, teams), teams,
        [(c, list(range(1, N_SEEDS + 1))) for c in ('East', 'West')], pts, ps,
        today=today, label='MLS ')


if __name__ == '__main__':
    main()
