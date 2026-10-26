"""Daily Matchups: every MLS match of a day, previewed from the ratings going
into it - home / draw / away probabilities (the pick is the most likely of
the three), COBI's goal line and O/U, and the stakes (each club's playoff
and MLS Cup odds with a win, a draw, and a loss). After the match: the
final score and whether COBI's pick was right. The MLS port of DUNCAN's
Daily Matchups (duncan/daily_matchups.py).

Everything comes from the season sim (title_odds.py): one run of 10k
simulations from the last snapshot before the day. Its Poisson goals give
each match's three probabilities, the line (mean goal margin, to the
nearest 0.25; Pick'em when it names a different favorite than the win
probabilities) and the O/U (mean total goals, to the nearest 0.5). Playoff
matches can't end level (extra time and penalties), so they get a winner's
probability only, from the bracket's next game, and their line and O/U use
the same goal model. A day with no snapshot yet this season (opening day)
previews from last season's final ratings. 2019 on, like the Cup odds.
"""
import hashlib
import multiprocessing as _mp
import os
import pickle

import numpy as np
import pandas as pd

import title_odds

N_SIMS = 10_000                   # fleet standard since 2026-10-02


def _record(w, d, l):
    return f"{w}-{d}-{l}"


def _per_point(y, pts, ctrl):
    """(mean of y, change in y per extra standings point) for outcome y
    (per-sim bool) and the club's final points in each sim. As DUNCAN's
    value of one more win: regress on final points plus both clubs' drift
    offsets, so the slope is the value of luck alone."""
    y = y.astype(float)
    m = float(y.mean())
    X = np.column_stack([np.ones(len(y)), pts] + ([ctrl] if ctrl is not None else []))
    if np.ptp(pts) == 0:
        return m, 0.0
    return m, max(float(np.linalg.lstsq(X, y, rcond=None)[0][1]), 0.0)


def _raw_split(y, won):
    """Playoff matches: no drift left, so split the sims by the result."""
    y = y.astype(float)
    return (float(y[won].mean()) if won.any() else None,
            float(y[~won].mean()) if (~won).any() else None)


def _wdl(m, b, pw, pd_):
    """Mean m split into (win, draw, loss) for +b per point (3 / 1 / 0)."""
    clip = lambda v: min(max(v, 0.0), 1.0)
    ept = 3 * pw + pd_
    return clip(m + b * (3 - ept)), clip(m + b * (1 - ept)), clip(m - b * ept)


def _line(margin, fav_home):
    """Mean goal margin to the nearest 0.25, home's perspective; 0 when it
    names a different favorite than the win probabilities."""
    line = round(float(margin) * 4) / 4
    if line != 0 and (line > 0) != fav_home:
        return 0.0
    return line


def build_season(season, games, fixtures, ratings, conference_for, mu_games, kickoffs,
                 n_sims=N_SIMS, log=print):
    """games: MLS matches (date, stage, home_team, away_team, home_score,
    away_score, shootout_winner, event_id); fixtures: scheduled matches
    (date, home_team, away_team, stage, event_id) or None. ratings: (season,
    date, team, rating, rank, rating_o, rating_d) for this season and the
    last. kickoffs: {event_id: UTC ISO}. Returns the season's days."""
    g = games[games['date'].dt.year == season]
    if g.empty:
        return []
    snaps = {d: x.set_index('team') for d, x in ratings.groupby('date')}
    cur = sorted(d for d in snaps if int(snaps[d]['season'].iloc[0]) == season)
    prev = sorted(d for d in snaps if d not in set(cur) and d < (cur[0] if cur else pd.Timestamp.max))
    sim_ratings = {d: dict(zip(snaps[d].index, zip(snaps[d]['rating_o'], snaps[d]['rating_d']))) for d in cur}
    sim = title_odds.SeasonSim(season, g, fixtures, conference_for, sim_ratings, mu_games)
    rs_keys = set(zip(sim.rs['date'], sim.rs['home_team'], sim.rs['away_team']))
    names = title_odds.round_names(season)[0]
    h_home, h_away = title_odds.HFA * title_odds.OFF_SHARE, title_odds.HFA * (1 - title_odds.OFF_SHARE)

    rows = g[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'shootout_winner', 'stage', 'event_id']]
    if fixtures is not None and len(fixtures):
        fx = fixtures.assign(home_score=np.nan, away_score=np.nan, shootout_winner=np.nan)
        rows = pd.concat([rows, fx[rows.columns]], ignore_index=True)
    rows = rows[rows['home_team'].isin(sim.idx) & rows['away_team'].isin(sim.idx)]
    rows = rows.sort_values(['date', 'home_team'], kind='stable').reset_index(drop=True)
    days = sorted(set(rows['date']))
    unplayed_days = sorted(set(rows.loc[rows['home_score'].isna(), 'date']))
    preview_day = unplayed_days[0] if unplayed_days else None

    days_out, previewed = [], False
    for day in days:
        if previewed:
            break                                     # live season: preview only the next match day
        before = [d for d in cur if d < day]
        if before:
            snap_d = before[-1]
        elif prev:
            snap_d = prev[-1]                         # opening day: last season's final ratings
            sim.ratings[snap_d] = dict(zip(snaps[snap_d].index, zip(snaps[snap_d]['rating_o'], snaps[snap_d]['rating_d'])))
        else:
            continue
        rt = snaps[snap_d]
        cap = {}
        sim.odds_at(snap_d, n_sims=n_sims, capture=cap)
        idx = {t: i for i, t in enumerate(cap['teams'])}
        made, champ = cap['made'], cap['champ']
        rest = cap.get('rest')
        E, Psim = cap.get('E'), cap.get('P')
        value = {}

        def per_point(t, opp):
            if t not in value:
                it = idx[t]
                ctrl = None if E is None else E[:, [it, idx[opp]]]
                value[t] = (_per_point(made[:, it], Psim[:, it], ctrl), _per_point(champ == it, Psim[:, it], ctrl))
            return value[t]
        ps_next = {frozenset((a, b)): (a, won, kind, rnd) for a, b, won, kind, rnd in cap['ps_next']}
        mu = sim._mu(snap_d)

        # Regular-season records going into the day; bo3 series scores.
        played = rows[(rows['date'] < day) & rows['home_score'].notna()]
        rec, series = {}, {}
        for x in played.itertuples(index=False):
            hs, as_ = x.home_score, x.away_score
            if x.stage == 'regular-season':
                for t, us, them in ((x.home_team, hs, as_), (x.away_team, as_, hs)):
                    w, d_, l = rec.get(t, (0, 0, 0))
                    rec[t] = (w + (us > them), d_ + (us == them), l + (us < them))
            else:
                win = x.home_team if hs > as_ else x.away_team if as_ > hs else x.shootout_winner
                k = frozenset((x.home_team, x.away_team))
                series.setdefault(k, {}).setdefault(win, 0)
                series[k][win] += 1

        games_out = []
        for x in rows[rows['date'] == day].itertuples(index=False):
            h, a = x.home_team, x.away_team
            blank = pd.Series({'rating': 0.0, 'rating_o': 0.0, 'rating_d': 0.0, 'rank': np.nan})
            rh = rt.loc[h] if h in rt.index else blank
            ra = rt.loc[a] if a in rt.index else blank
            stakes, ser, p_draw = None, None, None
            if x.stage == 'regular-season':
                if rest is None or (x.date, h, a) not in rs_keys:
                    continue
                j = rest.index[(rest['home_team'] == h) & (rest['away_team'] == a) & (rest['date'] == day)]
                if not len(j):
                    continue
                hg, ag = cap['hg'][:, j[0]], cap['ag'][:, j[0]]
                p_h, p_draw, p_a = float((hg > ag).mean()), float((hg == ag).mean()), float((ag > hg).mean())
                margin, total = float((hg - ag).mean()), float((hg + ag).mean())
                stakes = {}
                for t, opp, pw, pl in ((h, a, p_h, p_a), (a, h, p_a, p_h)):
                    (po_m, po_b), (ti_m, ti_b) = per_point(t, opp)
                    po = _wdl(po_m, po_b, pw, p_draw)
                    ti = _wdl(ti_m, ti_b, pw, p_draw)
                    stakes['home' if t == h else 'away'] = {
                        'po_win': po[0], 'po_draw': po[1], 'po_loss': po[2],
                        'title_win': ti[0], 'title_draw': ti[1], 'title_loss': ti[2]}
            else:
                hit = ps_next.get(frozenset((h, a)))
                if hit is None:
                    continue                          # an unplayed match after the matchup ended
                ta, won_a, kind, rnd = hit
                hw = won_a if ta == h else ~won_a
                p_h = float(hw.mean()); p_a = 1 - p_h
                # Line and O/U from the same goal model: 90 minutes, home edge
                # except at a neutral MLS Cup site (none since 2012).
                lam_h = mu + title_odds.RATING_SCALE * (rh['rating_o'] - ra['rating_d']) + h_home
                lam_a = mu + title_odds.RATING_SCALE * (ra['rating_o'] - rh['rating_d']) - h_away
                lam_h = float(np.clip(lam_h, title_odds.LAMBDA_MIN, title_odds.LAMBDA_MAX))
                lam_a = float(np.clip(lam_a, title_odds.LAMBDA_MIN, title_odds.LAMBDA_MAX))
                margin, total = lam_h - lam_a, lam_h + lam_a
                stakes = {}
                for t, won in ((h, hw), (a, ~hw)):
                    ti_w, ti_l = _raw_split(champ == idx[t], won)
                    stakes['home' if t == h else 'away'] = {
                        'po_win': None, 'po_draw': None, 'po_loss': None,
                        'title_win': ti_w, 'title_draw': None, 'title_loss': ti_l}
                sw = series.get(frozenset((h, a)), {})
                ser = {'kind': kind, 'round': names[rnd - 1],
                       'home_wins': sw.get(h, 0), 'away_wins': sw.get(a, 0)}
            game = {
                'home': h, 'away': a,
                'home_record': _record(*rec.get(h, (0, 0, 0))), 'away_record': _record(*rec.get(a, (0, 0, 0))),
                'home_rank': None if pd.isna(rh['rank']) else int(rh['rank']),
                'away_rank': None if pd.isna(ra['rank']) else int(ra['rank']),
                'p_home': round(p_h, 4), 'p_draw': None if p_draw is None else round(p_draw, 4),
                'p_away': round(p_a, 4),
                'line': _line(margin, p_h >= p_a),             # home by this many (negative = away)
                'total': round(total * 2) / 2,                 # COBI O/U: projected combined goals
                'kickoff': kickoffs.get(str(x.event_id)),
                'stakes': None if stakes is None else
                          {k: {kk: (None if vv is None else round(vv, 4)) for kk, vv in v.items()} for k, v in stakes.items()},
                'result': None,
            }
            if ser is not None:
                game['series'] = ser
            if not pd.isna(x.home_score):
                game['result'] = {'home': int(x.home_score), 'away': int(x.away_score)}
                if x.home_score == x.away_score and isinstance(x.shootout_winner, str):
                    game['result']['pens'] = 'home' if x.shootout_winner == h else 'away'
            game['_q'] = (min(rh['rating'], ra['rating']), max(rh['rating'], ra['rating']))
            games_out.append(game)
        previewed = preview_day is not None and day >= preview_day and bool(games_out)
        if games_out:
            days_out.append({'date': day.strftime('%Y-%m-%d'),
                             'playoffs': any('series' in x for x in games_out), 'games': games_out})
        log(f'  {season} {day.date()}: {len(games_out)} matches')
    return days_out


# ── Juice: Quality x Stakes ──────────────────────────────────────────────────
# As DUNCAN, after DILLON (weekly_matchups.py): Quality = 35% the worse
# club's rating, 35% the better club's, 30% closeness; Stakes = both clubs'
# playoff-odds swing + K x their Cup-odds swing (win vs loss), K ramping
# 4 -> 8 over the regular season (playoff matches: Cup swing only, K = 8).
# Each is ranked 0-100 and Juice = sqrt(Quality x Stakes). Closeness counts a
# draw as close: 1 - |P(home) - P(away)|.
#
# The ranking pool is COMPLETED seasons only, so history only re-ranks once a
# year, when a season finishes.
TITLE_K_START, TITLE_K_END = 4.0, 8.0


def _pct(ref_sorted, x):
    lo = np.searchsorted(ref_sorted, x, side='left')
    hi = np.searchsorted(ref_sorted, x, side='right')
    return ((lo + hi + 1) / 2) / len(ref_sorted)


def add_juice(seasons, pool_seasons):
    rows = []
    for season, days in seasons.items():
        rs_days = [d['date'] for d in days if not d['playoffs']]
        first, last = (rs_days[0], rs_days[-1]) if rs_days else (None, None)
        span = (pd.Timestamp(last) - pd.Timestamp(first)).days if rs_days else 1
        for d in days:
            if d['playoffs']:
                k = TITLE_K_END
            else:
                frac = (pd.Timestamp(d['date']) - pd.Timestamp(first)).days / max(span, 1)
                k = TITLE_K_START + (TITLE_K_END - TITLE_K_START) * min(frac, 1.0)
            for g in d['games']:
                qmin, qmax = g.pop('_q')
                close = 1 - abs(g['p_home'] - g['p_away'])
                stake = None
                if g['stakes'] is not None:
                    st = list(g['stakes'].values())
                    po = sum((s['po_win'] or 0) - (s['po_loss'] or 0) for s in st)
                    ti = sum((s['title_win'] or 0) - (s['title_loss'] or 0) for s in st)
                    stake = po + k * ti
                rows.append((g, season in pool_seasons, qmin, qmax, close, stake))
    if not rows:
        return
    df = pd.DataFrame([r[1:] for r in rows], columns=['pool', 'qmin', 'qmax', 'close', 'stake'])
    ref = df[df['pool']] if df['pool'].any() else df
    parts = {c: np.sort(ref[c].to_numpy()) for c in ('qmin', 'qmax', 'close')}
    q_raw = sum(w * _pct(parts[c], df[c].to_numpy()) for c, w in (('qmin', .35), ('qmax', .35), ('close', .3)))
    pool_mask = df['pool'].to_numpy() if df['pool'].any() else np.ones(len(df), bool)
    q = _pct(np.sort(q_raw[pool_mask]), q_raw) * 100
    st_ref = np.sort(ref['stake'].dropna().to_numpy())
    for (g, *_), qq, ss in zip(rows, q, df['stake']):
        g['quality'] = int(round(min(qq, 100)))
        if pd.isna(ss):
            g['stakes_score'] = g['juice'] = None
        else:
            s_pct = min(_pct(st_ref, ss) * 100, 100)
            g['stakes_score'] = int(round(s_pct))
            g['juice'] = int(round(np.sqrt(min(qq, 100) * s_pct)))


# ── Per-season cache ─────────────────────────────────────────────────────────
CACHE_DIR = 'matchups_cache'
_HERE = os.path.dirname(os.path.abspath(__file__))
_JOB = {}


def _fingerprint(season, J):
    h = hashlib.sha256()
    for f in ('daily_matchups.py', 'title_odds.py', 'mls_playoff_seeds.json'):
        p = os.path.join(_HERE, f)
        if os.path.exists(p):
            h.update(open(p, 'rb').read())
    g = J['games'][J['games']['date'].dt.year.isin([season - 1, season])]
    h.update(g.sort_values(list(g.columns), kind='stable', na_position='last').to_csv(index=False).encode())
    if season == J['current'] and J['fixtures'] is not None:
        h.update(J['fixtures'].sort_values(['date', 'home_team']).to_csv(index=False).encode())
    r = J['ratings'][J['ratings']['season'].isin([season - 1, season])].sort_values(['date', 'team']).copy()
    for c in ('rating', 'rating_o', 'rating_d'):
        r[c] = r[c].round(3)
    h.update(r.to_csv(index=False).encode())
    teams = sorted(set(g['home_team']) | set(g['away_team']))
    h.update(repr([(t, J['conference_for'](t, str(season))) for t in teams]).encode())
    h.update(repr(sorted((k, v) for k, v in J['kickoffs'].items() if k in set(g['event_id'].astype(str)))).encode())
    return h.hexdigest()


def _one(season):
    J = _JOB
    r = J['ratings']
    days = build_season(season, J['games'], J['fixtures'] if season == J['current'] else None,
                        r[r['season'].isin([season - 1, season])], J['conference_for'], J['mu_games'],
                        J['kickoffs'], log=lambda *_: None)
    return season, days


def build_cached(seasons, games, fixtures, ratings, conference_for, current, kickoffs, workers=None, log=print):
    """{season: days} for every season, reusing cached ones. games: MLS
    matches with stage; ratings: season, date, team, rating, rank, rating_o,
    rating_d."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    games = games.copy()
    games['stage'] = games['stage'].fillna('')
    mu_games = games[games['stage'] == 'regular-season']
    _JOB.update(games=games, fixtures=fixtures, ratings=ratings, conference_for=conference_for,
                current=current, mu_games=mu_games, kickoffs=kickoffs)
    out, todo, sigs = {}, [], {}
    for s in seasons:
        sigs[s] = _fingerprint(s, _JOB)
        path = os.path.join(CACHE_DIR, f'{s}.pkl')
        if os.path.exists(path):
            try:
                old_sig, days = pickle.load(open(path, 'rb'))
                if old_sig == sigs[s]:
                    out[s] = days
                    continue
            except Exception:
                pass
        todo.append(s)
    log(f'  daily matchups: {len(seasons) - len(todo)} seasons from cache, computing {len(todo)}')
    if todo:
        with _mp.get_context('fork').Pool(workers or os.cpu_count()) as pool:
            for s, days in pool.imap_unordered(_one, sorted(todo, reverse=True)):
                out[s] = days
                pickle.dump((sigs[s], days), open(os.path.join(CACHE_DIR, f'{s}.pkl'), 'wb'))
    return {s: out[s] for s in seasons}
