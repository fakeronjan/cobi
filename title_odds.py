"""COBI MLS Cup odds: Monte Carlo of the rest of the season.

For every rating snapshot from FIRST_SEASON on, simulate the remaining
regular-season schedule, seed each conference from the simulated table, then
play out that season's playoff bracket. Anything already played (regular
season or playoffs) as of the snapshot date is fixed, not simulated.

Match model: Poisson goals from the snapshot's O/D ratings, mirroring the
half-equations cobi._solve_wls_od fits:
    home goals ~ Poisson(mu + k*(O_home - D_away) + hfa * off_share)
    away goals ~ Poisson(mu + k*(O_away - D_home) - hfa * (1 - off_share))
with mu = league mean goals per team per match over the prior year and
k = RATING_SCALE shrinking the raw ratings, which are overconfident as match
predictors.

Stages come from ESPN's season slug (all_club_games.csv `stage`, backfilled
for 2019+), so the regular-season/playoff split and each playoff game's round
are ground truth rather than date heuristics.
"""
import hashlib
import json as _json
import multiprocessing as _mp
import os as _os
import pickle

import numpy as np
import pandas as pd

FIRST_SEASON = 2019
# Simulation count (fleet standard since 2026-10-02): 10k for every date,
# playoffs included.
N_SIMS = 10_000
HFA = 0.5          # cobi.home_field_adv
OFF_SHARE = 0.5    # cobi.od_off_share
ET_FACTOR = 1 / 3  # extra time = 30 of 90 minutes
LAMBDA_MIN, LAMBDA_MAX = 0.15, 5.0
# Shrink on O/D before they become goal means. Fit by log loss on every
# 2019-2025 regular-season match using the prior snapshot's ratings: 0.5 is
# best overall (1.0427 vs 1.0553 unshrunk), and fitting on 2019-22 alone also
# picks 0.5, which then beats unshrunk on held-out 2023-25 (1.0534 vs 1.0625).
RATING_SCALE = 0.5
# Ratings aren't fixed for the rest of the season: each simulation gives
# every team a random rating offset for the remaining matches, SD =
# DRIFT_SD0 * (share of regular season left)**DRIFT_K, fit to how far MLS
# ratings actually moved from each date to the end of the regular season
# (split evenly between attack and defense, so net rating moves by the
# full offset). Zero once the regular season is over. (Same fix as DILLON.)
# ON since 2026-10-03 (was off from 2026-09-25). Rechecked over 2019-25 with
# more than champion odds: without drift, final points fell outside the
# 10th-90th projection band 30% of the time in the first quarter (should
# be <= ~20%); with it, 23%, and playoff-qualification log loss improves
# early (0.586 -> 0.571), flat later. Champion -log p is slightly worse
# early (2.49 -> 2.55), but that rests on 7 champions. 0.518 is the measured
# rating movement, not tuned to these checks (0.75 would hit the band
# target, but that would be fitting to the test).
DRIFT_SD0, DRIFT_K = 0.518, 0.51

# The real playoff seeds: {season: {'East'/'West': [seed 1, seed 2, ...]}}.
# Once the regular season is over these ARE the seeds: MLS's last tiebreaker
# (disciplinary points) isn't in our data, so exact ties otherwise fall to a
# coin flip per sim. From Wikipedia's playoff brackets; cobi.py adds each
# new season.
_SEEDS = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'mls_playoff_seeds.json')
REAL_SEEDS = ({int(k): v for k, v in _json.load(open(_SEEDS)).items()}
              if _os.path.exists(_SEEDS) else {})

# ── Playoff formats ─────────────────────────────────────────────────────────
# Per-conference bracket as (match_id, round, kind, slot_a, slot_b). A slot is
# an int seed or 'W:<match_id>' (that match's winner). Better seed hosts
# (bo3: better seed hosts games 1 and 3). Kinds:
#   single_et  one match, extra time then penalties if level
#   single_pk  one match, straight to penalties if level
#   bo3        best of three, every game straight to penalties if level
FORMAT_14 = [  # 2019, 2021, 2022: 7 per conference, top seed bye
    ('r1a', 'r1', 'single_et', 2, 7),
    ('r1b', 'r1', 'single_et', 3, 6),
    ('r1c', 'r1', 'single_et', 4, 5),
    ('sfa', 'sf', 'single_et', 1, 'W:r1c'),
    ('sfb', 'sf', 'single_et', 'W:r1a', 'W:r1b'),
    ('cf',  'cf', 'single_et', 'W:sfa', 'W:sfb'),
]
FORMAT_18 = [  # 2023+: 9 per conference, 8v9 wild card, best-of-3 round one
    ('wc',  'wc', 'single_pk', 8, 9),
    ('r1a', 'r1', 'bo3', 1, 'W:wc'),
    ('r1b', 'r1', 'bo3', 2, 7),
    ('r1c', 'r1', 'bo3', 3, 6),
    ('r1d', 'r1', 'bo3', 4, 5),
    ('sfa', 'sf', 'single_et', 'W:r1a', 'W:r1d'),
    ('sfb', 'sf', 'single_et', 'W:r1b', 'W:r1c'),
    ('cf',  'cf', 'single_et', 'W:sfa', 'W:sfb'),
]
FORMAT_2020_EAST = [  # 10 teams, two play-in games
    ('pia', 'playin', 'single_et', 8, 9),
    ('pib', 'playin', 'single_et', 7, 10),
    ('r1a', 'r1', 'single_et', 1, 'W:pia'),
    ('r1b', 'r1', 'single_et', 2, 'W:pib'),
    ('r1c', 'r1', 'single_et', 3, 6),
    ('r1d', 'r1', 'single_et', 4, 5),
    ('sfa', 'sf', 'single_et', 'W:r1a', 'W:r1d'),
    ('sfb', 'sf', 'single_et', 'W:r1b', 'W:r1c'),
    ('cf',  'cf', 'single_et', 'W:sfa', 'W:sfb'),
]
FORMAT_2020_WEST = [  # 8 teams
    ('r1a', 'r1', 'single_et', 1, 8),
    ('r1b', 'r1', 'single_et', 2, 7),
    ('r1c', 'r1', 'single_et', 3, 6),
    ('r1d', 'r1', 'single_et', 4, 5),
    ('sfa', 'sf', 'single_et', 'W:r1a', 'W:r1d'),
    ('sfb', 'sf', 'single_et', 'W:r1b', 'W:r1c'),
    ('cf',  'cf', 'single_et', 'W:sfa', 'W:sfb'),
]


def bracket_for(season, conference):
    if season == 2020:
        return FORMAT_2020_EAST if conference == 'East' else FORMAT_2020_WEST
    if season <= 2022:
        return FORMAT_14
    return FORMAT_18


def round_order(season):
    """Round tags first to last (the MLS Cup final is 'final')."""
    if season == 2020:
        return ['playin', 'r1', 'sf', 'cf', 'final']
    if season <= 2022:
        return ['r1', 'sf', 'cf', 'final']
    return ['wc', 'r1', 'sf', 'cf', 'final']


def round_names(season):
    """(full, short) names for every round, first to last."""
    full = {'playin': 'Play-In Round', 'wc': 'Wild Card', 'r1': 'Round One',
            'sf': 'Conference Semifinals', 'cf': 'Conference Finals', 'final': 'MLS Cup'}
    short = {'playin': 'Play-In', 'wc': 'Wild Card', 'r1': 'R1', 'sf': 'R2',
             'cf': 'Conf Final', 'final': 'Final'}
    order = round_order(season)
    return [full[t] for t in order], [short[t] for t in order]


def entry_rounds(season):
    """Seed label ('E1', 'W8', ...) -> round number that seed enters."""
    order = round_order(season)
    out = {}
    for c in ('East', 'West'):
        for _, tag, _, *slots in bracket_for(season, c):
            for sl in slots:
                if isinstance(sl, int):
                    out.setdefault(f"{c[0]}{sl}", order.index(tag) + 1)
    return out


def uses_ppg(season):
    """2020 seeded on points per game (uneven schedules)."""
    return season == 2020


def stage_round(stage):
    """Map an ESPN season slug to our round tag (None = not a playoff game)."""
    s = stage.lower() if isinstance(stage, str) else ''
    if s == 'mls-cup':
        return 'final'
    if 'playoffs' not in s:
        return None
    if 'wild-card' in s:
        return 'wc'
    if 'play-in' in s:
        return 'playin'
    if 'round-one' in s or 'first-round' in s:
        return 'r1'
    if 'semifinal' in s:
        return 'sf'
    if s.endswith('---final') or s.endswith('---finals'):  # 2019 spells it 'finals'
        return 'cf'
    return None


def _winner(row):
    """Winner of a played match (score, then shootout), or None if level."""
    if row['home_score'] > row['away_score']:
        return row['home_team']
    if row['away_score'] > row['home_score']:
        return row['away_team']
    sw = row.get('shootout_winner')
    if isinstance(sw, str) and sw in (row['home_team'], row['away_team']):
        return sw
    return None


class SeasonSim:
    """All inputs for one season; call odds_at(date) per snapshot."""

    def __init__(self, season, games, fixtures, conference_for, ratings, mu_games):
        self.season = season
        rs = games[games['stage'] == 'regular-season']
        fx = fixtures[fixtures['stage'] == 'regular-season'] if fixtures is not None else fixtures
        teams = sorted(set(rs['home_team']) | set(rs['away_team']) |
                       (set(fx['home_team']) | set(fx['away_team']) if fx is not None else set()))
        self.conf = {t: conference_for(t, str(season)) for t in teams}
        self.teams = [t for t in teams if self.conf[t] in ('East', 'West')]
        self.idx = {t: i for i, t in enumerate(self.teams)}
        rs = rs[rs['home_team'].isin(self.idx) & rs['away_team'].isin(self.idx)]

        played = rs[['date', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
        future = (fx[['date', 'home_team', 'away_team']].copy() if fx is not None
                  else played.iloc[0:0][['date', 'home_team', 'away_team']])
        future = future[future['home_team'].isin(self.idx) & future['away_team'].isin(self.idx)]
        self.rs = pd.concat([played, future], ignore_index=True).sort_values('date', kind='stable')
        self.rs['h'] = self.rs['home_team'].map(self.idx)
        self.rs['a'] = self.rs['away_team'].map(self.idx)
        self.rs_complete = future.empty
        self.decision_day = played['date'].max() if self.rs_complete and not played.empty else None

        ps = games[games['stage'].map(stage_round).notna()].copy()
        ps['round'] = ps['stage'].map(stage_round)
        ps['winner'] = ps.apply(_winner, axis=1)
        self.ps = ps.sort_values('date', kind='stable')

        self.ratings = ratings  # date -> {team: (O, D)}
        self.mu_games = mu_games  # all MLS RS games (for the league goal mean)

    # ── helpers ──
    def _mu(self, d):
        g = self.mu_games[(self.mu_games['date'] <= d) &
                          (self.mu_games['date'] > d - pd.Timedelta(days=365))]
        if len(g) < 100:
            return 1.5
        return float((g['home_score'].sum() + g['away_score'].sum()) / (2 * len(g)))

    def odds_at(self, d, n_sims=N_SIMS):
        T = len(self.teams)
        rng = np.random.default_rng(int(d.strftime('%Y%m%d')))
        rt = self.ratings.get(d, {})
        O = np.array([rt.get(t, (0.0, 0.0))[0] for t in self.teams])
        D = np.array([rt.get(t, (0.0, 0.0))[1] for t in self.teams])
        mu = self._mu(d)
        h_home, h_away = HFA * OFF_SHARE, HFA * (1 - OFF_SHARE)

        E = None   # per-sim rating offsets, set once the remaining schedule is known

        def lam(att, dfn, edge, rows=None):
            """rows: None = no per-sim offsets; 'rs' = (sims x matches) for
            the regular-season matrix; 'sim' = one match per sim."""
            x = mu + RATING_SCALE * (O[att] - D[dfn]) + edge
            if E is not None and rows == 'rs':
                x = x + RATING_SCALE * (E[:, att] - E[:, dfn]) / 2
            elif E is not None and rows == 'sim':
                x = x + RATING_SCALE * (E[np.arange(len(att)), att] - E[np.arange(len(att)), dfn]) / 2
            return np.clip(x, LAMBDA_MIN, LAMBDA_MAX)

        # ── regular season ──
        done = self.rs[(self.rs['date'] <= d) & self.rs['home_score'].notna()]
        rest = self.rs[(self.rs['date'] > d) | self.rs['home_score'].isna()]
        frac_left = len(rest) / max(len(self.rs), 1)
        sd = DRIFT_SD0 * frac_left ** DRIFT_K if frac_left > 0 else 0.0
        if sd > 0:
            E = rng.normal(0.0, sd, (n_sims, T))
        pts = np.zeros(T); w = np.zeros(T); gf = np.zeros(T); ga = np.zeros(T)
        agf = np.zeros(T); aga = np.zeros(T); gp = np.zeros(T)
        for h, a, hs, as_ in done[['h', 'a', 'home_score', 'away_score']].itertuples(index=False):
            hs, as_ = int(hs), int(as_)
            gp[h] += 1; gp[a] += 1
            gf[h] += hs; ga[h] += as_; gf[a] += as_; ga[a] += hs
            agf[a] += as_; aga[a] += hs
            if hs > as_: pts[h] += 3; w[h] += 1
            elif as_ > hs: pts[a] += 3; w[a] += 1
            else: pts[h] += 1; pts[a] += 1
        P = np.tile(pts, (n_sims, 1)); W = np.tile(w, (n_sims, 1))
        GF = np.tile(gf, (n_sims, 1)); GA = np.tile(ga, (n_sims, 1))
        AGF = np.tile(agf, (n_sims, 1)); AGA = np.tile(aga, (n_sims, 1))
        GP = np.tile(gp, (n_sims, 1))
        if len(rest):
            h = rest['h'].to_numpy(); a = rest['a'].to_numpy()
            G = len(rest)
            hg = rng.poisson(lam(h, a, h_home, 'rs'), size=(n_sims, G))
            ag = rng.poisson(lam(a, h, -h_away, 'rs'), size=(n_sims, G))
            Hm = np.zeros((G, T)); Hm[np.arange(G), h] = 1
            Am = np.zeros((G, T)); Am[np.arange(G), a] = 1
            hw = (hg > ag).astype(float); aw = (ag > hg).astype(float); dr = (hg == ag).astype(float)
            P += (3 * hw + dr) @ Hm + (3 * aw + dr) @ Am
            W += hw @ Hm + aw @ Am
            GF += hg @ Hm + ag @ Am
            GA += ag @ Hm + hg @ Am
            AGF += ag @ Am
            AGA += hg @ Am
            GP += np.ones((n_sims, G)) @ (Hm + Am)
        primary = P / np.maximum(GP, 1) if uses_ppg(self.season) else P
        # Projected points (Standings' Proj Points bar): 10th/50th/90th
        # percentile of simulated final points while games remain; the bar
        # runs to the most points possible (3 per match). No random draws.
        proj = ((np.quantile(P, [0.1, 0.5, 0.9], axis=0, method='inverted_cdf'), 3 * GP[0])
                if len(rest) else None)
        seed_pts = primary  # also decides MLS Cup hosting

        # ── seeding ──
        seeds = {}
        sim_ix = np.arange(n_sims)
        for c in ('East', 'West'):
            cidx = np.array([self.idx[t] for t in self.teams if self.conf[t] == c])
            k = len(cidx)
            # np.lexsort: LAST key is primary. MLS order: points (PPG in
            # 2020), wins, GD, GF, away GD, away GF, then a coin toss
            # (disciplinary points aren't in our data).
            keys = [rng.random((n_sims, k)),
                    AGF[:, cidx], (AGF - AGA)[:, cidx],
                    GF[:, cidx], (GF - GA)[:, cidx], W[:, cidx],
                    primary[:, cidx]]
            flat = [(-kk).ravel() for kk in keys] + [np.repeat(sim_ix, k)]
            order = np.lexsort(flat).reshape(n_sims, k)
            seeds[c] = cidx[order % k]  # (n_sims, k) team idx, best first
        if rest.empty and self.season in REAL_SEEDS:
            for c, real in REAL_SEEDS[self.season].items():
                top = [self.idx[t] for t in real]
                full = top + [t for t in seeds[c][0] if t not in top]
                seeds[c] = np.broadcast_to(np.array(full), (n_sims, len(full)))

        # ── playoffs ──
        order_tags = round_order(self.season)
        n_rounds = len(order_tags)
        self.used_actual = 0      # validation: real playoff games the bracket consumed
        self.rs_complete = rest.empty
        self.seeds = {}
        if self.rs_complete:
            for c, arr in seeds.items():
                for k, t in enumerate(arr[0]):
                    self.seeds[self.teams[t]] = f"{c[0]}{k + 1}"
        self.matchups = []        # (round, kind, team_a, team_b, games, decided winner)
        reach = np.zeros((n_rounds + 2, T))   # [0] playoffs, [k] reach round k, [-1] champion
        entered = np.zeros((n_sims, T), dtype=bool)

        def enter(t, rnd):
            new = ~entered[sim_ix, t]
            entered[sim_ix, t] = True
            np.add.at(reach[0], t[new], 1)
            for k in range(2, rnd):   # a bye counts as getting through
                np.add.at(reach[k], t[new], 1)
            np.add.at(reach[rnd], t, 1)

        # Every real playoff match's host, in order per pair (played or not as
        # of d: set by the seeding, not the result). A real match uses them.
        real_hosts = {}
        for r in self.ps.itertuples(index=False):
            real_hosts.setdefault(frozenset((r.home_team, r.away_team)), []).append(r.home_team)
        self.host_miss = 0

        ps_done = self.ps[self.ps['date'] <= d]
        by_pair = {}
        for r in ps_done.itertuples(index=False):
            by_pair.setdefault(frozenset((r.home_team, r.away_team)), []).append(r)

        def actual(a_arr, b_arr):
            """Played games between this match's teams, if they're fixed."""
            if not (np.all(a_arr == a_arr[0]) and np.all(b_arr == b_arr[0])):
                return []
            key = frozenset((self.teams[a_arr[0]], self.teams[b_arr[0]]))
            return by_pair.get(key, [])

        def one_game(a, b, a_hosts, pens_only):
            ea = np.where(a_hosts, h_home, -h_away)
            eb = np.where(a_hosts, -h_away, h_home)
            ga_ = rng.poisson(lam(a, b, ea, 'sim')); gb_ = rng.poisson(lam(b, a, eb, 'sim'))
            if not pens_only:
                lv = ga_ == gb_
                ga_ = ga_ + np.where(lv, rng.poisson(lam(a, b, ea, 'sim') * ET_FACTOR), 0)
                gb_ = gb_ + np.where(lv, rng.poisson(lam(b, a, eb, 'sim') * ET_FACTOR), 0)
            coin = rng.random(len(a)) < 0.5
            return np.where(ga_ > gb_, True, np.where(gb_ > ga_, False, coin))

        def play(kind, a, sa, b, sb, host_a, rnd):
            """a/b team idx arrays, sa/sb their seeds; returns winner arrays."""
            enter(a, rnd); enter(b, rnd)
            played = actual(a, b)
            tA = self.teams[a[0]]
            fixed = np.all(a == a[0]) and np.all(b == b[0])
            hosts = real_hosts.get(frozenset((tA, self.teams[b[0]])), []) if fixed else []
            if hosts:
                real_a = hosts[0] == tA          # single match, and bo3 game 1, at home field
                self.host_miss += bool(np.asarray(host_a)[0]) != real_a
                host_a = np.full(len(a), real_a)
            if played and self.rs_complete:
                tB = self.teams[b[0]]
                games_ = [(r.winner, int(r.home_score if r.home_team == tA else r.away_score),
                           int(r.away_score if r.home_team == tA else r.home_score),
                           r.home_score == r.away_score) for r in played[:3 if kind == 'bo3' else 1]]
                wins = [x[0] for x in games_]
                if kind == 'bo3':
                    decided = tA if wins.count(tA) >= 2 else (tB if wins.count(tB) >= 2 else None)
                else:
                    decided = wins[0]
                self.matchups.append((rnd, kind, tA, tB, games_, decided))
            elif self.rs_complete and np.all(a == a[0]) and np.all(b == b[0]):
                self.matchups.append((rnd, kind, tA, self.teams[b[0]], [], None))
            if kind == 'bo3':
                wa = np.zeros(len(a), dtype=int); wb = np.zeros(len(a), dtype=int)
                for g in range(3):
                    if g < len(played):
                        won = np.full(len(a), played[g].winner == tA)
                        self.used_actual += 1
                    else:
                        if g < len(hosts):
                            at = np.full(len(a), hosts[g] == tA)
                        else:
                            at = host_a if g != 1 else ~host_a
                        won = one_game(a, b, at, pens_only=True)
                    live = (wa < 2) & (wb < 2)
                    wa += (won & live); wb += (~won & live)
                a_wins = wa >= 2
            else:
                if played:
                    a_wins = np.full(len(a), played[-1].winner == tA)
                    self.used_actual += 1
                else:
                    a_wins = one_game(a, b, host_a, pens_only=(kind == 'single_pk'))
            return (np.where(a_wins, a, b), np.where(a_wins, sa, sb))

        conf_champ = {}
        for c in ('East', 'West'):
            S = seeds[c]
            res = {}
            def slot(x):
                if isinstance(x, int):
                    return S[:, x - 1], np.full(n_sims, x)
                return res[x[2:]]
            for mid, tag, kind, xa, xb in bracket_for(self.season, c):
                a, sa = slot(xa); b, sb = slot(xb)
                res[mid] = play(kind, a, sa, b, sb, host_a=sa < sb,
                                rnd=order_tags.index(tag) + 1)
            conf_champ[c] = res['cf']
        e, _ = conf_champ['East']; wt, _ = conf_champ['West']
        host_e = seed_pts[sim_ix, e] >= seed_pts[sim_ix, wt]
        champ, _ = play('single_et', e, np.zeros(n_sims), wt, np.zeros(n_sims), host_e,
                        rnd=n_rounds)
        np.add.at(reach[-1], champ, 1)
        reach /= n_sims
        cols = ['playoffs'] + [f'r{k}' for k in range(2, n_rounds + 1)] + ['champ']
        rows = np.vstack([reach[0]] + [reach[k] for k in range(2, n_rounds + 1)] + [reach[-1]])
        res = pd.DataFrame(rows.T, index=self.teams, columns=cols)
        if proj is not None:
            (res['proj_lo'], res['proj_mid'], res['proj_hi']), res['proj_max'] = proj
        return res


def compute(games, fixtures, ratings_df, conference_for, current_season, seasons=None, log=print):
    """Return (odds, brackets) for FIRST_SEASON+: odds = long DataFrame
    (season, date, team, playoffs, r2.., champ); brackets = {season: {date:
    (seeds, matchups, n_sims)}} for snapshots after the regular season.

    games:      MLS matches with date (Timestamp), stage, scores, shootout_winner
    fixtures:   scheduled current-season matches (date, home_team, away_team, stage)
    ratings_df: cobi_ratings_final rows (date, season, team, rating_o, rating_d)
    """
    games = games.copy()
    games['stage'] = games['stage'].fillna('')
    mu_games = games[games['stage'] == 'regular-season']
    out = []
    brackets = {}
    for season in (seasons or range(FIRST_SEASON, current_season + 1)):
        g = games[games['date'].dt.year == season]
        if g.empty:
            continue
        missing = (g['stage'] == '').sum()
        if missing:
            raise RuntimeError(f"{missing} {season} MLS games have no ESPN stage - run backfill_stage.py")
        fx = fixtures if season == current_season else None
        rs_sub = ratings_df[ratings_df['season'] == season]
        ratings = {d: dict(zip(sub['team'], zip(sub['rating_o'], sub['rating_d'])))
                   for d, sub in rs_sub.groupby('date')}
        sim = SeasonSim(season, g, fx, conference_for, ratings, mu_games)
        dates = sorted(ratings)
        for d in dates:
            n = N_SIMS
            o = sim.odds_at(d, n_sims=n)
            if sim.rs_complete:
                brackets.setdefault(season, {})[d] = (dict(sim.seeds), list(sim.matchups), n)
            o.index.name = 'team'
            o = o.reset_index()
            o['season'] = season
            o['date'] = d
            out.append(o)
        log(f"  {season}: {len(dates)} snapshots")
    return pd.concat(out, ignore_index=True), brackets


# ---------------------------------------------------------------------------
# Per-season cache. Every snapshot seeds its own RNG from its date, so a
# season's odds depend only on the engine and that season's inputs; finished
# seasons are reused until one of those changes. Fingerprint = this file +
# the season's games and the prior year's regular season (the league goal
# mean looks back 365 days), full sort key so tie order can't change the
# hash + its ratings to 3dp + its fixtures + its teams' conferences.
_JOB = {}


def _fingerprint(season, games, fixtures, ratings_df, conference_for, current_season):
    h = hashlib.sha256(open(_os.path.abspath(__file__), 'rb').read())
    yr = games['date'].dt.year
    g = games[(yr == season) | ((yr == season - 1) & (games['stage'] == 'regular-season'))]
    h.update(g.sort_values(list(g.columns), kind='stable').to_csv(index=False).encode())
    r = ratings_df[ratings_df['season'] == season].sort_values(['date', 'team']).copy()
    r[['rating_o', 'rating_d']] = r[['rating_o', 'rating_d']].round(3)
    h.update(r.to_csv(index=False).encode())
    if season == current_season and fixtures is not None:
        h.update(fixtures.sort_values(list(fixtures.columns), kind='stable').to_csv(index=False).encode())
    teams = sorted(set(g['home_team']) | set(g['away_team']))
    h.update(repr([(t, conference_for(t, season)) for t in teams]).encode())
    h.update(repr(REAL_SEEDS.get(season)).encode())     # this season's real seeds only
    return h.hexdigest()


def _one(season):
    j = _JOB
    return season, compute(j['games'], j['fixtures'], j['ratings'], j['conf'], j['current'],
                           seasons=[season], log=lambda *_: None)


def compute_cached(games, fixtures, ratings_df, conference_for, current_season,
                   cache_dir='title_odds_cache', workers=None, log=print):
    """compute() over every season, reusing cached seasons whose fingerprint
    still matches and recomputing the rest in parallel."""
    _os.makedirs(cache_dir, exist_ok=True)
    yrs = set(games['date'].dt.year)
    seasons = [s for s in range(FIRST_SEASON, current_season + 1) if s in yrs]
    results, todo, sigs = {}, [], {}
    for s in seasons:
        sigs[s] = _fingerprint(s, games, fixtures, ratings_df, conference_for, current_season)
        path = _os.path.join(cache_dir, f'{s}.pkl')
        if _os.path.exists(path):
            try:
                sig, payload = pickle.load(open(path, 'rb'))
                if sig == sigs[s]:
                    results[s] = payload
                    continue
            except Exception:
                pass
        todo.append(s)
    log(f"  {len(results)} seasons from cache, computing {len(todo)}: {todo}")
    if todo:
        _JOB.update(games=games, fixtures=fixtures, ratings=ratings_df, conf=conference_for,
                    current=current_season)
        ctx = _mp.get_context('fork')   # workers inherit _JOB; no re-import of the caller
        with ctx.Pool(workers or _os.cpu_count()) as pool:
            for s, payload in pool.imap_unordered(_one, todo):
                results[s] = payload
                pickle.dump((sigs[s], payload), open(_os.path.join(cache_dir, f'{s}.pkl'), 'wb'))
                log(f"  {s} done")
    odds = pd.concat([results[s][0] for s in seasons], ignore_index=True)
    brackets = {}
    for s in seasons:
        brackets.update(results[s][1])
    return odds, brackets
