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
import numpy as np
import pandas as pd

FIRST_SEASON = 2019
# Simulation counts (fleet standard): regular-season dates 10k; once the
# regular season is over, 100k (10k leaves a visible ~1-point day-to-day
# wobble in playoff odds).
N_SIMS = 10000
N_SIMS_PLAYOFFS = 100_000
HFA = 0.5          # cobi.home_field_adv
OFF_SHARE = 0.5    # cobi.od_off_share
ET_FACTOR = 1 / 3  # extra time = 30 of 90 minutes
LAMBDA_MIN, LAMBDA_MAX = 0.15, 5.0
# Shrink on O/D before they become goal means. Fit by log loss on every
# 2019-2025 regular-season match using the prior snapshot's ratings: 0.5 is
# best overall (1.0427 vs 1.0553 unshrunk), and fitting on 2019-22 alone also
# picks 0.5, which then beats unshrunk on held-out 2023-25 (1.0534 vs 1.0625).
RATING_SCALE = 0.5

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

        def lam(att, dfn, edge):
            return np.clip(mu + RATING_SCALE * (O[att] - D[dfn]) + edge, LAMBDA_MIN, LAMBDA_MAX)

        # ── regular season ──
        done = self.rs[(self.rs['date'] <= d) & self.rs['home_score'].notna()]
        rest = self.rs[(self.rs['date'] > d) | self.rs['home_score'].isna()]
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
            hg = rng.poisson(lam(h, a, h_home), size=(n_sims, G))
            ag = rng.poisson(lam(a, h, -h_away), size=(n_sims, G))
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

        # ── playoffs ──
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
            ga_ = rng.poisson(lam(a, b, ea)); gb_ = rng.poisson(lam(b, a, eb))
            if not pens_only:
                lv = ga_ == gb_
                ga_ = ga_ + np.where(lv, rng.poisson(lam(a, b, ea) * ET_FACTOR), 0)
                gb_ = gb_ + np.where(lv, rng.poisson(lam(b, a, eb) * ET_FACTOR), 0)
            coin = rng.random(len(a)) < 0.5
            return np.where(ga_ > gb_, True, np.where(gb_ > ga_, False, coin))

        def play(kind, a, sa, b, sb, host_a):
            """a/b team idx arrays, sa/sb their seeds; returns winner arrays."""
            played = actual(a, b)
            tA = self.teams[a[0]]
            if kind == 'bo3':
                wa = np.zeros(len(a), dtype=int); wb = np.zeros(len(a), dtype=int)
                for g in range(3):
                    if g < len(played):
                        won = np.full(len(a), played[g].winner == tA)
                    else:
                        hosts = host_a if g != 1 else ~host_a
                        won = one_game(a, b, hosts, pens_only=True)
                    live = (wa < 2) & (wb < 2)
                    wa += (won & live); wb += (~won & live)
                a_wins = wa >= 2
            else:
                if played:
                    a_wins = np.full(len(a), played[-1].winner == tA)
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
            for mid, _rnd, kind, xa, xb in bracket_for(self.season, c):
                a, sa = slot(xa); b, sb = slot(xb)
                res[mid] = play(kind, a, sa, b, sb, host_a=sa < sb)
            conf_champ[c] = res['cf']
        e, _ = conf_champ['East']; wt, _ = conf_champ['West']
        host_e = seed_pts[sim_ix, e] >= seed_pts[sim_ix, wt]
        champ, _ = play('single_et', e, np.zeros(n_sims), wt, np.zeros(n_sims), host_e)
        probs = np.bincount(champ, minlength=T) / n_sims
        return dict(zip(self.teams, probs))


def compute(games, fixtures, ratings_df, conference_for, current_season, log=print):
    """Return DataFrame(season, date, team, title_odds) for FIRST_SEASON+.

    games:      MLS matches with date (Timestamp), stage, scores, shootout_winner
    fixtures:   scheduled current-season matches (date, home_team, away_team, stage)
    ratings_df: cobi_ratings_final rows (date, season, team, rating_o, rating_d)
    """
    games = games.copy()
    games['stage'] = games['stage'].fillna('')
    mu_games = games[games['stage'] == 'regular-season']
    out = []
    for season in range(FIRST_SEASON, current_season + 1):
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
            rs_left = ((sim.rs['date'] > d) | sim.rs['home_score'].isna()).any()
            n = N_SIMS if rs_left else N_SIMS_PLAYOFFS
            for team, p in sim.odds_at(d, n_sims=n).items():
                out.append((season, d, team, p))
        log(f"  {season}: {len(dates)} snapshots")
    return pd.DataFrame(out, columns=['season', 'date', 'team', 'title_odds'])
