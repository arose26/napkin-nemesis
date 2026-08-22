#!/usr/bin/env python3
"""napkin-100k-connect4 - a self-play net that fits in a 100,000-byte source file.

Second instance of the napkin-100k question, same 100,000-UTF-8-byte cap (the cap
is PLATFORM-WIDE on CodinGame, measured across seven arenas in the tic-tac-toe
repo), different environment: CodinGame's Connect 4.

One file, four jobs: an exact offline replica of the venue's referee, a GPU
self-play trainer for a policy+value net inside it, scripted opponents to measure
that net against OFFLINE, and a packer that emits the trained net as a single
self-contained C++ source.

Semantics replicated from the Java source (AshKcg/cg-multi-connect4), not folklore:
- Board is 7 rows x 9 columns; rows arrive top-first, so cell = row * 9 + col with
  row 0 at the TOP. A chip dropped in column c settles in the lowest empty cell.
- Win = 4 connected in any of the four directions. The referee only tests lines
  through the cell that just changed, which is equivalent to a global test because
  the game stops the instant a line completes.
- THE STEAL: the second player's first action (turnIndex == 1) may be -2 instead of
  a column. It places nothing; it repaints the first player's single chip as the
  second player's own. This is the pie rule, and it is the whole reason this arena
  is not just "connect 4": the opening is a bid, not a move.
- turnIndex counts actions including a steal. Draw when turnIndex reaches
  63 + (1 if a steal was used), which is exactly a full board.
- Both league levels ship byte-identical statements: one ruleset from Wood up.

Usage:
  napkin_c4.py selfcheck                     asserts on the rules and encoders
  napkin_c4.py bench                         reference-engine plies/s
  napkin_c4.py fuzz --other PATH             cross-fuzz vs a second engine
  napkin_c4.py match --a POL --b POL          scripted opponents, offline
  napkin_c4.py cg --policy POL                CG protocol adapter (parity harness)
  napkin_c4.py gpu-parity --check-encode      tensor engine vs the reference
  napkin_c4.py train-gpu --iters N            GPU self-play
  napkin_c4.py pack --net PT --out CPP        checkpoint -> single C++ source
  napkin_c4.py check-pack --cpp CPP --net PT  emitted forward pass vs int8 numpy
  napkin_c4.py check-bot --cpp CPP            legality + every forced win taken
  napkin_c4.py bench-net --vs POL             net vs an opponent, both seats
  napkin_c4.py snapshot                       append a ladder snapshot

Scripted opponents (random | greedy | ab) exist for OFFLINE measurement only.
Standing series rule: every submission to the arena is the net.
"""

import argparse
import importlib.util
import os
import random
import sys
import time

ROWS, COLS = 7, 9
CELLS = ROWS * COLS               # 63
ACT_STEAL = 9                     # our action id for the referee's -2
N_ACT = COLS + 1                  # 9 columns + steal


def _build_lines():
    """Every 4-in-a-row on the 7x9 board, as tuples of cell indices."""
    out = []
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = []
                for k in range(4):
                    rr, cc = r + dr * k, c + dc * k
                    if not (0 <= rr < ROWS and 0 <= cc < COLS):
                        break
                    cells.append(rr * COLS + cc)
                if len(cells) == 4:
                    out.append(tuple(cells))
    return tuple(out)


LINES = _build_lines()                                          # 126
LINE_MASKS = tuple(sum(1 << c for c in L) for L in LINES)
LINES_AT = tuple(tuple(i for i, L in enumerate(LINES) if cell in L)
                 for cell in range(CELLS))


def _pc(x: int) -> int:
    return bin(x).count("1")


class Engine:
    """Exact replica of the CG Connect 4 referee's transition function."""

    def __init__(self):
        self.bb = [0, 0]              # per-player 63-bit occupancy
        self.fill = [0] * COLS        # chips per column
        self.turn = 0                 # the referee's turnIndex
        self.steal_used = False
        self.last = -1                # the cell that changed most recently
        self._over = False
        self._winner = -1             # 0 / 1, or -1 for a draw or not-yet
        self._valid = None

    # -- read interface ------------------------------------------------------

    @property
    def current_player(self) -> int:
        return self.turn % 2

    @property
    def game_over(self) -> bool:
        return self._over

    @property
    def winner(self) -> int:
        return self._winner

    def valid_actions(self) -> set:
        if self._valid is not None:
            return self._valid
        if self._over:
            self._valid = set()
        else:
            va = {c for c in range(COLS) if self.fill[c] < ROWS}
            if self.turn == 1:
                va.add(ACT_STEAL)
            self._valid = va
        return self._valid

    def _four_through(self, p: int, cell: int) -> bool:
        mine = self.bb[p]
        for i in LINES_AT[cell]:
            m = LINE_MASKS[i]
            if (mine & m) == m:
                return True
        return False

    # -- transition ----------------------------------------------------------

    def play(self, a: int) -> None:
        if a not in self.valid_actions():
            raise ValueError(f"invalid action {a}")
        p = self.turn % 2
        self._valid = None
        if a == ACT_STEAL:
            # repaint, do not place: the referee moves the one chip's ownership
            cell = self.last
            self.bb[0] &= ~(1 << cell)
            self.bb[1] |= 1 << cell
            self.steal_used = True
        else:
            cell = (ROWS - 1 - self.fill[a]) * COLS + a
            self.bb[p] |= 1 << cell
            self.fill[a] += 1
        self.last = cell
        if self._four_through(p, cell):
            self._over = True
            self._winner = p
        self.turn += 1
        if not self._over and self.turn >= CELLS + (1 if self.steal_used else 0):
            self._over = True
            self._winner = -1

    # -- fast copy for search ------------------------------------------------

    def get_state(self):
        return (self.bb[0], self.bb[1], tuple(self.fill), self.turn,
                self.steal_used, self.last, self._over, self._winner)

    def set_state(self, s) -> None:
        (b0, b1, fill, self.turn, self.steal_used, self.last,
         self._over, self._winner) = s
        self.bb = [b0, b1]
        self.fill = list(fill)
        self._valid = None


# -- encoder -------------------------------------------------------------------
# Layout, and the reason for every block. The raw planes alone were what left the
# tic-tac-toe value head explaining 3.5% of outcome variance; adding derived
# features -- the same position restated in terms the game is actually about --
# was the only change that moved it. Connect 4's own idiom is threats and
# threat parity, so that is what the tail encodes.
#
#   [0:63]     my chips                    (row 0 = top, cell = r*9+c)
#   [63:126]   their chips
#   [126:189]  my threat squares           empty cells that complete a 4 for me
#   [189:252]  their threat squares        (count of lines / 3)
#   [252:261]  legal columns
#   [261:270]  column fill / 7
#   [270:279]  I win by dropping in column c right now
#   [279:288]  they win by dropping in column c right now
#   [288:297]  dropping in c hands them the win directly above  (the C4 trap)
#   [297]      steal is available to me this turn
#   [298]      turnIndex / 63
#   [299]      I am the first player
#   [300]      my chips - their chips      (steal makes this informative)
#   [301:305]  threat squares by row parity measured from the bottom row:
#              mine-even, mine-odd, theirs-even, theirs-odd   (/8)
N_IN = 305

ROW_FROM_BOTTOM = tuple(ROWS - 1 - (cell // COLS) for cell in range(CELLS))
CELL_COL = tuple(cell % COLS for cell in range(CELLS))
CELL_MIRROR = tuple((cell // COLS) * COLS + (COLS - 1 - cell % COLS)
                    for cell in range(CELLS))
COL_MIRROR = tuple(COLS - 1 - c for c in range(COLS))


def threat_squares(bb_me: int, bb_op: int):
    """[63] counts: how many 4-lines this empty cell would complete for `me`."""
    out = [0] * CELLS
    for i, m in enumerate(LINE_MASKS):
        mm = bb_me & m
        if (bb_op & m) == 0 and _pc(mm) == 3:
            e = m & ~mm
            out[e.bit_length() - 1] += 1
    return out


def encode_planes(eng: Engine):
    """Feature vector for the side to move, length N_IN.

    Always the side to move -- taking a perspective argument is a standing
    invitation for the three implementations of this function (here, the tensor
    engine, the emitted C++) to disagree about whose turn it is.
    """
    me = eng.turn % 2
    opp = 1 - me
    bm, bo = eng.bb[me], eng.bb[opp]
    f = [0.0] * N_IN
    for cell in range(CELLS):
        f[cell] = float((bm >> cell) & 1)
        f[63 + cell] = float((bo >> cell) & 1)

    tm = threat_squares(bm, bo)
    to = threat_squares(bo, bm)
    for cell in range(CELLS):
        f[126 + cell] = tm[cell] / 3.0
        f[189 + cell] = to[cell] / 3.0

    va = eng.valid_actions()
    for c in range(COLS):
        legal = 1.0 if c in va else 0.0
        f[252 + c] = legal
        f[261 + c] = eng.fill[c] / float(ROWS)
        if not legal:
            continue
        land = (ROWS - 1 - eng.fill[c]) * COLS + c
        f[270 + c] = 1.0 if tm[land] else 0.0
        f[279 + c] = 1.0 if to[land] else 0.0
        if eng.fill[c] + 1 < ROWS:
            f[288 + c] = 1.0 if to[land - COLS] else 0.0

    f[297] = 1.0 if ACT_STEAL in va else 0.0
    f[298] = eng.turn / float(CELLS)
    f[299] = 1.0 if me == 0 else 0.0
    f[300] = float(_pc(bm) - _pc(bo))
    for cell in range(CELLS):
        odd = ROW_FROM_BOTTOM[cell] & 1
        f[301 + odd] += tm[cell] / 8.0
        f[303 + odd] += to[cell] / 8.0
    return f


# -- scripted opponents (OFFLINE measurement only) -----------------------------

class RandomPolicy:
    name = "random"

    def __init__(self, seed=0, budget_ms=None):
        self.rng = random.Random(seed)

    def act(self, eng):
        return self.rng.choice(sorted(eng.valid_actions()))


class GreedyPolicy:
    """Win now; else block their win now; else avoid handing them the cell above;
    else prefer the centre. One ply of tactics, no search."""
    name = "greedy"

    def __init__(self, seed=0, budget_ms=None):
        self.rng = random.Random(seed)

    def act(self, eng):
        me = eng.turn % 2
        bm, bo = eng.bb[me], eng.bb[1 - me]
        tm, to = threat_squares(bm, bo), threat_squares(bo, bm)
        cols = sorted(a for a in eng.valid_actions() if a != ACT_STEAL)
        land = {c: (ROWS - 1 - eng.fill[c]) * COLS + c for c in cols}
        for c in cols:
            if tm[land[c]]:
                return c
        for c in cols:
            if to[land[c]]:
                return c
        safe = [c for c in cols
                if eng.fill[c] + 1 >= ROWS or not to[land[c] - COLS]]
        pool = safe or cols
        centre = min(abs(c - COLS // 2) for c in pool)
        best = [c for c in pool if abs(c - COLS // 2) == centre]
        return self.rng.choice(best)


class ABPolicy:
    """Time-budgeted iterative-deepening alpha-beta over a hand-tuned linear
    evaluation. A second offline opinion, nothing more: the tic-tac-toe run
    measured that a single scripted opponent has almost no resolution as an
    instrument, so this exists to be disagreed with, not believed."""
    name = "ab"

    def __init__(self, seed=0, budget_ms=90):
        self.rng = random.Random(seed)
        self.budget = (budget_ms or 90) / 1000.0

    def _eval(self, eng, me):
        bm, bo = eng.bb[me], eng.bb[1 - me]
        tm, to = threat_squares(bm, bo), threat_squares(bo, bm)
        s = 0.0
        for cell in range(CELLS):
            w = 1.0 + 0.5 * (1 - (ROW_FROM_BOTTOM[cell] & 1))
            s += w * (tm[cell] - to[cell])
            if (bm >> cell) & 1:
                s += 0.15 * (4 - abs(CELL_COL[cell] - COLS // 2))
            elif (bo >> cell) & 1:
                s -= 0.15 * (4 - abs(CELL_COL[cell] - COLS // 2))
        return s

    def _neg(self, eng, me, depth, alpha, beta, deadline):
        if eng.game_over:
            if eng.winner < 0:
                return 0.0
            return 1e6 if eng.winner == me else -1e6
        if depth == 0 or time.perf_counter() > deadline:
            sign = 1.0 if eng.current_player == me else -1.0
            return sign * self._eval(eng, me)
        st = eng.get_state()
        maxi = eng.current_player == me
        best = -1e9 if maxi else 1e9
        for a in sorted(eng.valid_actions(),
                        key=lambda a: abs((a if a != ACT_STEAL else 4) - COLS // 2)):
            eng.play(a)
            v = self._neg(eng, me, depth - 1, alpha, beta, deadline)
            eng.set_state(st)
            if maxi:
                best = max(best, v)
                alpha = max(alpha, best)
            else:
                best = min(best, v)
                beta = min(beta, best)
            if alpha >= beta:
                break
        return best

    def act(self, eng):
        me = eng.turn % 2
        deadline = time.perf_counter() + self.budget
        st = eng.get_state()
        va = sorted(eng.valid_actions())
        best = self.rng.choice(va)
        for depth in range(1, 9):
            bv, bm = -1e9, best
            for a in va:
                eng.play(a)
                v = self._neg(eng, me, depth - 1, -1e9, 1e9, deadline)
                eng.set_state(st)
                if v > bv:
                    bv, bm = v, a
            if time.perf_counter() > deadline and depth > 1:
                break
            best = bm
            if bv >= 1e6:
                break
        return best


class StealPolicy(GreedyPolicy):
    """Greedy, but always takes the steal. Exists so the venue-parity fuzz can
    drive the one rule this arena adds instead of hitting it one game in ten."""
    name = "steal"

    def act(self, eng):
        if ACT_STEAL in eng.valid_actions():
            return ACT_STEAL
        return GreedyPolicy.act(self, eng)


POLICIES = {"random": RandomPolicy, "greedy": GreedyPolicy, "ab": ABPolicy,
            "steal": StealPolicy}


def run_game(eng, pol0, pol1):
    while not eng.game_over:
        eng.play((pol0 if eng.current_player == 0 else pol1).act(eng))
    return eng.winner


def cmd_match(args):
    a_wins = b_wins = draws = 0
    for g in range(args.games):
        pa = POLICIES[args.a](seed=args.seed + g, budget_ms=args.budget_ms)
        pb = POLICIES[args.b](seed=args.seed + 10000 + g, budget_ms=args.budget_ms)
        a_first = g % 2 == 0
        w = run_game(Engine(), pa if a_first else pb, pb if a_first else pa)
        if w < 0:
            draws += 1
        elif (w == 0) == a_first:
            a_wins += 1
        else:
            b_wins += 1
    n = args.games
    print(f"{args.a} vs {args.b}: {a_wins}-{b_wins}-{draws} over {n} games "
          f"(seats alternate), score {(a_wins + 0.5 * draws) / n:.3f}")
    return 0


def load_other(path: str):
    spec = importlib.util.spec_from_file_location("other_engine", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Engine


def cmd_fuzz(args):
    Other = load_other(args.other)
    rng = random.Random(args.seed)
    plies = 0
    t0 = time.time()
    for g in range(args.games):
        gseed = rng.randrange(2 ** 30)
        grng = random.Random(gseed)
        mine, other = Engine(), Other()
        while True:
            vm, vo = mine.valid_actions(), other.valid_actions()
            ctx = f"game {g} seed {gseed} ply {plies}"
            if vm != vo:
                print(f"DIVERGENCE (valid sets) {ctx}\n mine-only: "
                      f"{sorted(vm - vo)}\n other-only: {sorted(vo - vm)}")
                return 1
            if mine.game_over != other.game_over:
                print(f"DIVERGENCE (game_over) {ctx}: "
                      f"{mine.game_over} vs {other.game_over}")
                return 1
            if mine.game_over:
                if mine.winner != other.winner:
                    print(f"DIVERGENCE (winner) {ctx}: "
                          f"{mine.winner} vs {other.winner}")
                    return 1
                break
            a = grng.choice(sorted(vm))
            mine.play(a)
            other.play(a)
            plies += 1
    print(f"PARITY OK: {args.games} games, {plies} plies, 0 divergences "
          f"({time.time() - t0:.1f}s)")
    return 0


def cmd_cg(args):
    """CodinGame protocol adapter for Connect 4.

    Reconstructs the board from the referee's own rows every turn, then asserts
    that its independently computed valid-action set matches the one the referee
    sent. Mismatches print to stderr (visible in replays); the referee is
    trusted for actual play. This is the harness the Java-side fuzz driver runs.
    """
    eng = Engine()
    pol = POLICIES[args.policy](seed=args.seed, budget_ms=args.budget_ms)
    my_id, _opp = map(int, input().split())
    mismatches = 0
    while True:
        try:
            turn = int(input())
        except EOFError:
            return 0
        rows = [input().strip() for _ in range(ROWS)]
        n = int(input())
        ref = {int(input()) for _ in range(n)}
        ref = {ACT_STEAL if a == -2 else a for a in ref}
        int(input())                                   # oppPreviousAction

        # rebuild from the referee's board, so a desync cannot hide
        bb = [0, 0]
        fill = [0] * COLS
        for r, line in enumerate(rows):
            for c, ch in enumerate(line):
                if ch != ".":
                    bb[int(ch)] |= 1 << (r * COLS + c)
                    fill[c] += 1
        chips = sum(fill)
        eng.bb, eng.fill, eng.turn = bb, fill, turn
        eng.steal_used = (turn - chips) == 1
        eng.last = -1
        if turn == 1:
            eng.last = (bb[0] | bb[1]).bit_length() - 1
        eng._over, eng._winner, eng._valid = False, -1, None
        if eng.valid_actions() != ref:
            mismatches += 1
            print(f"PARITY MISMATCH turn {turn}: ref {sorted(ref)} vs replica "
                  f"{sorted(eng.valid_actions())}", file=sys.stderr, flush=True)
            eng._valid = ref
        if turn % 2 != my_id:
            print(f"PARITY SEAT turn {turn} my_id {my_id}", file=sys.stderr,
                  flush=True)
        a = pol.act(eng)
        print(-2 if a == ACT_STEAL else a, flush=True)
        print(f"PREDICT turn={turn} steal_used={eng.steal_used} "
              f"mismatches={mismatches}", file=sys.stderr, flush=True)


def cmd_bench(args):
    rng = random.Random(0)
    plies = 0
    t0 = time.perf_counter()
    for _ in range(2000):
        eng = Engine()
        while not eng.game_over:
            eng.play(rng.choice(sorted(eng.valid_actions())))
            plies += 1
    dt = time.perf_counter() - t0
    print(f"{plies} plies in {dt:.2f}s = {plies / dt:,.0f} plies/s "
          f"(reference engine, random playouts)")
    return 0


# == the tensor engine =========================================================
# House rule (memory: maximize-gpu-usage): the ENVIRONMENT goes on the GPU, not
# just the net. Connect 4 is friendlier than UTTT here -- 63 cells, 10 actions,
# and win/threat detection is one 63x126 matmul against the line-membership
# matrix, which is the same op that produces every derived feature.

_TABLE_CACHE = {}


def _tables(device):
    import torch
    key = str(device)
    hit = _TABLE_CACHE.get(key)
    if hit is not None:
        return hit
    lmat = torch.zeros(len(LINES), CELLS, device=device)
    for i, L in enumerate(LINES):
        for c in L:
            lmat[i, c] = 1.0
    odd = torch.tensor([float(ROW_FROM_BOTTOM[c] & 1) for c in range(CELLS)],
                       device=device)
    out = (lmat, odd, 1.0 - odd,
           torch.arange(COLS, device=device),
           torch.tensor(CELL_MIRROR, dtype=torch.long, device=device),
           torch.tensor(COL_MIRROR, dtype=torch.long, device=device))
    _TABLE_CACHE[key] = out
    return out


class TensorC4:
    """B independent Connect 4 games advanced in lockstep on GPU.

      pl     [B,2,63] float32  occupancy planes, cell = row*9+col, row 0 top
      fill   [B,9]    long     chips per column
      turn   [B]      long     the referee's turnIndex
      stl    [B]      bool     a steal has been used in this game
      last   [B]      long     most recently changed cell
      done   [B]      bool
      winner [B]      long     0/1, or -1 for a draw / still running
    """

    def __init__(self, batch, device):
        import torch
        self.B = batch
        self.device = device
        (self.LMAT, self.ODD, self.EVEN, self.CIDX,
         self.CMIR, self.KMIR) = _tables(device)
        self.pl = torch.zeros(batch, 2, CELLS, device=device)
        self.fill = torch.zeros(batch, COLS, dtype=torch.long, device=device)
        self.turn = torch.zeros(batch, dtype=torch.long, device=device)
        self.stl = torch.zeros(batch, dtype=torch.bool, device=device)
        self.last = torch.full((batch,), -1, dtype=torch.long, device=device)
        self.done = torch.zeros(batch, dtype=torch.bool, device=device)
        self.winner = torch.full((batch,), -1, dtype=torch.long, device=device)
        self.ar = torch.arange(batch, device=device)

    @property
    def side(self):
        return self.turn % 2

    # -- observation ---------------------------------------------------------

    def legal_mask(self):
        """[B,10] bool: nine columns then the steal."""
        import torch
        cols = self.fill < ROWS
        steal = (self.turn == 1).unsqueeze(1)
        return torch.cat([cols, steal], dim=1) & ~self.done.unsqueeze(1)

    def landing(self):
        """[B,9] long: the cell a chip would settle in. Garbage where full, which
        every consumer masks with the legal columns."""
        return (ROWS - 1 - self.fill).clamp(min=0) * COLS + self.CIDX

    def tactics(self):
        """Everything one ply of tactics needs, shared by the encoder and the
        scripted GPU opponent so they cannot drift apart.

        Threat squares are one matmul against the line-membership matrix: a line
        with three of mine and none of theirs contributes its single empty cell.
        """
        me = self.side
        mine = self.pl[self.ar, me]
        theirs = self.pl[self.ar, 1 - me]
        cm = mine @ self.LMAT.T
        co = theirs @ self.LMAT.T
        empty = 1.0 - mine - theirs
        tm = (((cm > 2.5) & (co < 0.5)).float() @ self.LMAT) * empty
        to = (((co > 2.5) & (cm < 0.5)).float() @ self.LMAT) * empty

        legal = self.legal_mask()
        lc = legal[:, :COLS].float()
        land = self.landing()
        win_me = (tm.gather(1, land) > 0).float() * lc
        win_op = (to.gather(1, land) > 0).float() * lc
        room = ((self.fill + 1) < ROWS).float()
        above = (land - COLS).clamp(min=0)
        gives = (to.gather(1, above) > 0).float() * room * lc
        return mine, theirs, tm, to, legal, lc, win_me, win_op, gives

    def encode(self):
        """[B,N_IN] float32, identical to encode_planes() feature for feature.
        `gpu-parity --check-encode` asserts that."""
        import torch
        mine, theirs, tm, to, legal, lc, win_me, win_op, gives = self.tactics()
        me = self.side
        odd_m = (tm * self.ODD).sum(1, keepdim=True) / 8.0
        even_m = (tm * self.EVEN).sum(1, keepdim=True) / 8.0
        odd_o = (to * self.ODD).sum(1, keepdim=True) / 8.0
        even_o = (to * self.EVEN).sum(1, keepdim=True) / 8.0
        return torch.cat([
            mine, theirs, tm / 3.0, to / 3.0,
            lc, self.fill.float() / float(ROWS), win_me, win_op, gives,
            legal[:, COLS:COLS + 1].float(),
            (self.turn.float() / float(CELLS)).unsqueeze(1),
            (me == 0).float().unsqueeze(1),
            (mine.sum(1) - theirs.sum(1)).unsqueeze(1),
            even_m, odd_m, even_o, odd_o,
        ], dim=1)

    # -- transition ----------------------------------------------------------

    def step(self, acts):
        """Apply one action per game. Illegal or finished entries are no-ops, so
        a caller may speculatively step every action of every game."""
        import torch
        legal = self.legal_mask()
        ok = legal.gather(1, acts.unsqueeze(1)).squeeze(1)
        if not bool(ok.any()):
            return
        side = self.side
        is_steal = ok & (acts == ACT_STEAL)
        is_drop = ok & ~is_steal

        col = acts.clamp(max=COLS - 1)
        land = (ROWS - 1 - self.fill.gather(1, col.unsqueeze(1)).squeeze(1)
                ).clamp(min=0) * COLS + col

        di = self.ar[is_drop]
        if di.numel():
            self.pl[di, side[is_drop], land[is_drop]] = 1.0
            self.fill[di, col[is_drop]] += 1
        si = self.ar[is_steal]
        if si.numel():
            cell = self.last[is_steal]
            self.pl[si, 0, cell] = 0.0
            self.pl[si, 1, cell] = 1.0
            self.stl[si] = True
        self.last = torch.where(is_drop, land, self.last)

        # a win can only be created by the cell that just changed, so a global
        # test over the mover's plane is equivalent -- and one matmul
        moved = self.pl[self.ar, side]
        won = ((moved @ self.LMAT.T) > 3.5).any(dim=1) & ok
        self.winner = torch.where(won, side, self.winner)
        self.done = self.done | won

        self.turn = self.turn + ok.long()
        drw = (~self.done) & (self.turn >= CELLS + self.stl.long())
        self.winner = torch.where(drw, torch.full_like(self.winner, -1),
                                  self.winner)
        self.done = self.done | drw

    def reset_done(self):
        d = self.done
        if not bool(d.any()):
            return 0
        n = int(d.sum())
        self.pl[d] = 0.0
        self.fill[d] = 0
        self.turn[d] = 0
        self.stl[d] = False
        self.last[d] = -1
        self.winner[d] = -1
        self.done[d] = False
        return n

    def clone_repeat(self, n):
        c = TensorC4(self.B * n, self.device)
        rep = lambda x: x.repeat_interleave(n, dim=0).clone()
        c.pl = rep(self.pl)
        c.fill = rep(self.fill)
        c.turn = rep(self.turn)
        c.stl = rep(self.stl)
        c.last = rep(self.last)
        c.done = rep(self.done)
        c.winner = rep(self.winner)
        return c


def _exact_or_value(g, v, me):
    """My value at position `g`: the exact result where the game ended there,
    the value head otherwise. `v` must already be from g's side-to-move view."""
    import torch
    exact = torch.where(g.winner == me, torch.ones_like(v),
                        torch.where(g.winner < 0, torch.zeros_like(v),
                                    -torch.ones_like(v)))
    sign = torch.where(g.side == me, torch.ones_like(v), -torch.ones_like(v))
    return torch.where(g.done, exact, sign * v)


def improved_policy(t, net, tau=1.0, depth=2):
    """Policy improvement by exhaustive shallow search, entirely on GPU.

    depth 1: play all 10 actions of all games in one cloned batch and score the
    children with the value head.
    depth 2: expand again and take the opponent's BEST reply to each of my
    actions -- exact minimax over 100 grandchildren per game. Connect 4 is
    decided by one-move-deep tactics ("that drop hands them the cell above"), so
    a depth-1 target is systematically blind to the thing the game is about.
    Eighty-one actions made this unaffordable in the tic-tac-toe repo; nine make
    it a 100x batch on a GPU that was not the bottleneck anyway.

    Terminals are exact at both levels; only non-terminal leaves use the net.
    """
    import torch
    B, dev = t.B, t.device
    legal = t.legal_mask()
    me = t.side
    kids = t.clone_repeat(N_ACT)
    kids.step(torch.arange(N_ACT, device=dev).repeat(B))
    me_k = me.repeat_interleave(N_ACT)

    if depth <= 1:
        with torch.no_grad():
            _, v = net(kids.encode())
        q = _exact_or_value(kids, v, me_k).view(B, N_ACT)
    else:
        legal_b = kids.legal_mask()
        gk = kids.clone_repeat(N_ACT)
        gk.step(torch.arange(N_ACT, device=dev).repeat(B * N_ACT))
        with torch.no_grad():
            _, vg = net(gk.encode())
        me_g = me_k.repeat_interleave(N_ACT)
        vg = _exact_or_value(gk, vg, me_g).view(B * N_ACT, N_ACT)
        # the opponent moves next, so they pick the reply worst for me
        worst = vg.masked_fill(~legal_b, 1e9).min(dim=1).values
        # unless my move already ended the game, in which case it is settled and
        # there is no reply to minimise over
        settled = torch.where(kids.winner == me_k, torch.ones_like(worst),
                              torch.where(kids.winner < 0,
                                          torch.zeros_like(worst),
                                          -torch.ones_like(worst)))
        q = torch.where(kids.done, settled, worst).view(B, N_ACT)

    q = q.masked_fill(~legal, -1e9)
    pi = torch.softmax(q / tau, dim=1) * legal
    pi = pi / pi.sum(dim=1, keepdim=True).clamp_min(1e-9)
    return q, pi, legal


def augment(x, pi, cmir, kmir):
    """The board's ONLY symmetry: left-right mirror. Connect 4 has no rotations
    (gravity picks a direction), so this is 2x free data, not 8x.

    The feature vector is not uniform -- four per-cell planes take the cell
    permutation, five per-column blocks take the column permutation, and the
    trailing scalars are invariant. Treating it as one array is how a symmetry
    silently corrupts a training set.
    """
    import torch
    n = x.shape[0]
    parts = [x[:, :4 * CELLS].view(n, 4, CELLS)[:, :, cmir].reshape(n, 4 * CELLS)]
    for off in (252, 261, 270, 279, 288):
        parts.append(x[:, off:off + COLS][:, kmir])
    parts.append(x[:, 297:])
    pa = torch.cat([pi[:, :COLS][:, kmir], pi[:, COLS:]], dim=1)
    return torch.cat(parts, dim=1), pa


def gpu_greedy(t, jitter=0.0):
    """The GreedyPolicy tactics, vectorised: win now, else block, else avoid
    handing them the cell above, else nearest the centre. The training-loop
    yardstick; never submitted.

    `jitter` perturbs only the centre preference, by less than the gap between
    tiers, so it randomises tie-breaks without ever overriding a win or a block.
    Without it every game in an evaluation batch is the same game -- which is how
    a benchmark reports a confident 0.500 that means nothing.
    """
    import torch
    _, _, _, _, legal, lc, win_me, win_op, gives = t.tactics()
    centre = torch.tensor([-abs(c - COLS // 2) for c in range(COLS)],
                          dtype=torch.float32, device=t.device)
    score = centre.unsqueeze(0).expand(t.B, COLS) * 0.1
    if jitter:
        score = score + torch.rand(t.B, COLS, device=t.device) * jitter
    score = score - gives * 5.0 + win_op * 50.0 + win_me * 500.0
    score = score.masked_fill(lc < 0.5, -1e9)
    return score.argmax(dim=1)


# == the net ===================================================================
# Budget arithmetic against the measured 100,000-UTF-8-byte cap, int8 weights in
# base85 (4 bytes -> 5 chars = 1.25 chars per weight):
#
#   305 -> 160 -> 112, policy head 112->10, value head 112->1
#   = 48,800 + 17,920 + 1,120 + 112 = 67,952 weights
#   67,952 * 1.25            = 84,940 chars of weights
#   + base85 chunk quoting   =    855
#   + 282 float biases       =  2,800
#   + C++ inference + search =  9,800
#   ------------------------------------------------  98,450 of 100,000 (MEASURED)
#
# Width is a budget decision, not a taste one: 160-112 is the widest that still
# packs under the cap with this harness, measured by emitting at six widths
# (160-120 overshoots by 266 bytes). Nine actions instead of eighty-one is what
# buys the wider trunk here; 305 inputs is what spends it back.
AZ_TRUNK1 = int(os.environ.get("NAPKIN_TRUNK1", 160))
AZ_TRUNK2 = int(os.environ.get("NAPKIN_TRUNK2", 112))


def build_aznet(device="cpu", t1=None, t2=None):
    import torch
    import torch.nn as nn
    t1 = AZ_TRUNK1 if t1 is None else t1
    t2 = AZ_TRUNK2 if t2 is None else t2

    class _AZ(nn.Module):
        def __init__(self):
            super().__init__()
            self.t1 = nn.Linear(N_IN, t1)
            self.t2 = nn.Linear(t1, t2)
            self.ph = nn.Linear(t2, N_ACT)
            self.vh = nn.Linear(t2, 1)

        def forward(self, x):
            h = torch.relu(self.t1(x))
            h = torch.relu(self.t2(h))
            return self.ph(h), torch.tanh(self.vh(h)).squeeze(-1)

    return _AZ().to(device)


def load_aznet(path, device="cpu"):
    """Build from the shape STORED IN THE CHECKPOINT, never from the ambient
    constants, which are environment-configurable."""
    import torch
    ck = torch.load(path, map_location=device)
    shape = ck.get("shape")
    if not shape or len(shape) != 4:
        raise ValueError(f"{path}: missing or malformed 'shape'")
    net = build_aznet(device, t1=shape[1], t2=shape[2])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, shape


def aznet_param_bytes():
    return (N_IN * AZ_TRUNK1 + AZ_TRUNK1 * AZ_TRUNK2
            + AZ_TRUNK2 * N_ACT + AZ_TRUNK2)


def random_openings(t, k):
    """Advance a batch by `k` uniformly-random legal plies, in PAIRS.

    This is the only honest way I have found to diversify a match between two
    deterministic players. The alternatives both failed measurably here:
    perturbing one player's own policy handicaps it (0.712 vs greedy), and
    jittering the opponent's tie-breaks does not diversify at all -- 1024 games
    produced 5 distinct final positions, so a 1.000 score meant "won 5 games".
    Randomising the POSITION instead leaves both players at full strength.

    Games 2i and 2i+1 receive the same opening and opposite seats, so the result
    is a paired comparison and seat advantage cancels exactly.
    """
    import torch
    for _ in range(k):
        lg = t.legal_mask()[::2].float()
        a = torch.multinomial(lg + 1e-9, 1).squeeze(1)
        t.step(a.repeat_interleave(2))


def _distinct(t):
    """How many distinct final positions a batch actually produced. Reported
    alongside every score, because a benchmark that has quietly collapsed to a
    handful of games looks exactly like a benchmark that works."""
    import torch
    return len({tuple(r.tolist())
                for r in (t.pl[:, 0] + 2 * t.pl[:, 1]).to(torch.int8)})


def play_out(net, other, device, games, seed, open_plies=8, depth=2):
    """Net vs `other` from random paired openings, both playing their best.

    `other` is called as other(t) -> [B] actions; pass a net-backed closure to
    measure against a past self, or gpu_greedy for the scripted yardstick.
    Returns (score, wins, draws, distinct_final_positions).
    """
    import torch
    games -= games % 2
    torch.manual_seed(seed)
    t = TensorC4(games, device)
    random_openings(t, open_plies)
    seat = torch.arange(games, device=device) % 2
    while not bool(t.done.all()):
        _, pi, _ = improved_policy(t, net, tau=0.3, depth=depth)
        t.step(torch.where(t.side == seat, pi.argmax(dim=1), other(t)))
    win = (t.winner == seat).float()
    draw = (t.winner < 0).float()
    return (float((win + 0.5 * draw).mean()), float(win.mean()),
            float(draw.mean()), _distinct(t))


def net_mover(other_net, depth=2):
    def act(t):
        _, pi, _ = improved_policy(t, other_net, tau=0.3, depth=depth)
        return pi.argmax(dim=1)
    return act


def evaluate_vs_greedy(net, device, games, seed, open_plies=8, depth=2):
    """Score against the scripted yardstick. Returns (score, distinct)."""
    sc, _, _, n = play_out(net, gpu_greedy, device, games, seed,
                           open_plies=open_plies, depth=depth)
    return sc, n


def best_response_step(t, learner_seat, learner_mv, opp_mv):
    """Who moves, and whose plies get recorded, in best-response mode.

    Returns (mv, record): `mv` takes the learner's action on the learner's turns
    and the frozen opponent's on the others; `record` is true only for live
    learner plies.

    Extracted so it can be asserted on. Recording an OPPONENT ply would store it
    with the learner's outcome sign, which silently inverts the value target for
    that sample -- the same class of bug that cost the tic-tac-toe repo two
    debugging rounds ("negating is right in pure self-play but wrong against a
    fixed opponent").
    """
    import torch
    mine = t.side == learner_seat
    return torch.where(mine, learner_mv, opp_mv), (~t.done) & mine


def cmd_train_gpu(args):
    """Self-play on GPU with one-ply-improved policy targets.

    Everything except the optimiser step is batched tensor work: the games, the
    legality, the candidate expansion and the evaluation.
    """
    import torch
    import torch.nn.functional as F

    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if args.init:
        # continue from a trained net instead of restarting. The measured plateau
        # means a fresh run spends its first million games re-learning what this
        # checkpoint already knows.
        net, shape = load_aznet(args.init, device)
        net.train()
        if list(shape) != [N_IN, AZ_TRUNK1, AZ_TRUNK2, N_ACT]:
            # every dimension, not just the trunk: a checkpoint from a different
            # encoder loads with a cryptic state-dict error instead of a clear one
            raise ValueError(
                f"{args.init} has shape {list(shape)} but this build is "
                f"{[N_IN, AZ_TRUNK1, AZ_TRUNK2, N_ACT]}. Warm starting across a "
                f"different feature layout would train on the wrong inputs.")
        print(f"warm start from {args.init} (trunk {shape[1]}-{shape[2]})", flush=True)
    else:
        net = build_aznet(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    # A frozen opponent turns self-play into best-response training: only the
    # learner's transitions are stored, so the net is optimised specifically
    # against THAT policy. With --opponent pointing at our own deployed bot this
    # trains a nemesis, which measures how exploitable the deployed bot is.
    # Self-play's opponent distribution is a single policy (the current net); this
    # is the tool for asking whether that is why our offline edge does not reach
    # the ladder.
    opp_net = None
    if getattr(args, "opponent", None):
        opp_net, oshape = load_aznet(args.opponent, device)
        if list(oshape) != [N_IN, AZ_TRUNK1, AZ_TRUNK2, N_ACT]:
            raise ValueError(f"{args.opponent} has shape {list(oshape)} but this "
                             f"build is {[N_IN, AZ_TRUNK1, AZ_TRUNK2, N_ACT]}")
        for q in opp_net.parameters():
            q.requires_grad_(False)
        print(f"frozen opponent: {args.opponent} (best-response mode)", flush=True)

    B = args.batch_games
    t = TensorC4(B, device)
    MAXP = CELLS + 1
    # which seat the learner occupies, alternating so neither seat is favoured
    learner_seat = (torch.arange(B, device=device) % 2) if opp_net is not None else None

    st_x = torch.zeros(B, MAXP, N_IN, device=device)
    st_p = torch.zeros(B, MAXP, N_ACT, device=device)
    st_m = torch.zeros(B, MAXP, dtype=torch.long, device=device)
    plies = torch.zeros(B, dtype=torch.long, device=device)

    cap = args.buffer
    bf_x = torch.zeros(cap, N_IN, device=device)
    bf_p = torch.zeros(cap, N_ACT, device=device)
    bf_z = torch.zeros(cap, device=device)
    ptr = filled = games_done = 0
    ar = torch.arange(B, device=device)
    _, _, _, _, cmir, kmir = _tables(device)
    t0 = time.time()
    loss = loss_p = loss_v = torch.zeros((), device=device)

    for it in range(1, args.iters + 1):
        for _ in range(args.steps_per_iter):
            x = t.encode()
            _, pi, legal = improved_policy(t, net, tau=args.tau,
                                            depth=args.depth)
            # Exploration belongs in the opening. Sampling every ply makes the
            # outcome nearly unpredictable, and then the value head is being
            # asked to regress noise.
            u = legal.float()
            u = u / u.sum(dim=1, keepdim=True).clamp_min(1e-9)
            act_p = (1 - args.eps) * pi + args.eps * u
            mv_s = torch.multinomial(act_p.clamp_min(1e-12), 1).squeeze(1)
            mv = torch.where(plies < args.opening_plies, mv_s, pi.argmax(dim=1))

            live = ~t.done
            if opp_net is not None:
                _, opi, _ = improved_policy(t, opp_net, tau=args.tau,
                                            depth=args.depth)
                mv, live = best_response_step(t, learner_seat, mv,
                                              opi.argmax(dim=1))
            slot = plies.clamp(max=MAXP - 1)
            st_x[ar, slot] = torch.where(live.unsqueeze(1), x, st_x[ar, slot])
            st_p[ar, slot] = torch.where(live.unsqueeze(1), pi, st_p[ar, slot])
            st_m[ar, slot] = torch.where(live, t.side, st_m[ar, slot])
            plies = plies + live.long()

            t.step(mv)

            fin = t.done
            if bool(fin.any()):
                w = t.winner.unsqueeze(1).expand(B, MAXP)
                z = torch.where(w < 0, torch.zeros(B, MAXP, device=device),
                                torch.where(w == st_m,
                                            torch.ones(B, MAXP, device=device),
                                            -torch.ones(B, MAXP, device=device)))
                valid = (torch.arange(MAXP, device=device).unsqueeze(0)
                         < plies.unsqueeze(1)) & fin.unsqueeze(1)
                nsel = int(valid.sum())
                if nsel:
                    idx = (ptr + torch.arange(nsel, device=device)) % cap
                    bf_x[idx] = st_x[valid]
                    bf_p[idx] = st_p[valid]
                    bf_z[idx] = z[valid]
                    ptr = int((ptr + nsel) % cap)
                    filled = min(cap, filled + nsel)
                games_done += int(fin.sum())
                plies = torch.where(fin, torch.zeros_like(plies), plies)
                t.reset_done()

        if filled < args.batch:
            continue
        for _ in range(args.updates_per_iter):
            idx = torch.randint(0, filled, (args.batch,), device=device)
            xb, pb = bf_x[idx], bf_p[idx]
            if args.augment and bool(torch.randint(0, 2, (1,)).item()):
                xb, pb = augment(xb, pb, cmir, kmir)
            logits, v = net(xb)
            loss_p = -(pb * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss_v = F.mse_loss(v, bf_z[idx])
            loss = loss_p + args.value_weight * loss_v
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

        if it % args.eval_every == 0 or it == args.iters:
            wr, nd = evaluate_vs_greedy(net, device, args.eval_games,
                                        args.seed + it, depth=args.depth)
            print(f"iter {it}/{args.iters} games {games_done} buf {filled} "
                  f"loss {float(loss):.4f} (p {float(loss_p):.4f} "
                  f"v {float(loss_v):.4f}) vs-greedy {wr:.3f} "
                  f"[{nd} distinct] ({time.time() - t0:.0f}s)", flush=True)
        ck = {"state_dict": net.state_dict(),
              "shape": [N_IN, AZ_TRUNK1, AZ_TRUNK2, N_ACT], "iter": it,
              "games": games_done}
        torch.save(ck, args.out)
        # past-self is the only yardstick with headroom once greedy is saturated,
        # and it does not exist unless it was saved on the way past
        if args.snapshot_every and it % args.snapshot_every == 0:
            torch.save(ck, f"{args.out.rsplit('.', 1)[0]}_it{it:05d}.pt")
    rate = games_done / max(1e-9, time.time() - t0)
    print(f"saved {args.out} after {games_done} self-play games "
          f"({rate:,.0f} games/s, {aznet_param_bytes()} weights)")
    return 0


def _wilson(p, n):
    import math
    z = 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - hw, c + hw


def cmd_eval_net(args):
    """Measure a checkpoint against a past self or the scripted yardstick.

    Once the net saturates greedy, the only measurement with headroom left is
    net-vs-past-self, which is why train-gpu snapshots. Every result reports the
    number of DISTINCT final positions: a score computed over five repeated games
    is not a score, and that failure is invisible without this column.
    """
    import torch
    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu")
    net, shape = load_aznet(args.net, device)
    if args.vs_net:
        ref, rshape = load_aznet(args.vs_net, device)
        if rshape[0] != shape[0]:
            raise ValueError(f"{args.vs_net} has {rshape[0]} inputs, "
                             f"{args.net} has {shape[0]}: not comparable")
        other, label = net_mover(ref, depth=args.depth), args.vs_net
    elif args.vs == "greedy":
        other, label = gpu_greedy, "greedy"
    else:
        def other(t):
            return torch.multinomial(t.legal_mask().float() + 1e-9, 1).squeeze(1)
        label = "random"
    sc, w, d, nd = play_out(net, other, device, args.games, args.seed,
                            open_plies=args.open_plies, depth=args.depth)
    n = args.games - args.games % 2
    lo, hi = _wilson(sc, n)
    print(f"{args.net} (trunk {shape[1]}-{shape[2]}) vs {label}: score {sc:.3f} "
          f"(95% Wilson {lo:.3f}..{hi:.3f}), win {w:.3f} draw {d:.3f}, "
          f"{n} games from {args.open_plies} random opening plies, "
          f"{nd} distinct final positions")
    return 0


def cmd_gpu_parity(args):
    """A new engine is worth nothing until it agrees with the verified one.

    Plays random games with the tensor engine and the reference engine in
    lockstep: legal-action SET every ply, outcome every game, and with
    --check-encode every one of the N_IN features.
    """
    import numpy as np
    import torch

    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    B = args.games
    t = TensorC4(B, device)
    refs = [Engine() for _ in range(B)]
    plies = 0
    while True:
        alive = [k for k in range(B) if not refs[k].game_over]
        if not alive:
            break
        gl = t.legal_mask().cpu().numpy()
        enc = t.encode().cpu().numpy() if args.check_encode else None
        for k in alive:
            ref = refs[k].valid_actions()
            got = set(np.nonzero(gl[k])[0].tolist())
            if ref != got:
                print(f"DIVERGENCE game {k} ply {plies}\n"
                      f"  ref-only {sorted(ref - got)}\n"
                      f"  gpu-only {sorted(got - ref)}")
                return 1
            if enc is not None:
                want = np.asarray(encode_planes(refs[k]), dtype=np.float32)
                if not np.allclose(enc[k], want, atol=1e-6):
                    bad = int(np.abs(enc[k] - want).argmax())
                    print(f"ENCODER DIVERGENCE game {k} ply {plies} feature "
                          f"{bad}: gpu {enc[k][bad]} ref {want[bad]}")
                    return 1
        choice = torch.zeros(B, dtype=torch.long, device=device)
        for k in alive:
            a = rng.choice(sorted(refs[k].valid_actions()))
            choice[k] = a
            refs[k].play(a)
        t.step(choice)
        plies += 1
        for k in alive:
            if refs[k].game_over != bool(t.done[k]):
                print(f"DONE MISMATCH game {k} ply {plies}: "
                      f"ref {refs[k].game_over} gpu {bool(t.done[k])}")
                return 1
            if refs[k].game_over and refs[k].winner != int(t.winner[k]):
                print(f"WINNER MISMATCH game {k}: ref {refs[k].winner} "
                      f"gpu {int(t.winner[k])}")
                return 1
    print(f"GPU PARITY OK: {B} games, {plies} plies, legal sets + outcomes"
          f"{' + encoder' if args.check_encode else ''} identical to the "
          f"reference engine")
    return 0


# == the packer ================================================================

# No '?': three of these characters in a row form a C trigraph ("??)", "??-"),
# and a weight payload whose meaning depends on the compiler's trigraph setting
# is not a payload. ',' replaces it. No '"' and no '\\' either, for the obvious
# reason. Asserted 85 distinct characters in selfcheck.
B85_ALPHABET = ("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "abcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>,@^_`{|}~")


def b85_encode(data: bytes) -> str:
    """RFC1924-style base85; 4 bytes -> 5 chars. Pads the tail with zeros."""
    out = []
    data = data + b"\x00" * ((-len(data)) % 4)
    for i in range(0, len(data), 4):
        v = int.from_bytes(data[i:i + 4], "big")
        chunk = []
        for _ in range(5):
            v, rem = divmod(v, 85)
            chunk.append(B85_ALPHABET[rem])
        out.append("".join(reversed(chunk)))
    return "".join(out)


def b85_decode(text: str, n: int) -> bytes:
    inv = {c: i for i, c in enumerate(B85_ALPHABET)}
    out = bytearray()
    for i in range(0, len(text) - 4, 5):
        v = 0
        for k in range(5):
            v = v * 85 + inv[text[i + k]]
        out.extend(v.to_bytes(4, "big"))
    return bytes(out[:n])


def quantize_int8(w):
    """Per-tensor symmetric quantisation. Returns (int8 array, scale)."""
    import numpy as np
    s = float(np.abs(w).max()) / 127.0
    if s == 0:
        s = 1e-8
    return np.clip(np.rint(w / s), -127, 127).astype(np.int8), s


CPP_TEMPLATE = r"""/* napkin-100k-connect4: a GPU-self-play-trained policy+value net, weights and
 * all, in one file, driving a negamax search. The value head scores leaves, the
 * policy head orders moves, terminals are exact. No hand-written evaluation.
 * Disclosed bot - github.com/arose26/napkin-100k-connect4, account Napkin100k.
 * Trunk %(shape)s, int8 weights decoded from base85 below. */
#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline")
#pragma GCC target("sse","sse2","sse3","ssse3","sse4","popcnt","abm","avx","avx2")
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cmath>

static const int N_IN=%(n_in)d, T1=%(t1)d, T2=%(t2)d, N_ACT=%(n_act)d;
static const int ROWS=7, COLS=9, CELLS=63, A_STEAL=9;
static const float S1=%(s1).9gf, S2=%(s2).9gf, SP=%(sp).9gf, SV=%(sv).9gf;
static const char* W85 =
%(w85)s;
static const float B1[]={%(b1)s};
static const float B2[]={%(b2)s};
static const float BP[]={%(bp)s};
static const float BV=%(bv).9gf;
static int8_t W[%(nw)d];

static void decodeWeights(){
    static const char* A="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                         "abcdefghijklmnopqrstuvwxyz!#$%%&()*+-;<=>,@^_`{|}~";
    int inv[256]; for(int i=0;i<256;i++) inv[i]=-1;
    for(int i=0;i<85;i++) inv[(unsigned char)A[i]]=i;
    size_t n=strlen(W85), out=0;
    for(size_t i=0;i+4<n;i+=5){
        uint32_t v=0;
        for(int k=0;k<5;k++) v=v*85u+(uint32_t)inv[(unsigned char)W85[i+k]];
        for(int k=3;k>=0;k--)
            if(out+(size_t)k<(size_t)%(nw)d) W[out+k]=(int8_t)((v>>(8*(3-k)))&0xFF);
        out+=4;
    }
}

static float h1_[T1], h2_[T2], pol[N_ACT], val;
static void forward(const float* x){
    const int8_t* w=W;
    for(int j=0;j<T1;j++){ float a=0.f; const int8_t* r=w+(size_t)j*N_IN;
        for(int i=0;i<N_IN;i++) a+=x[i]*(float)r[i];
        a=a*S1+B1[j]; h1_[j]=a>0.f?a:0.f; }
    w+=(size_t)N_IN*T1;
    for(int j=0;j<T2;j++){ float a=0.f; const int8_t* r=w+(size_t)j*T1;
        for(int i=0;i<T1;i++) a+=h1_[i]*(float)r[i];
        a=a*S2+B2[j]; h2_[j]=a>0.f?a:0.f; }
    w+=(size_t)T1*T2;
    for(int j=0;j<N_ACT;j++){ float a=0.f; const int8_t* r=w+(size_t)j*T2;
        for(int i=0;i<T2;i++) a+=h2_[i]*(float)r[i];
        pol[j]=a*SP+BP[j]; }
    w+=(size_t)T2*N_ACT;
    { float a=0.f; for(int i=0;i<T2;i++) a+=h2_[i]*(float)w[i];
      val=tanhf(a*SV+BV); }
}

/* the 126 four-in-a-rows, and the ones through each cell: built at startup so
   they cost run time instead of source bytes */
static uint64_t LM[126]; static int NLM=0;
static int LAT[63][16], LATn[63];
static void initLines(){
    static const int DR[4]={0,1,1,1}, DC[4]={1,0,1,-1};
    for(int r=0;r<ROWS;r++)for(int c=0;c<COLS;c++)for(int d=0;d<4;d++){
        uint64_t m=0; int ok=1;
        for(int k=0;k<4;k++){ int rr=r+DR[d]*k, cc=c+DC[d]*k;
            if(rr<0||rr>=ROWS||cc<0||cc>=COLS){ok=0;break;}
            m|=1ULL<<(rr*COLS+cc); }
        if(ok) LM[NLM++]=m;
    }
    for(int cell=0;cell<CELLS;cell++){ LATn[cell]=0;
        for(int i=0;i<NLM;i++) if(LM[i]>>cell&1ULL) LAT[cell][LATn[cell]++]=i; }
}

struct Pos{ uint64_t bb[2]; int fill[COLS]; int turn; int stl; int last; };
struct Undo{ int a, cell, fill, last; };

static inline int side(const Pos&p){ return p.turn&1; }
static int gen(const Pos&p,int* out){
    int n=0;
    for(int c=0;c<COLS;c++) if(p.fill[c]<ROWS) out[n++]=c;
    if(p.turn==1) out[n++]=A_STEAL;
    return n;
}
/* returns 1 if this action wins for the mover */
static inline int mk(Pos&p,int a,Undo&u){
    int s=side(p); u.a=a; u.last=p.last;
    if(a==A_STEAL){ u.cell=p.last; u.fill=-1;
        p.bb[0]&=~(1ULL<<u.cell); p.bb[1]|=1ULL<<u.cell; p.stl=1; }
    else { int cell=(ROWS-1-p.fill[a])*COLS+a; u.cell=cell; u.fill=p.fill[a];
        p.bb[s]|=1ULL<<cell; p.fill[a]++; p.last=cell; }
    p.turn++;
    uint64_t m=p.bb[s]; int cell=u.cell;
    for(int k=0;k<LATn[cell];k++){ uint64_t L=LM[LAT[cell][k]];
        if((m&L)==L) return 1; }
    return 0;
}
static inline void unmk(Pos&p,const Undo&u){
    p.turn--; int s=side(p);
    if(u.a==A_STEAL){ p.bb[1]&=~(1ULL<<u.cell); p.bb[0]|=1ULL<<u.cell; p.stl=0; }
    else { p.bb[s]&=~(1ULL<<u.cell); p.fill[u.a]=u.fill; }
    p.last=u.last;
}
static inline int isFull(const Pos&p){ return p.turn>=CELLS+(p.stl?1:0); }

static float feat[N_IN];
/* must match encode_planes() in napkin_c4.py feature for feature */
static void encode(const Pos&p){
    memset(feat,0,sizeof(feat));
    int me=side(p), op=1-me;
    uint64_t bm=p.bb[me], bo=p.bb[op];
    for(int i=0;i<CELLS;i++){
        feat[i]=(float)((bm>>i)&1ULL);
        feat[63+i]=(float)((bo>>i)&1ULL);
    }
    int tm[63], to[63]; memset(tm,0,sizeof(tm)); memset(to,0,sizeof(to));
    for(int i=0;i<NLM;i++){
        uint64_t L=LM[i], mm=bm&L, oo=bo&L;
        if(!oo&&__builtin_popcountll(mm)==3) tm[__builtin_ctzll(L&~mm)]++;
        if(!mm&&__builtin_popcountll(oo)==3) to[__builtin_ctzll(L&~oo)]++;
    }
    float om=0.f,em=0.f,oo_=0.f,eo=0.f;
    for(int i=0;i<CELLS;i++){
        feat[126+i]=(float)tm[i]/3.f;
        feat[189+i]=(float)to[i]/3.f;
        int odd=(ROWS-1-i/COLS)&1;
        if(odd){ om+=(float)tm[i]/8.f; oo_+=(float)to[i]/8.f; }
        else   { em+=(float)tm[i]/8.f; eo+=(float)to[i]/8.f; }
    }
    for(int c=0;c<COLS;c++){
        feat[261+c]=(float)p.fill[c]/(float)ROWS;
        if(p.fill[c]>=ROWS) continue;
        feat[252+c]=1.f;
        int land=(ROWS-1-p.fill[c])*COLS+c;
        feat[270+c]=tm[land]?1.f:0.f;
        feat[279+c]=to[land]?1.f:0.f;
        if(p.fill[c]+1<ROWS) feat[288+c]=to[land-COLS]?1.f:0.f;
    }
    feat[297]=(p.turn==1)?1.f:0.f;
    feat[298]=(float)p.turn/(float)CELLS;
    feat[299]=(me==0)?1.f:0.f;
    feat[300]=(float)(__builtin_popcountll(bm)-__builtin_popcountll(bo));
    feat[301]=em; feat[302]=om; feat[303]=eo; feat[304]=oo_;
}

static std::chrono::steady_clock::time_point deadline;
static bool timeUp=false; static long evals=0;
static inline bool tick(){
    if((++evals&31)==0&&std::chrono::steady_clock::now()>deadline) timeUp=true;
    return timeUp;
}

/* value from the side-to-move's point of view */
static float negamax(Pos&p,int depth,float alpha,float beta){
    if(isFull(p)) return 0.f;
    int mv[10]; int n=gen(p,mv);
    if(n==0) return 0.f;
    encode(p); forward(feat);
    if(depth==0||tick()) return val;
    float pr[10];
    for(int k=0;k<n;k++) pr[k]=pol[mv[k]];
    for(int k=1;k<n;k++){ int c=mv[k]; float q=pr[k]; int j=k-1;
        while(j>=0&&pr[j]<q){mv[j+1]=mv[j];pr[j+1]=pr[j];j--;} mv[j+1]=c;pr[j+1]=q; }
    float best=-2.f;
    for(int k=0;k<n;k++){
        Undo u; int won=mk(p,mv[k],u);
        float v = won ? 1.f : -negamax(p,depth-1,-beta,-alpha);
        unmk(p,u);
        if(v>best) best=v;
        if(best>alpha) alpha=best;
        if(alpha>=beta) break;
        if(timeUp) break;
    }
    return best;
}

int main(){
    initLines(); decodeWeights();
    int myId,oppId;
    if(scanf("%%d%%d",&myId,&oppId)!=2) return 0;
    bool first=true;
    while(true){
        Pos p; memset(&p,0,sizeof(p)); p.last=-1;
        if(scanf("%%d",&p.turn)!=1) return 0;
        int chips=0;
        for(int r=0;r<ROWS;r++){
            char row[64];
            if(scanf("%%63s",row)!=1) return 0;
            for(int c=0;c<COLS;c++){
                if(row[c]=='.') continue;
                int who=row[c]-'0';
                if(who<0||who>1) return 1;
                p.bb[who]|=1ULL<<(r*COLS+c); p.fill[c]++; chips++;
                p.last=r*COLS+c;
            }
        }
        /* a steal places no chip, so the shortfall identifies it exactly */
        p.stl = (p.turn-chips)==1 ? 1 : 0;
        int n; if(scanf("%%d",&n)!=1||n<1||n>10) return 1;
        int mv[10], m=0;
        for(int i=0;i<n;i++){ int a;
            if(scanf("%%d",&a)!=1) return 1;
            mv[m++] = (a==-2) ? A_STEAL : a; }
        int prev; if(scanf("%%d",&prev)!=1) return 1;

        /* the raw net at the root, reported every turn: check-pack compares
           these two numbers against an independent int8 reference */
        encode(p); forward(feat);
        int best=mv[0]; { float bp=-1e30f;
            for(int k=0;k<m;k++) if(pol[mv[k]]>bp){bp=pol[mv[k]];best=mv[k];} }
        fprintf(stderr,"v=%%.6f am=%%d\n",val,best);

        /* an immediate win is exact knowledge; never let the clock decide it */
        int winNow=-1;
        for(int k=0;k<m;k++){ Undo u; int w=mk(p,mv[k],u); unmk(p,u);
            if(w){ winNow=mv[k]; break; } }
        if(winNow>=0){
            printf("%%d\n", winNow==A_STEAL?-2:winNow); fflush(stdout); continue;
        }

        int budget = first?900:85; first=false;
        deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(budget);
        timeUp=false; evals=0;
        int reached=0;
        for(int depth=1;depth<=20;depth++){
            float bv=-2.f; int bm=best; bool aborted=false;
            for(int k=0;k<m;k++){
                Undo u; int won=mk(p,mv[k],u);
                float v = won?1.f:-negamax(p,depth-1,-2.f,2.f);
                unmk(p,u);
                if(timeUp&&depth>1){aborted=true;break;}
                if(v>bv){bv=v;bm=mv[k];}
            }
            if(aborted) break;
            best=bm; reached=depth;
            if(bv>=1.f) break;
            if(timeUp) break;
        }
        fprintf(stderr,"d=%%d e=%%ld\n",reached,evals);
        printf("%%d\n", best==A_STEAL?-2:best); fflush(stdout);
    }
}
"""


def cmd_pack(args):
    """Pack the trained policy+value net into one searching C++ source."""
    import numpy as np
    import torch

    ck = torch.load(args.net, map_location="cpu")
    sd = ck["state_dict"]
    n_in, t1, t2, n_act = ck["shape"]
    if n_in != N_IN or n_act != N_ACT:
        raise ValueError(
            f"{args.net} expects {n_in} inputs / {n_act} actions but this build's "
            f"encoder emits {N_IN}/{N_ACT}. Packing it would feed the net a "
            f"different feature layout than it trained on.")
    w1, b1 = sd["t1.weight"].numpy(), sd["t1.bias"].numpy()
    w2, b2 = sd["t2.weight"].numpy(), sd["t2.bias"].numpy()
    wp, bp = sd["ph.weight"].numpy(), sd["ph.bias"].numpy()
    wv, bv = sd["vh.weight"].numpy(), sd["vh.bias"].numpy()
    q1, s1 = quantize_int8(w1)
    q2, s2 = quantize_int8(w2)
    qp, sp = quantize_int8(wp)
    qv, sv = quantize_int8(wv)
    blob = q1.tobytes() + q2.tobytes() + qp.tobytes() + qv.tobytes()
    txt = b85_encode(blob)
    w85 = "\n".join(f'  "{txt[i:i + 500]}"' for i in range(0, len(txt), 500))
    fmt = lambda a: ",".join(f"{float(v):.6g}f" for v in np.asarray(a).ravel())
    src = CPP_TEMPLATE % {
        "shape": f"{n_in}-{t1}-{t2}", "n_in": n_in, "t1": t1, "t2": t2,
        "n_act": n_act, "s1": s1, "s2": s2, "sp": sp, "sv": sv,
        "w85": w85, "nw": len(blob), "b1": fmt(b1), "b2": fmt(b2),
        "bp": fmt(bp), "bv": float(bv[0]),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(src)

    # the encoding is verified here, not by eye: pull the blob back out of the
    # emitted source and require it bit-for-bit
    got = b85_decode(txt, len(blob))
    assert got == blob, "base85 round trip is not the identity"
    n = len(src.encode("utf-8"))
    print(f"packed {args.net} -> {args.out}: {n} UTF-8 bytes "
          f"({100000 - n} under the cap), {len(blob)} int8 weights, "
          f"blob round trip exact")
    return 0 if n <= 100000 else 1


def _int8_reference(net_path):
    """The same arithmetic the emitted C++ performs, sharing none of its code."""
    import numpy as np
    import torch
    ck = torch.load(net_path, map_location="cpu")
    sd = ck["state_dict"]
    q, s = {}, {}
    for k in ("t1", "t2", "ph", "vh"):
        q[k], s[k] = quantize_int8(sd[f"{k}.weight"].numpy())
        q[k] = q[k].astype(np.float32)
    b = {k: sd[f"{k}.bias"].numpy() for k in ("t1", "t2", "ph", "vh")}

    def fwd(x):
        h = np.maximum(q["t1"] @ x * s["t1"] + b["t1"], 0.0)
        h = np.maximum(q["t2"] @ h * s["t2"] + b["t2"], 0.0)
        pol = q["ph"] @ h * s["ph"] + b["ph"]
        val = float(np.tanh(q["vh"] @ h * s["vh"] + b["vh"])[0])
        return pol, val
    return fwd, ck["shape"]


def _feed(proc, eng):
    """Write one CG-format turn to a bot's stdin."""
    proc.stdin.write(f"{eng.turn}\n")
    occ = [["."] * COLS for _ in range(ROWS)]
    for p in (0, 1):
        for cell in range(CELLS):
            if (eng.bb[p] >> cell) & 1:
                occ[cell // COLS][cell % COLS] = str(p)
    for r in range(ROWS):
        proc.stdin.write("".join(occ[r]) + "\n")
    va = sorted(eng.valid_actions())
    proc.stdin.write(f"{len(va)}\n")
    for a in va:
        proc.stdin.write(f"{-2 if a == ACT_STEAL else a}\n")
    proc.stdin.write(f"{eng.last % COLS if eng.last >= 0 else -1}\n")
    proc.stdin.flush()
    return va


def _spawn(cpp):
    import subprocess
    import tempfile
    tmp = tempfile.mkdtemp()
    exe = os.path.join(tmp, "bot")
    subprocess.run(["g++", "-O2", "-o", exe, cpp], check=True)
    return tmp, exe


def cmd_check_pack(args):
    """The packer gate: does the emitted C++ compute the same forward pass?

    The bot prints its root value head and its policy argmax to stderr every
    turn. Those are compared against an independent numpy int8 reference over
    random legal positions. Any drift here invalidates every ladder claim
    downstream, so this runs before any submission.
    """
    import shutil
    import subprocess
    import numpy as np

    fwd, shape = _int8_reference(args.net)
    tmp, exe = _spawn(args.cpp)
    rng = random.Random(args.seed)
    worst_v = 0.0
    agree = total = 0
    try:
        for g in range(args.games):
            seat = g % 2
            eng = Engine()
            proc = subprocess.Popen([exe], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, bufsize=1)
            proc.stdin.write(f"{seat} {1 - seat}\n")
            proc.stdin.flush()
            try:
                while not eng.game_over:
                    if eng.current_player != seat:
                        eng.play(rng.choice(sorted(eng.valid_actions())))
                        continue
                    va = _feed(proc, eng)
                    x = np.asarray(encode_planes(eng), dtype=np.float32)
                    pol, val = fwd(x)
                    want_am = max(va, key=lambda a: pol[a])
                    line = ""
                    while "v=" not in line:
                        line = proc.stderr.readline()
                        if not line:
                            raise AssertionError("bot died before reporting")
                    got_v = float(line.split("v=")[1].split()[0])
                    got_am = int(line.split("am=")[1])
                    total += 1
                    worst_v = max(worst_v, abs(got_v - val))
                    if got_am == want_am or abs(pol[got_am] - pol[want_am]) < 1e-4:
                        agree += 1
                    out = proc.stdout.readline()
                    if not out:
                        raise AssertionError("bot died mid-game")
                    mv = int(out)
                    mv = ACT_STEAL if mv == -2 else mv
                    assert mv in eng.valid_actions(), f"illegal move {mv}"
                    eng.play(mv)
            finally:
                proc.kill()
                proc.wait()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    pct = 100.0 * agree / max(1, total)
    print(f"check-pack {args.cpp} (trunk {shape[1]}-{shape[2]}): {total} "
          f"positions, max |value drift| {worst_v:.6f} (tol {args.tol}), "
          f"policy argmax agreement {agree}/{total} ({pct:.2f}%)")
    return 0 if worst_v <= args.tol and pct >= args.min_agree else 1


def cmd_check_bot(args):
    """Correctness gate for the packed bot: every move legal, every forced win
    taken. The search knows terminals exactly, so a declined forced win means the
    value signs or the clock are wrong -- which is how the tic-tac-toe bot's time
    budget was caught."""
    import shutil
    import subprocess

    tmp, exe = _spawn(args.cpp)
    rng = random.Random(args.seed)
    offered = taken = moves = 0
    try:
        for g in range(args.games):
            seat = g % 2
            eng = Engine()
            proc = subprocess.Popen([exe], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, text=True, bufsize=1)
            proc.stdin.write(f"{seat} {1 - seat}\n")
            proc.stdin.flush()
            try:
                while not eng.game_over:
                    if eng.current_player != seat:
                        eng.play(rng.choice(sorted(eng.valid_actions())))
                        continue
                    va = sorted(eng.valid_actions())
                    st = eng.get_state()
                    wins = []
                    for a in va:
                        eng.play(a)
                        if eng.game_over and eng.winner == seat:
                            wins.append(a)
                        eng.set_state(st)
                    _feed(proc, eng)
                    line = proc.stdout.readline()
                    if not line:
                        raise AssertionError("bot died mid-game")
                    mv = int(line)
                    mv = ACT_STEAL if mv == -2 else mv
                    assert mv in eng.valid_actions(), \
                        f"ILLEGAL move {mv}; legal={va}"
                    moves += 1
                    if wins:
                        offered += 1
                        taken += mv in wins
                    eng.play(mv)
            finally:
                proc.kill()
                proc.wait()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"check-bot {args.cpp}: {moves} moves all legal, forced wins taken "
          f"{taken}/{offered}")
    return 0 if offered == taken else 1


def cmd_bench_net(args):
    """Packed bot vs a scripted baseline, both seats, with a Wilson interval.
    Remember what the tic-tac-toe run measured: one scripted opponent has almost
    no resolution. This is a smoke test, not a strength estimate."""
    import math
    import shutil
    import subprocess

    tmp, exe = _spawn(args.cpp)
    rng = random.Random(args.seed)
    w = l = d = 0
    try:
        for g in range(args.games):
            seat = g % 2
            eng = Engine()
            opp = POLICIES[args.vs](seed=args.seed + g, budget_ms=args.budget_ms)
            proc = subprocess.Popen([exe], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, text=True, bufsize=1)
            proc.stdin.write(f"{seat} {1 - seat}\n")
            proc.stdin.flush()
            try:
                while not eng.game_over:
                    if eng.current_player != seat:
                        eng.play(opp.act(eng))
                        continue
                    _feed(proc, eng)
                    line = proc.stdout.readline()
                    if not line:
                        raise AssertionError("bot died mid-game")
                    mv = int(line)
                    eng.play(ACT_STEAL if mv == -2 else mv)
            finally:
                proc.kill()
                proc.wait()
            if eng.winner < 0:
                d += 1
            elif eng.winner == seat:
                w += 1
            else:
                l += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n = args.games
    p = (w + 0.5 * d) / n
    z = 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    print(f"bench-net {args.cpp} vs {args.vs}: {w}-{l}-{d} over {n} games, "
          f"score {p:.3f} (95% Wilson {c - hw:.3f}..{c + hw:.3f})")
    return 0


CG_LB = "https://www.codingame.com/services/Leaderboards/getFilteredPuzzleLeaderboard"
CG_HANDLE = "22639068dad6ecdf6717bb383d739a954432057"  # Napkin100k, public


def cmd_snapshot(args):
    """Append one ladder snapshot (our rank + league sizes + top 5) to a JSONL.
    Unauthenticated and read-only; keep the cadence polite (daily)."""
    import datetime
    import json
    import urllib.request

    def post(body):
        req = urllib.request.Request(CG_LB, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=20))

    mine = post([args.arena, CG_HANDLE, "global",
                 {"active": True, "column": "KEYWORD", "filter": args.pseudo}])
    top = post([args.arena, "", "global",
                {"active": False, "column": "", "filter": ""}])
    me = (mine.get("users") or [None])[0]
    snap = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
        "arena": args.arena,
        "total_bots": mine.get("count"),
        "leagues": {k: v.get("divisionAgentsCount")
                    for k, v in (top.get("leagues") or {}).items()},
        "me": None if not me else {
            "pseudo": me.get("pseudo"), "global_rank": me.get("rank"),
            "score": me.get("score"),
            "league_index": (me.get("league") or {}).get("divisionIndex")},
        "top5": [{"rank": u["rank"], "pseudo": u["pseudo"], "score": u.get("score"),
                  "league_index": (u.get("league") or {}).get("divisionIndex")}
                 for u in (top.get("users") or [])[:5]],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(snap) + "\n")
    print(json.dumps(snap))
    return 0


# -- selfcheck -----------------------------------------------------------------

def _mirror(eng):
    """The left-right image of a position: the game's only symmetry."""
    m = Engine()
    for p in (0, 1):
        b = 0
        for cell in range(CELLS):
            if (eng.bb[p] >> cell) & 1:
                b |= 1 << CELL_MIRROR[cell]
        m.bb[p] = b
    m.fill = [eng.fill[COL_MIRROR[c]] for c in range(COLS)]
    m.turn = eng.turn
    m.steal_used = eng.steal_used
    m.last = CELL_MIRROR[eng.last] if eng.last >= 0 else -1
    m._over, m._winner = eng._over, eng._winner
    return m


def cmd_selfcheck(args):
    # the line table
    assert len(LINES) == 126, len(LINES)
    assert len({tuple(sorted(L)) for L in LINES}) == 126
    for L in LINES:
        assert len(set(L)) == 4 and all(0 <= c < CELLS for c in L)

    # opening: nine columns, no steal for the first player
    e = Engine()
    assert e.valid_actions() == set(range(COLS)), e.valid_actions()
    assert e.current_player == 0
    e.play(4)
    # the second player's first action may steal
    assert e.valid_actions() == set(range(COLS)) | {ACT_STEAL}
    assert e.turn == 1 and e.current_player == 1
    assert e.bb[0] == 1 << (6 * COLS + 4) and e.bb[1] == 0

    # STEAL repaints rather than places
    s = Engine()
    s.play(4)
    s.play(ACT_STEAL)
    assert s.bb[0] == 0, s.bb[0]
    assert s.bb[1] == 1 << (6 * COLS + 4)
    assert s.fill == [0, 0, 0, 0, 1, 0, 0, 0, 0]
    assert s.steal_used and s.turn == 2 and s.current_player == 0
    assert ACT_STEAL not in s.valid_actions()

    # gravity: three chips in one column stack upward
    g = Engine()
    for a in (0, 0, 0):
        g.play(a)
    assert g.fill[0] == 3
    for r in (6, 5, 4):
        assert (g.bb[0] | g.bb[1]) >> (r * COLS) & 1

    # vertical, horizontal and both diagonals, each hand-checked
    v = Engine()
    for a in (0, 1, 0, 1, 0, 1, 0):
        v.play(a)
    assert v.game_over and v.winner == 0 and v.turn == 7

    h = Engine()
    for a in (0, 0, 1, 1, 2, 2, 3):
        h.play(a)
    assert h.game_over and h.winner == 0

    # / diagonal for player 0: cells (6,0),(5,1),(4,2),(3,3)
    d = Engine()
    for a in (0, 1, 1, 2, 2, 3, 2, 3, 3, 8, 3):
        d.play(a)
    assert d.game_over and d.winner == 0, (d.game_over, d.winner)

    # the game stops the instant a line completes -- no move after
    try:
        v.play(2)
        raise AssertionError("played on a finished game")
    except ValueError:
        pass

    # illegal actions
    f = Engine()
    for _ in range(ROWS):
        f.play(3)
    assert 3 not in f.valid_actions()
    try:
        f.play(3)
        raise AssertionError("played a full column")
    except ValueError:
        pass
    try:
        Engine().play(ACT_STEAL)
        raise AssertionError("stole on the first turn")
    except ValueError:
        pass

    # random games: a draw is exactly a full board at the referee's turn count,
    # and a win is always a real 4-in-a-row
    rng = random.Random(11)
    draws = 0
    for _ in range(400):
        e = Engine()
        while not e.game_over:
            e.play(rng.choice(sorted(e.valid_actions())))
        if e.winner < 0:
            draws += 1
            assert all(x == ROWS for x in e.fill), e.fill
            assert e.turn == CELLS + (1 if e.steal_used else 0), e.turn
        else:
            assert any((e.bb[e.winner] & m) == m for m in LINE_MASKS)
            assert e.turn <= CELLS + 1
    assert draws >= 0

    # encoder shape and the hand-checked threat features
    t = Engine()
    for a in (0, 8, 1, 7, 2, 6):
        t.play(a)
    x = encode_planes(t)
    assert len(x) == N_IN
    assert x[126 + 6 * COLS + 3] == 1 / 3.0, x[126 + 6 * COLS + 3]
    assert x[189 + 6 * COLS + 5] == 1 / 3.0
    assert x[270 + 3] == 1.0 and sum(x[270:279]) == 1.0
    assert x[279 + 5] == 1.0 and sum(x[279:288]) == 1.0
    assert sum(x[288:297]) == 0.0
    assert x[252:261] == [1.0] * COLS
    assert x[297] == 0.0 and x[299] == 1.0 and x[300] == 0.0
    assert abs(x[298] - 6 / 63.0) < 1e-9
    # both threat squares sit on the bottom row, which is even from the bottom
    assert abs(x[301] - 1 / 8.0) < 1e-9 and x[302] == 0.0
    assert abs(x[303] - 1 / 8.0) < 1e-9 and x[304] == 0.0

    # the steal flag is a feature, and it is on exactly when the steal is legal
    st = Engine()
    st.play(4)
    assert encode_planes(st)[297] == 1.0
    assert encode_planes(st)[299] == 0.0

    # mirroring a position must equal permuting its features -- this is the
    # assertion that makes augmentation safe
    rng = random.Random(5)
    for _ in range(200):
        e = Engine()
        for _ in range(rng.randrange(0, 30)):
            if e.game_over:
                break
            e.play(rng.choice(sorted(e.valid_actions())))
        if e.game_over:
            continue
        a = encode_planes(e)
        b = encode_planes(_mirror(e))
        for cell in range(CELLS):
            for off in (0, 63, 126, 189):
                assert a[off + cell] == b[off + CELL_MIRROR[cell]], (off, cell)
        for c in range(COLS):
            for off in (252, 261, 270, 279, 288):
                assert a[off + c] == b[off + COL_MIRROR[c]], (off, c)
        assert a[297:] == b[297:]

    # best-response collection: only the LEARNER's plies may be recorded, and the
    # opponent must actually get to move. Storing an opponent ply would invert the
    # value target's sign for that sample.
    try:
        import torch
        dev = "cpu"
        B = 8
        tt = TensorC4(B, dev)
        seat = torch.arange(B, device=dev) % 2
        rec_sides, moved = [], 0
        for _ in range(12):
            lg = tt.legal_mask().float()
            a_l = torch.multinomial(lg + 1e-9, 1).squeeze(1)
            a_o = torch.multinomial(lg + 1e-9, 1).squeeze(1)
            mv, rec = best_response_step(tt, seat, a_l, a_o)
            # every recorded entry must be a position where the learner is to move
            assert bool((tt.side[rec] == seat[rec]).all()), "recorded an opponent ply"
            # and the action taken must be the learner's on its own turns
            assert bool((mv[rec] == a_l[rec]).all()), "learner's action not used"
            opp_turn = (tt.side != seat) & ~tt.done
            assert bool((mv[opp_turn] == a_o[opp_turn]).all()), "opponent did not move"
            assert not bool(rec[opp_turn].any()), "recorded on the opponent's turn"
            rec_sides.append(int(rec.sum()))
            moved += 1
            tt.step(mv)
        assert sum(rec_sides) > 0, "nothing was ever recorded"
        # over a full game each side moves about half the plies
        assert sum(rec_sides) <= moved * B, "recorded more plies than were played"
    except ImportError:
        print("selfcheck: torch absent, skipped the best-response contract")

    # the base85 alphabet: 85 distinct characters, none of them able to break a
    # C string literal or form a trigraph
    assert len(B85_ALPHABET) == 85 == len(set(B85_ALPHABET))
    assert not (set(B85_ALPHABET) & set('"\\?'))
    assert all(32 <= ord(c) < 127 for c in B85_ALPHABET)

    # base85 is exactly invertible on the sizes the packer uses
    rng = random.Random(3)
    for n in (1, 3, 4, 5, 1000, 70688):
        blob = bytes(rng.randrange(256) for _ in range(n))
        assert b85_decode(b85_encode(blob), n) == blob, n

    # quantisation keeps the sign and the ordering of the biggest weights
    try:
        import numpy as np
        w = np.random.RandomState(0).randn(64, 32).astype(np.float32)
        q, s = quantize_int8(w)
        assert np.abs(q.astype(np.float32) * s - w).max() < s
    except ImportError:
        print("selfcheck: numpy absent, skipped the quantisation assert")

    # a checkpoint round trip, which is what --init and pack both depend on
    try:
        import tempfile
        import torch
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ck.pt")
            net = build_aznet("cpu")
            want = [N_IN, AZ_TRUNK1, AZ_TRUNK2, N_ACT]
            torch.save({"state_dict": net.state_dict(), "shape": want}, path)
            back, shape = load_aznet(path, "cpu")
            assert list(shape) == want, (shape, want)
            for k, v in net.state_dict().items():
                assert torch.equal(v, back.state_dict()[k]), k
            torch.save({"state_dict": net.state_dict()}, path)
            try:
                load_aznet(path, "cpu")
                raise AssertionError("loaded a checkpoint with no shape")
            except ValueError:
                pass
    except ImportError:
        print("selfcheck: torch absent, skipped the checkpoint round trip")

    # the budget arithmetic the C++ template depends on
    assert N_IN == 305 and N_ACT == 10
    assert aznet_param_bytes() == (305 * AZ_TRUNK1 + AZ_TRUNK1 * AZ_TRUNK2
                                   + AZ_TRUNK2 * 10 + AZ_TRUNK2)

    print(f"selfcheck OK ({draws} of 400 random games were draws; "
          f"{aznet_param_bytes()} weights at trunk "
          f"{AZ_TRUNK1}-{AZ_TRUNK2} = "
          f"{aznet_param_bytes() * 1.25 / 1000:.1f}k base85 chars)")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selfcheck")
    sub.add_parser("bench")
    f = sub.add_parser("fuzz")
    f.add_argument("--other", required=True)
    f.add_argument("--games", type=int, default=2000)
    f.add_argument("--seed", type=int, default=1)
    m = sub.add_parser("match")
    m.add_argument("--a", required=True, choices=POLICIES)
    m.add_argument("--b", required=True, choices=POLICIES)
    m.add_argument("--games", type=int, default=100)
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--budget-ms", type=int, default=90)
    c = sub.add_parser("cg")
    c.add_argument("--policy", required=True, choices=POLICIES)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--budget-ms", type=int, default=90)
    en = sub.add_parser("eval-net")
    en.add_argument("--net", default="out/gpunet.pt")
    en.add_argument("--vs-net", default=None,
                    help="a past-self checkpoint; overrides --vs")
    en.add_argument("--vs", default="greedy", choices=("greedy", "random"))
    en.add_argument("--games", type=int, default=1024)
    en.add_argument("--open-plies", type=int, default=8)
    en.add_argument("--depth", type=int, default=2, choices=(1, 2))
    en.add_argument("--device", default="auto")
    en.add_argument("--seed", type=int, default=0)
    gp = sub.add_parser("gpu-parity")
    gp.add_argument("--games", type=int, default=64)
    gp.add_argument("--seed", type=int, default=0)
    gp.add_argument("--device", default="auto")
    gp.add_argument("--check-encode", action="store_true")
    tg = sub.add_parser("train-gpu")
    tg.add_argument("--iters", type=int, default=200)
    tg.add_argument("--batch-games", type=int, default=1024)
    tg.add_argument("--steps-per-iter", type=int, default=12)
    tg.add_argument("--updates-per-iter", type=int, default=48)
    tg.add_argument("--batch", type=int, default=2048)
    tg.add_argument("--buffer", type=int, default=800000)
    tg.add_argument("--lr", type=float, default=1e-3)
    tg.add_argument("--tau", type=float, default=0.5)
    tg.add_argument("--depth", type=int, default=2, choices=(1, 2),
                    help="plies of exact search behind the policy target")
    tg.add_argument("--eps", type=float, default=0.08)
    tg.add_argument("--opening-plies", type=int, default=10)
    tg.add_argument("--value-weight", type=float, default=1.0)
    tg.add_argument("--augment", action="store_true", default=True)
    tg.add_argument("--eval-every", type=int, default=10)
    tg.add_argument("--eval-games", type=int, default=256)
    tg.add_argument("--device", default="auto")
    tg.add_argument("--seed", type=int, default=0)
    tg.add_argument("--opponent", default=None,
                    help="freeze this checkpoint as the opponent "
                         "(best-response / nemesis training)")
    tg.add_argument("--init", default=None,
                    help="warm start from a checkpoint instead of random init")
    tg.add_argument("--snapshot-every", type=int, default=250)
    tg.add_argument("--out", default="out/gpunet.pt")
    pk = sub.add_parser("pack")
    pk.add_argument("--net", default="out/gpunet.pt")
    pk.add_argument("--out", default="out/c4_bot.cpp")
    cp = sub.add_parser("check-pack")
    cp.add_argument("--net", default="out/gpunet.pt")
    cp.add_argument("--cpp", default="out/c4_bot.cpp")
    cp.add_argument("--games", type=int, default=6)
    cp.add_argument("--seed", type=int, default=5)
    cp.add_argument("--tol", type=float, default=1e-4)
    cp.add_argument("--min-agree", type=float, default=100.0)
    cb = sub.add_parser("check-bot")
    cb.add_argument("--cpp", default="out/c4_bot.cpp")
    cb.add_argument("--games", type=int, default=8)
    cb.add_argument("--seed", type=int, default=5)
    bn = sub.add_parser("bench-net")
    bn.add_argument("--cpp", default="out/c4_bot.cpp")
    bn.add_argument("--vs", default="greedy", choices=POLICIES)
    bn.add_argument("--games", type=int, default=100)
    bn.add_argument("--budget-ms", type=int, default=40)
    bn.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("snapshot")
    s.add_argument("--arena", default="connect-4")
    s.add_argument("--pseudo", default="Napkin100k")
    s.add_argument("--out", default="out/ladder_snapshots.jsonl")

    args = ap.parse_args()
    table = {
        "selfcheck": cmd_selfcheck, "bench": cmd_bench, "fuzz": cmd_fuzz,
        "match": cmd_match, "cg": cmd_cg, "gpu-parity": cmd_gpu_parity,
        "train-gpu": cmd_train_gpu, "pack": cmd_pack,
        "check-pack": cmd_check_pack, "check-bot": cmd_check_bot,
        "bench-net": cmd_bench_net, "snapshot": cmd_snapshot,
        "eval-net": cmd_eval_net,
    }
    sys.exit(table[args.cmd](args))


if __name__ == "__main__":
    main()
