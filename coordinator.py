#!/usr/bin/env python3
"""
Erdos #396 multi-host coordinator.

Distributes the search for one k across machines as bounded-range jobs run by
the solver's worker mode (./binary <k> <start_L> <end_L>), collects per-range
minima, and determines each answer with exact minimum semantics:

  - ranges are [start + j*G, start + (j+1)*G), G a multiple of 2^26, so every
    range owns the disjoint candidate territory [range_start+K, range_end+K)
  - the answer for k is the result of the LOWEST-indexed range that reported a
    hit, once every range below it has completed with no hit
  - ranges above the first hit that were already in flight are wasted work,
    bounded by (#slots - 1) jobs

Operation:
  - config: coord-config.json (see coord-config.example.json)
  - state:  coord-state.json, rewritten atomically after every event; safe to
    kill and restart (in-flight jobs are simply re-run)
  - output: appends answers to results-396-dist.txt

Hosts are reached with plain `ssh <host>` (use ~/.ssh/config for users/keys/
ports) or run locally with "ssh": "local". Each host entry needs the solver
binary already built in <workdir> on that machine.
"""

import json
import os
import subprocess
import sys
import threading
import time

CHUNK_MAX = 1 << 26  # ranges must be multiples of this (see worker-mode notes)
CONFIG = sys.argv[1] if len(sys.argv) > 1 else "coord-config.json"
STATE = "coord-state.json"
RESULTS = "results-396-dist.txt"

lock = threading.Condition()


def log(msg):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg, flush=True)


def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def save_state(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE)


class Scheduler:
    """Job bookkeeping for the current k. Call holding `lock`."""

    def __init__(self, state, G, reset_start=False):
        self.st = state  # {"k": int, "start_L": int, "done": {str(j): n}, "answers": [...]}
        self.G = G
        self.reset_start = reset_start
        self.t0 = time.time()          # start of the current k (resumes count as restarts)
        self.cands_done = len(self.st.get("done", {})) * G
        self.inflight = set()
        self.retry = []

    def done(self):
        return {int(j): n for j, n in self.st["done"].items()}

    def hit_j(self):
        hits = [j for j, n in self.done().items() if n > 0]
        return min(hits) if hits else None

    def next_job(self):
        """Return job index j to run, or None if nothing useful right now."""
        if self.retry:
            return self.retry.pop(0)
        d = self.done()
        h = self.hit_j()
        if h is not None:
            for j in range(h):
                if j not in d and j not in self.inflight:
                    return j
            return None  # only waiting on in-flight ranges below the hit
        j = 0
        while j in d or j in self.inflight:
            j += 1
        return j

    def job_range(self, j):
        s = self.st["start_L"] + j * self.G
        return s, s + self.G

    def report(self, j, n):
        self.st["done"][str(j)] = n
        self.inflight.discard(j)
        d = self.done()
        h = self.hit_j()
        if h is not None and all(jj in d and d[jj] == 0 for jj in range(h)):
            return d[h]  # answer for this k
        return None

    def advance(self, answer):
        k = self.st["k"]
        self.st["answers"].append([k, answer])
        self.st["k"] = k + 1
        # #396: answers are monotone in k, so the next search starts at the
        # previous answer. #389 (binary erdos_389_v10, "k" = the problem's n,
        # RESULT reports N; min k = N - (n-1)): answers are NOT monotone
        # (e.g. n=7 < n=6), so set "reset_start": true to rescan from 1.
        self.st["start_L"] = 1 if self.reset_start else answer
        self.st["done"] = {}
        self.inflight = set()
        self.retry = []
        self.t0 = time.time()
        self.cands_done = 0


def run_job(host, binary, workdir, k, s, e, timeout, env=None):
    # per-host env (e.g. {"NT": "7"} to leave a core free for a GPU worker
    # sharing the box) is injected into the remote/local command line
    env_str = "".join(f"{key}={val} " for key, val in (env or {}).items())
    cmd_str = f"cd {workdir} && {env_str}./{binary} {k} {s} {e}"
    if host == "local":
        cmd = ["bash", "-c", cmd_str]
    else:
        cmd = ["ssh", "-o", "BatchMode=yes", host, cmd_str]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "RESULT" and int(parts[1]) == k:
            return int(parts[2])
    raise RuntimeError(f"no RESULT line (rc={out.returncode}, stderr={out.stderr[-200:]!r})")


def worker_thread(hostcfg, slot, sched, cfg, stop):
    name = f"{hostcfg['name']}#{slot}"
    fails = 0
    while not stop.is_set():
        with lock:
            if sched.st["k"] > cfg.get("kmax", 99):
                return
            j = sched.next_job()
            if j is None:
                lock.wait(2.0)
                continue
            sched.inflight.add(j)
            k = sched.st["k"]
            s, e = sched.job_range(j)
        log(f"{name}: k={k} job {j} [{s}, {e})")
        t_job = time.time()
        try:
            n = run_job(hostcfg.get("ssh", "local"), hostcfg["binary"],
                        hostcfg["workdir"], k, s, e, cfg.get("job_timeout_s", 7200),
                        hostcfg.get("env"))
            fails = 0
        except Exception as ex:
            fails += 1
            log(f"{name}: job {j} FAILED ({ex}); requeueing" +
                (", cooling down 60s" if fails >= 3 else ""))
            with lock:
                sched.inflight.discard(j)
                sched.retry.append(j)
                lock.notify_all()
            if fails >= 3:
                time.sleep(60)
                fails = 0
            continue
        dt = max(time.time() - t_job, 1e-9)
        # a hit job early-exits at the answer, so count only what it scanned
        scanned = (min(n, e) - s) if n else (e - s)
        spd = scanned / dt
        with lock:
            if sched.st["k"] != k:
                continue  # stale result from a previous k (shouldn't happen)
            sched.cands_done += scanned
            agg = sched.cands_done / max(time.time() - sched.t0, 1e-9)
            ans = sched.report(j, n)
            log(f"{name}: k={k} job {j} -> {n if n else 'no hit'} | "
                f"{spd / 1e9:.2f} G/s | all-hosts {agg / 1e9:.2f} G/s")
            if ans is not None:
                log(f"*** k = {k} | min n = {ans} ***")
                with open(RESULTS, "a") as f:
                    f.write(f"k = {k:2d} | min n = {ans}\n")
                sched.advance(ans)
            save_state(sched.st)
            lock.notify_all()


def main():
    cfg = load_json(CONFIG)
    if not cfg:
        sys.exit(f"config {CONFIG} not found - copy coord-config.example.json")
    G = int(cfg.get("range_chunks", 16384)) * CHUNK_MAX  # default 2^40 ~ 1.1e12 cands
    st = load_json(STATE)
    if not st:
        st = {"k": int(cfg["seed_k"]), "start_L": int(cfg["seed_L"]),
              "done": {}, "answers": []}
        save_state(st)
    log(f"coordinator: k={st['k']} start_L={st['start_L']} G={G} "
        f"({G >> 20} M candidates/job)")
    sched = Scheduler(st, G, reset_start=cfg.get("reset_start", False))
    stop = threading.Event()
    threads = []
    for h in cfg["hosts"]:
        for slot in range(int(h.get("slots", 1))):
            t = threading.Thread(target=worker_thread, args=(h, slot, sched, cfg, stop),
                                 daemon=True)
            t.start()
            threads.append(t)
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
        log("all workers finished (kmax reached)")
    except KeyboardInterrupt:
        stop.set()
        log("interrupted - state saved; restart to resume")


if __name__ == "__main__":
    main()