"""
Detailed Policy Conformance Analysis: Greedy FIFO vs SPT on BPI 2012

State:  (fastest_idle_resource_idx, spt_activity_idx, fifo_activity_idx)
Action: activity_name|Rn  (preserving resource identity)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import re
import copy
import numpy as np

sys.path.append("C:/Users/20183272/OneDrive - TU Eindhoven/Documents/GitHub/gympn")

from gympn.solvers import HeuristicSolver
from gympn_problem_from_json_with_ids import (
    load_parameters, create_task_assignment_problem, derive_problem_metadata
)
from policy_conformance import PolicyConformance
from BPI_2012_policy_conformance_with_ids import (
    make_spt_heuristic, make_fifo_heuristic, dprint
)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def _safe(name):
    return name.replace(" ", "_").replace(".", "_")


# ─────────────────────────────────────────────────────────────────────────────
# State variables
# ─────────────────────────────────────────────────────────────────────────────

def create_detailed_state_variables(parameters, valid_resources, time_scale,
                                    filtered_activities):
    """
    Build three state variable functions:
      1. fastest_resource  — idx of the idle resource with the lowest average
         processing time, or -1 when no resource is idle.
      2. spt_activity      — activity idx that normalised-SPT would select, or
         -1 when nothing can be assigned.
      3. fifo_activity     — activity idx of the longest-waiting case (smallest
         counter), or -1 when no case is waiting.
    """
    activities = parameters['activities']
    arm = parameters['activity_resource_mapping']
    activity_to_idx = {a: i for i, a in enumerate(activities)}
    idx_to_activity = {i: a for i, a in enumerate(activities)}
    vr_set = set(valid_resources)

    resource_avg = {}
    for idx, res_name in enumerate(valid_resources):
        durs = []
        for act, rmap in arm.items():
            if res_name in rmap:
                raw = rmap[res_name]
                if raw is not None and not (isinstance(raw, float) and raw != raw):
                    durs.append(max(0.01, abs(raw) / time_scale))
        resource_avg[idx] = sum(durs) / len(durs) if durs else float('inf')

    activity_avg = {}
    for act_name, rmap in arm.items():
        vals = [max(0.01, abs(d) / time_scale) for r, d in rmap.items()
                if r in vr_set and d is not None
                and not (isinstance(d, float) and d != d)]
        activity_avg[act_name] = sum(vals) / len(vals) if vals else 1.0

    # ── Var 1: fastest idle resource ─────────────────────────────────────
    _ravg = resource_avg

    def fastest_resource(resource_pool_queue):
        if not resource_pool_queue:
            return -1
        best = min(resource_pool_queue,
                   key=lambda t: _ravg.get(int(t['resource_idx']), float('inf')))
        return int(best['resource_idx'])

    # ── Vars 2 & 3 need all waiting-queue tokens + resource pool + clock ─
    safe_names = [_safe(a) for a in filtered_activities]
    waiting_params = [f"waiting_{s}_queue_tokens" for s in safe_names]
    param_str = ", ".join(["resource_pool_queue", "clock"] + waiting_params)

    # SPT activity (global minimum normalised duration across all bindings)
    spt_lines = [
        f"def spt_activity({param_str}):",
        "    import math",
        "    best_metric = math.inf",
        "    best_act_idx = -1",
        "    if not resource_pool_queue:",
        "        return -1",
    ]
    for act, s in zip(filtered_activities, safe_names):
        ai = activity_to_idx[act]
        spt_lines += [
            f"    if any(t.time <= clock for t in waiting_{s}_queue_tokens):",
            f"        _rmap = _arm.get(_i2a.get({ai}), {{}})",
            f"        for _rt in resource_pool_queue:",
            f"            _ri = int(_rt['resource_idx'])",
            f"            if _ri >= len(_vr): continue",
            f"            _rn = _vr[_ri]",
            f"            if _rn not in _rmap: continue",
            f"            _rd = _rmap[_rn]",
            f"            if _rd is None or (isinstance(_rd, float) and _rd != _rd):",
            f"                _d = 1.0",
            f"            else:",
            f"                _d = max(0.01, abs(_rd) / _ts)",
            f"            _avg = _aavg.get(_i2a.get({ai}), 1.0) or 1.0",
            f"            _m = _d / _avg",
            f"            if _m < best_metric:",
            f"                best_metric = _m",
            f"                best_act_idx = {ai}",
        ]
    spt_lines.append("    return best_act_idx")

    # FIFO activity (case with smallest counter across all waiting queues)
    fifo_lines = [
        f"def fifo_activity({param_str}):",
        "    best_counter = float('inf')",
        "    best_act_idx = -1",
    ]
    for act, s in zip(filtered_activities, safe_names):
        ai = activity_to_idx[act]
        fifo_lines += [
            f"    for _t in waiting_{s}_queue_tokens:",
            f"        if _t.time <= clock:",
            f"            _c = _t.value.get('counter', float('inf'))",
            f"            if _c < best_counter:",
            f"                best_counter = _c",
            f"                best_act_idx = {ai}",
        ]
    fifo_lines.append("    return best_act_idx")

    ns = {
        '_i2a': idx_to_activity,
        '_arm': arm,
        '_vr': valid_resources,
        '_ts': time_scale,
        '_aavg': activity_avg,
    }
    exec("\n".join(spt_lines), ns)
    exec("\n".join(fifo_lines), ns)

    return [fastest_resource, ns['spt_activity'], ns['fifo_activity']]


# ─────────────────────────────────────────────────────────────────────────────
# Action key helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_action_key(key):
    """Convert raw binding key to 'activity_name|Rn' or 'postpone'."""
    if key == "postpone" or key.startswith("None"):
        return "postpone"
    parts = key.split("|", 1)
    act = parts[0].replace("assign_", "")
    if len(parts) > 1:
        m = re.search(r"resource_idx['\"]?\s*:\s*(\d+\.?\d*)", parts[1])
        if m:
            return f"{act}|R{int(float(m.group(1)))}"
    return act


def remap_probs(pa):
    """Collapse raw binding keys to activity|resource labels."""
    out = {}
    for obs, probs in pa.items():
        out[obs] = {}
        for a, p in probs.items():
            short = parse_action_key(a)
            out[obs][short] = out[obs].get(short, 0.0) + p
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    dprint("=" * 80)
    dprint("DETAILED CONFORMANCE: Greedy FIFO vs SPT on BPI 2012")
    dprint("=" * 80)

    # ── Configuration ────────────────────────────────────────────────────
    simulation_length = 10
    num_episodes = 4
    min_resource_activities = 2
    max_resources = 50
    max_activities = None
    remove_self_loops = True
    arrival_rate_factor = 0.1

    dprint(f"[CONFIG] simulation_length={simulation_length}, episodes={num_episodes}")
    dprint(f"[CONFIG] min_resource_activities={min_resource_activities}, "
           f"max_resources={max_resources}, arrival_rate_factor={arrival_rate_factor}")

    # ── Load problem ─────────────────────────────────────────────────────
    dprint("\n[1] Loading problem...")
    parameters = load_parameters("simulation_parameters.json")
    problem = create_task_assignment_problem(
        parameters,
        min_resource_activities=min_resource_activities,
        max_resources=max_resources,
        max_activities=max_activities,
        remove_self_loops=remove_self_loops,
        arrival_rate_factor=arrival_rate_factor,
    )

    meta = derive_problem_metadata(
        parameters,
        min_resource_activities=min_resource_activities,
        max_resources=max_resources,
        max_activities=max_activities,
    )
    dprint(f"[SUCCESS] {len(meta['valid_resources'])} resources, "
           f"time_scale={meta['time_scale']:.0f}s")

    # ── Create solvers ───────────────────────────────────────────────────
    dprint("\n[2] Creating solvers...")
    spt_solver = HeuristicSolver(heuristic_function=make_spt_heuristic(
        parameters,
        valid_resources=meta['valid_resources'],
        time_scale=meta['time_scale'],
        normalize=True,
    ))
    fifo_solver = HeuristicSolver(heuristic_function=make_fifo_heuristic(
        parameters,
        valid_resources=meta['valid_resources'],
        time_scale=meta['time_scale'],
        tie_break='spt',
    ))
    dprint("[SUCCESS] Greedy FIFO (tie-break=spt) and normalised SPT created")

    # ── State variables ──────────────────────────────────────────────────
    dprint("\n[3] Creating state variables...")
    arm = parameters.get('activity_resource_mapping', {})
    filtered_acts = [a for a in parameters['activities'] if a in arm]

    state_variables = create_detailed_state_variables(
        parameters,
        valid_resources=meta['valid_resources'],
        time_scale=meta['time_scale'],
        filtered_activities=filtered_acts,
    )
    dprint(f"[SUCCESS] {len(state_variables)} state variables: "
           "(fastest_resource, spt_activity, fifo_activity)")

    # ── Conformance episodes ─────────────────────────────────────────────
    dprint("\n[4] Running conformance analysis...")
    excluded_attrs = ["case_id", "counter"]
    conformance = PolicyConformance(
        gym_problem=problem,
        heuristic_solver=fifo_solver,
        gym_solver=spt_solver,
        state_variables=state_variables,
        excluded_token_attrs=excluded_attrs,
    )

    for ep in range(num_episodes):
        dprint(f"  Episode {ep + 1}/{num_episodes}...")
        conformance.compute_state_action_mapping(horizon=simulation_length)

    # ── Compute probabilities & metrics ──────────────────────────────────
    dprint("\n[5] Computing metrics...")
    fifo_pa_raw = conformance.compute_action_probability(
        conformance.p1_state_action_mapping)
    spt_pa_raw = conformance.compute_action_probability(
        conformance.p2_state_action_mapping)

    fifo_pa = remap_probs(fifo_pa_raw)
    spt_pa = remap_probs(spt_pa_raw)

    emac = conformance.compute_state_action_emd(fifo_pa, spt_pa)
    max_vr = conformance.compute_max_visit_ratio()

    fifo_visits = conformance.p1_state_visit_frequency
    spt_visits = conformance.p2_state_visit_frequency

    all_actions = sorted({a for obs_p in list(fifo_pa.values()) + list(spt_pa.values())
                          for a in obs_p})

    dprint(f"  Unique observations visited: {len(set(fifo_visits) | set(spt_visits))}")
    dprint(f"  Unique action labels: {len(all_actions)}")

    # ── Expected reward rollouts (PERG) ──────────────────────────────────
    dprint("\n[6] Computing expected reward (PERG)...")
    exp_ret_fifo, exp_ret_spt, delta_exp_ret = conformance.expected_reward_run(
        fifo_solver, spt_solver,
        rho=0.99, eta=0, num_rollouts=5, num_steps=5,
    )
    dprint(f"  Observations evaluated for PERG: {len(delta_exp_ret)}")

    # ── Policy tables ────────────────────────────────────────────────────
    dprint("\n[7] Saving policy tables...")
    import pandas as pd

    def build_table(visit_freq, action_prob, expected_return, actions):
        all_obs = sorted(set(visit_freq) | set(action_prob))
        rows = []
        for obs in all_obs:
            row = {"Observation": str(obs), "Visit Freq": visit_freq.get(obs, 0)}
            probs = action_prob.get(obs, {})
            for a in actions:
                row[f"P({a})"] = round(probs.get(a, 0.0), 4)
            er = expected_return.get(obs, float('nan'))
            row["Exp. Return"] = round(er, 4) if er == er else float('nan')
            rows.append(row)
        return pd.DataFrame(rows)

    df_fifo = build_table(fifo_visits, fifo_pa, exp_ret_fifo, all_actions)
    df_spt = build_table(spt_visits, spt_pa, exp_ret_spt, all_actions)

    for label, df, suffix in [("FIFO", df_fifo, "fifo"), ("SPT", df_spt, "spt")]:
        name = f"policy_table_{suffix}_detailed_2"
        df.to_csv(f"{name}.csv", index=False)
        dprint(f"  Saved: {name}.csv ({len(df)} observations)")

    # ── Conformance results table ────────────────────────────────────────
    dprint("\n[8] Saving conformance results...")
    all_obs = sorted(max_vr.keys())
    rows = []
    for obs in all_obs:
        e = emac.get(obs, float('nan'))
        p = delta_exp_ret.get(obs, float('nan'))
        rows.append({
            "Observation": str(obs),
            "MaxVisitRatio": round(max_vr[obs], 4),
            "EMAC": round(e, 4) if e == e else float('nan'),
            "PERG": round(p, 4) if p == p else float('nan'),
        })
    df_results = pd.DataFrame(rows)
    df_results.to_csv("conformance_results.csv", index=False)
    dprint(f"  Saved: conformance_results.csv ({len(df_results)} observations)")

    # ── EMAC histogram ───────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    emac_vals = pd.to_numeric(df_results["EMAC"], errors="coerce").dropna()
    if len(emac_vals) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(emac_vals, bins=20, color="#4C72B0", edgecolor="white", alpha=0.9)
        ax.set_xlabel("EMAC", fontsize=12)
        ax.set_ylabel("Number of observations", fontsize=12)
        ax.set_title("EMAC: FIFO vs SPT",
                      fontsize=13, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig("emac_histogram_detailed.pdf", bbox_inches="tight")
        dprint("  Saved: emac_histogram_detailed.pdf")
        plt.close(fig)

    # ── PERG histogram ───────────────────────────────────────────────────
    perg_vals = pd.to_numeric(df_results["PERG"], errors="coerce").dropna()
    if len(perg_vals) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(perg_vals, bins=20, color="#DD8452", edgecolor="white", alpha=0.9)
        ax.set_xlabel("PERG", fontsize=12)
        ax.set_ylabel("Number of observations", fontsize=12)
        ax.set_title("PERG: FIFO vs SPT",
                      fontsize=13, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig("perg_histogram_detailed.pdf", bbox_inches="tight")
        dprint("  Saved: perg_histogram_detailed.pdf")
        plt.close(fig)

    # ── Summary ──────────────────────────────────────────────────────────
    dprint("\n" + "=" * 80)
    dprint("SUMMARY")
    dprint("=" * 80)
    dprint(f"  Observations visited:     {len(all_obs)}")
    dprint(f"  Observations with EMAC:   {len(emac_vals)}")
    if len(emac_vals) > 0:
        dprint(f"  Mean EMAC:                {emac_vals.mean():.4f}")
        dprint(f"  Median EMAC:              {emac_vals.median():.4f}")
        dprint(f"  EMAC = 1.0 (full disagr): {(emac_vals == 1.0).sum()}")
        dprint(f"  EMAC = 0.0 (full agree):  {(emac_vals == 0.0).sum()}")
    dprint(f"  Observations with PERG:   {len(perg_vals)}")
    if len(perg_vals) > 0:
        dprint(f"  Mean PERG:                {perg_vals.mean():.4f}")
        dprint(f"  Median PERG:              {perg_vals.median():.4f}")
        dprint(f"  Max |PERG|:               {perg_vals.abs().max():.4f}")
    dprint("=" * 80 + "\n")
