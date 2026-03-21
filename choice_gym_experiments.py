import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

sys.path.append("C:/Users/20183272/OneDrive - TU Eindhoven/Documents/GitHub/gympn")

import torch
from gympn.networks import HeteroActor
from torch_geometric.nn.conv.han_conv import HANConv
from torch_geometric.nn.aggr.basic import SumAggregation
torch.serialization.add_safe_globals([HeteroActor, HANConv, SumAggregation])

from simpn.simulator import SimToken
from gympn.simulator import GymProblem
from gympn.solvers import HeuristicSolver, GymSolver
from policy_conformance import PolicyConformance


# ─────────────────────────────────────────────────────────────────────────────
# Model setup 
# ─────────────────────────────────────────────────────────────────────────────

class HeuristicSolver(HeuristicSolver):
    @staticmethod
    def get_place_tokens_by_time(place_id, pn):
        return [t.value for p in pn.places if p._id == place_id
                for t in p.marking if t.time <= pn.clock]


def build_assembly_system():
    asm = GymProblem(allow_postpone=True, causal_rl=False)
    asm.rng = np.random.default_rng(42)

    arrival_chip       = asm.add_var("chip supply",       var_attributes=["chip_id"])
    arrival_phone_case = asm.add_var("phone case supply", var_attributes=["phone_case_id"])
    stock_chip         = asm.add_var("stock_chip",        var_attributes=["chip_id"])
    stock_phone_case   = asm.add_var("stock_phone_case",  var_attributes=["phone_case_id"])
    phone_resource     = asm.add_var("phone_resource",    var_attributes=["phone_id"])
    game_resource      = asm.add_var("game_resource",     var_attributes=["game_id"])

    asm.set_unobservable(token_attrs={
        'stock_chip': ['chip_id'], 'stock_phone_case': ['phone_case_id'],
        'phone_resource': ['phone_id'], 'game_resource': ['game_id'],
    })

    def chip_arrival(arrival):
        chip_id = arrival["chip_id"] + 1
        delay = np.random.default_rng(42 + chip_id).exponential(scale=3.0)
        return [SimToken({"chip_id": chip_id}, delay=delay),
                SimToken({"chip_id": chip_id}, delay=delay)]

    def phone_case_arrival(arrival):
        phone_case_id = arrival["phone_case_id"] + 1
        delay = np.random.default_rng(42 + phone_case_id).exponential(scale=5.0)
        return [SimToken({"phone_case_id": phone_case_id}, delay=delay),
                SimToken({"phone_case_id": phone_case_id}, delay=delay)]

    asm.add_event([arrival_chip], [arrival_chip, stock_chip],
                  behavior=chip_arrival, name="chip_arrival")
    asm.add_event([arrival_phone_case], [arrival_phone_case, stock_phone_case],
                  behavior=phone_case_arrival, name="phone_case_arrival")
    asm.add_action([stock_chip, game_resource], [game_resource],
                   behavior=lambda stock_chip, game_resource: [SimToken(game_resource, delay=1)],
                   reward_function=lambda stock_chip, game_resource: 1,
                   name="game_production")
    asm.add_action([stock_chip, stock_phone_case, phone_resource], [phone_resource],
                   behavior=lambda stock_chip, stock_phone_case, phone_resource:
                       [SimToken(phone_resource, delay=2)],
                   reward_function=lambda stock_chip, stock_phone_case, phone_resource: 3,
                   name="phone_production")

    arrival_chip.put({"chip_id": 0})
    arrival_phone_case.put({"phone_case_id": 0})
    game_resource.put({"game_id": 1})
    phone_resource.put({"phone_id": 1})
    return asm


def choice_pattern(pn, actions_dict):
    chips  = HeuristicSolver.get_place_tokens_by_time('stock_chip', pn)
    cases  = HeuristicSolver.get_place_tokens_by_time('stock_phone_case', pn)
    game_r = HeuristicSolver.get_place_tokens_by_time('game_resource', pn)
    phone_r = HeuristicSolver.get_place_tokens_by_time('phone_resource', pn)
    if cases and len(chips) > 1 and phone_r:
        return {'phone_production': actions_dict['phone_production'][0]}
    elif not cases and len(chips) >= 5 and game_r:
        return {'game_production': actions_dict['game_production'][0]}
    return 'postpone'


def choice_pattern_pi3(pn, actions_dict):
    chips  = HeuristicSolver.get_place_tokens_by_time('stock_chip', pn)
    cases  = HeuristicSolver.get_place_tokens_by_time('stock_phone_case', pn)
    game_r = HeuristicSolver.get_place_tokens_by_time('game_resource', pn)
    phone_r = HeuristicSolver.get_place_tokens_by_time('phone_resource', pn)
    if cases and len(chips) > 0 and phone_r:
        return {'phone_production': actions_dict['phone_production'][0]}
    elif not cases and len(chips) >= 2 and game_r:
        return {'game_production': actions_dict['game_production'][0]}
    return 'postpone'


W_P        = "data/train/best_policy.pth"
N_EPISODES = 10
HORIZON    = 100
EXCL       = ["chip_id", "phone_case_id", "game_id", "phone_id"]
STATE_VARS = [
    lambda stock_chip_queue_tokens, clock:
        len([t for t in stock_chip_queue_tokens if t.time <= clock]),
    lambda stock_phone_case_queue_tokens, clock:
        len([t for t in stock_phone_case_queue_tokens if t.time <= clock]),
]

# ─────────────────────────────────────────────────────────────────────────────
# Run conformance analysis for (π₁ vs π₂) and (π₁ vs π₃)
# ─────────────────────────────────────────────────────────────────────────────

def run_conformance(frozen_pn, solver_p1, solver_p2):
    asm = copy.deepcopy(frozen_pn)
    conf = PolicyConformance(asm, solver_p1, solver_p2, STATE_VARS,
                             excluded_token_attrs=EXCL)
    for ep in range(N_EPISODES):
        print(f"  episode {ep + 1}/{N_EPISODES}...")
        conf.compute_state_action_mapping(horizon=HORIZON)
    pa1 = conf.compute_action_probability(conf.p1_state_action_mapping)
    pa2 = conf.compute_action_probability(conf.p2_state_action_mapping)
    return pa1, pa2, conf


def short_action(key):
    base = key.split("|")[0]
    return "postpone" if base == "None" else base


def remap_probs(pa):
    out = {}
    for obs, probs in pa.items():
        out[obs] = {}
        for a, p in probs.items():
            s = short_action(a)
            out[obs][s] = out[obs].get(s, 0.0) + p
    return out


frozen = build_assembly_system()
pi1_solver = HeuristicSolver(heuristic_function=choice_pattern)
pi3_solver = HeuristicSolver(heuristic_function=choice_pattern_pi3)
pi2_solver = GymSolver(weights_path=W_P,
                        metadata=copy.deepcopy(frozen).make_metadata())

print("Running π₁ vs π₂ conformance...")
raw_pi1_12, raw_pi2, conf_12 = run_conformance(frozen, pi1_solver, pi2_solver)

print("Running π₁ vs π₃ conformance...")
raw_pi1_13, raw_pi3, conf_13 = run_conformance(frozen, pi1_solver, pi3_solver)

print("Running π₂ vs π₃ conformance...")
raw_pi2_23, raw_pi3_23, conf_23 = run_conformance(frozen, pi2_solver, pi3_solver)

def fill_missing(primary, fallback):
    """Use primary estimates; fill observations missing from primary using fallback."""
    out = dict(primary)
    for obs, probs in fallback.items():
        if obs not in out:
            out[obs] = probs
    return out

# π₁ appears in two runs (π₁ vs π₂  and  π₁ vs π₃)
# π₂ appears in two runs (π₁ vs π₂  and  π₂ vs π₃)
# π₃ appears in two runs (π₁ vs π₃  and  π₂ vs π₃)
# remap_probs is applied before combining so probabilities always sum to 1 per observation
pi1_pa = fill_missing(remap_probs(raw_pi1_12), remap_probs(raw_pi1_13))
pi2_pa = fill_missing(remap_probs(raw_pi2),    remap_probs(raw_pi2_23))
pi3_pa = fill_missing(remap_probs(raw_pi3),    remap_probs(raw_pi3_23))

# ─────────────────────────────────────────────────────────────────────────────
# Build heatmap matrices
# ─────────────────────────────────────────────────────────────────────────────

ACTIONS       = ["phone_production", "game_production", "postpone"]
ACTION_LABELS = ["Prod A", "Prod B", "Postpone"]

all_obs = sorted(set(pi1_pa) | set(pi2_pa) | set(pi3_pa))

def fmt_obs(obs):
    return f"({obs[0]}, {obs[1]})"

obs_labels = [fmt_obs(o) for o in all_obs]

def make_matrix(pa):
    mat = np.zeros((len(all_obs), len(ACTIONS)))
    for i, obs in enumerate(all_obs):
        probs = pa.get(obs, {})
        for j, act in enumerate(ACTIONS):
            mat[i, j] = probs.get(act, 0.0)
    return mat

mat_pi1 = make_matrix(pi1_pa)
mat_pi2 = make_matrix(pi2_pa)
mat_pi3 = make_matrix(pi3_pa)

# ─────────────────────────────────────────────────────────────────────────────
# Heatmap — 3 side-by-side subplots, one per policy
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(12, max(4, len(all_obs) * 0.15 + 1)),
                          sharey=True)

cmap = sns.color_palette("Blues", as_cmap=True)

policy_data   = [mat_pi1, mat_pi2, mat_pi3]
policy_labels = [r"$\pi_1$", r"$\pi_2$", r"$\pi_3$"]
policy_names  = ["pi1", "pi2", "pi3"]

for ax, mat, plabel, pname in zip(axes, policy_data, policy_labels, policy_names):
    sns.heatmap(
        mat,
        ax=ax,
        cmap=cmap,
        vmin=0, vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        xticklabels=ACTION_LABELS,
        yticklabels=obs_labels, #if pname == "pi1" else [],
        cbar=pname == "pi3",
        cbar_kws={"label": "Action Probability", "shrink": 0.8} if pname == "pi3" else {},
        annot_kws={"size": 8},
    )
    ax.set_title(f"Policy {plabel}", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Action", fontsize=11)
    if pname == "pi1":
        ax.set_ylabel(r"Observation $(m_c,\, m_d)$", fontsize=11)
    ax.tick_params(axis='x', labelsize=10, rotation=0)
    ax.tick_params(axis='y', labelsize=9, rotation=0)

plt.tight_layout()
plt.savefig("policy_heatmaps_resources.png", dpi=150, bbox_inches="tight")
print("Saved: policy_heatmaps_resources.png")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Compute per-observation metrics for the bar chart figure
# ─────────────────────────────────────────────────────────────────────────────

N_ROLLOUTS    = 10
ROLLOUT_STEPS = 10

# ── Visit ratios (merge frequencies from all runs each policy participates in)
def merge_freq(*freq_dicts):
    merged = {}
    for fd in freq_dicts:
        for obs, cnt in fd.items():
            merged[obs] = merged.get(obs, 0) + cnt
    return merged

def visit_ratio(freq_dict):
    total = sum(freq_dict.values()) or 1
    return {obs: count / total for obs, count in freq_dict.items()}

vr_pi1 = visit_ratio(merge_freq(conf_12.p1_state_visit_frequency,
                                 conf_13.p1_state_visit_frequency))
vr_pi2 = visit_ratio(merge_freq(conf_12.p2_state_visit_frequency,
                                 conf_23.p1_state_visit_frequency))
vr_pi3 = visit_ratio(merge_freq(conf_13.p2_state_visit_frequency,
                                 conf_23.p2_state_visit_frequency))

# ── EMAC ─────────────────────────────────────────────────────────────────────
emac_12 = conf_12.compute_state_action_emd(remap_probs(raw_pi1_12), remap_probs(raw_pi2))
emac_13 = conf_13.compute_state_action_emd(remap_probs(raw_pi1_13), remap_probs(raw_pi3))
emac_23 = conf_23.compute_state_action_emd(remap_probs(raw_pi2_23), remap_probs(raw_pi3_23))

# ── PERG ─────────────────────────────────────────────────────────────────────
print("Computing PERG π₁ vs π₂...")
_, _, perg_12 = conf_12.expected_reward_run(
    pi1_solver, pi2_solver, rho=0, eta=0,
    num_rollouts=N_ROLLOUTS, num_steps=ROLLOUT_STEPS)
print("Computing PERG π₁ vs π₃...")
_, _, perg_13 = conf_13.expected_reward_run(
    pi1_solver, pi3_solver, rho=0, eta=0,
    num_rollouts=N_ROLLOUTS, num_steps=ROLLOUT_STEPS)
print("Computing PERG π₂ vs π₃...")
_, _, perg_23 = conf_23.expected_reward_run(
    pi2_solver, pi3_solver, rho=0, eta=0,
    num_rollouts=N_ROLLOUTS, num_steps=ROLLOUT_STEPS)

# ── Shared observation axis ───────────────────────────────────────────────────
metrics_obs = sorted(
    set(vr_pi1) | set(vr_pi2) | set(vr_pi3)
    | set(emac_12) | set(emac_13) | set(emac_23)
    | set(perg_12) | set(perg_13) | set(perg_23)
)
x     = np.arange(len(metrics_obs))
w     = 0.25   # bar width
x_lbl = [fmt_obs(o) for o in metrics_obs]

# ─────────────────────────────────────────────────────────────────────────────
# Three independent bar chart figures
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    r"$\pi_1$":            "#4C72B0",
    r"$\pi_2$":            "#DD8452",
    r"$\pi_3$":            "#55A868",
    r"$\pi_1$ vs $\pi_2$": "#4C72B0",
    r"$\pi_1$ vs $\pi_3$": "#DD8452",
    r"$\pi_2$ vs $\pi_3$": "#55A868",
}

fig_w = max(4, len(metrics_obs) * 0.16 + 1)

def set_obs_xticks(ax):
    ax.set_xticks(x)
    ax.set_xticklabels(x_lbl, rotation=90, ha="center", fontsize=9)
    ax.set_xlabel(r"Observation $(m_c,\, m_d)$", fontsize=11)

# ── Figure 1: Visit ratio ─────────────────────────────────────────────────────
fig_vr, ax_vr = plt.subplots(figsize=(fig_w, 4))
for i, (lbl, vr) in enumerate([
    (r"$\pi_1$", vr_pi1),
    (r"$\pi_2$", vr_pi2),
    (r"$\pi_3$", vr_pi3),
]):
    vals = [vr.get(o, 0.0) for o in metrics_obs]
    ax_vr.bar(x + (i - 1) * w, vals, width=w, label=lbl,
              color=COLORS[lbl], alpha=0.85, edgecolor="white")
ax_vr.set_ylabel("Visit ratio", fontsize=11)
ax_vr.set_title("Visit ratio per observation", fontsize=12, fontweight="bold")
ax_vr.legend(fontsize=9)
ax_vr.set_ylim(0, None)
ax_vr.yaxis.grid(True, linestyle="--", alpha=0.5)
ax_vr.set_axisbelow(True)
set_obs_xticks(ax_vr)
fig_vr.tight_layout()
fig_vr.savefig("metrics_visit_ratio.png", dpi=150, bbox_inches="tight")
print("Saved: metrics_visit_ratio.png")
plt.show()

# ── Figure 2: EMAC ────────────────────────────────────────────────────────────
fig_emac, ax_emac = plt.subplots(figsize=(fig_w, 4))
for i, (lbl, emac) in enumerate([
    (r"$\pi_1$ vs $\pi_2$", emac_12),
    (r"$\pi_1$ vs $\pi_3$", emac_13),
    (r"$\pi_2$ vs $\pi_3$", emac_23),
]):
    vals = [emac.get(o, 0.0) for o in metrics_obs]
    ax_emac.bar(x + (i - 1) * w, vals, width=w, label=lbl,
                color=COLORS[lbl], alpha=0.85, edgecolor="white")
ax_emac.set_ylabel("EMAC", fontsize=11)
ax_emac.set_title("EMAC per observation", fontsize=12, fontweight="bold")
ax_emac.legend(fontsize=9)
ax_emac.set_ylim(0, 1)
ax_emac.yaxis.grid(True, linestyle="--", alpha=0.5)
ax_emac.set_axisbelow(True)
set_obs_xticks(ax_emac)
fig_emac.tight_layout()
fig_emac.savefig("metrics_emac.png", dpi=150, bbox_inches="tight")
print("Saved: metrics_emac.png")
plt.show()

# ── Figure 3: PERG ────────────────────────────────────────────────────────────
fig_perg, ax_perg = plt.subplots(figsize=(fig_w, 4))
for i, (lbl, perg) in enumerate([
    (r"$\pi_1$ vs $\pi_2$", perg_12),
    (r"$\pi_1$ vs $\pi_3$", perg_13),
    (r"$\pi_2$ vs $\pi_3$", perg_23),
]):
    vals = [perg.get(o, 0.0) for o in metrics_obs]
    ax_perg.bar(x + (i - 1) * w, vals, width=w, label=lbl,
                color=COLORS[lbl], alpha=0.85, edgecolor="white")
ax_perg.axhline(0, color="black", linewidth=0.8)
ax_perg.set_ylabel("PERG", fontsize=11)
ax_perg.set_title("PERG per observation", fontsize=12, fontweight="bold")
ax_perg.legend(fontsize=9)
ax_perg.yaxis.grid(True, linestyle="--", alpha=0.5)
ax_perg.set_axisbelow(True)
set_obs_xticks(ax_perg)
fig_perg.tight_layout()
fig_perg.savefig("metrics_perg.png", dpi=150, bbox_inches="tight")
print("Saved: metrics_perg.png")
plt.show()