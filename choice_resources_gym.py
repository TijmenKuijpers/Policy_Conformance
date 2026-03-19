
import sys
import copy
import numpy as np
import pandas as pd
import random
sys.path.append("C:/Users/20183272/OneDrive - TU Eindhoven/Documents/GitHub/gympn")

import torch
from gympn.networks import HeteroActor
from torch_geometric.nn.conv.han_conv import HANConv
from torch_geometric.nn.aggr.basic import SumAggregation

# Add all necessary classes to safe globals
torch.serialization.add_safe_globals([HeteroActor, HANConv, SumAggregation])

from simpn.simulator import SimToken
from simpn.visualisation import Visualisation
from gympn.simulator import GymProblem
from simpn.reporters import SimpleReporter
from gympn.solvers import HeuristicSolver, GymSolver, RandomSolver

from policy_conformance import PolicyConformance


# Override HeuristicSolver to include an extra static function
class HeuristicSolver(HeuristicSolver):
    @staticmethod
    def get_place_tokens_by_time(place_id: str, pn):
        """
        Get the tokens of a place in the process network by time. The tokens are filtered by the current clock of the process network.

        Parameters
        ----------
        :param place_id: the id of the place.
        :param pn: the GymProblem object representing the (observable) petri net.

        Returns
        ----------
        :return: list of active token values in the place (i.e. tokens with time <= current clock).
        """

        return [t.value for p in pn.places if p._id == place_id for t in p.marking if t.time <= pn.clock]


# Set up the assembly system model
assembly_system = GymProblem(allow_postpone=True, causal_rl=False)

# Create a separate random number generator for this instance, seeded deterministically
# This ensures the same sequence across different GymProblem instances
assembly_system.rng = np.random.default_rng(42)  # Fixed seed for reproducibility

# Define the variables that make up the state-space of the system
arrival_chip = assembly_system.add_var("chip supply", var_attributes=["chip_id"])
arrival_phone_case = assembly_system.add_var("phone case supply", var_attributes=["phone_case_id"])

stock_chip = assembly_system.add_var("stock_chip", var_attributes=["chip_id"])
stock_phone_case = assembly_system.add_var("stock_phone_case", var_attributes=["phone_case_id"])

phone_resource = assembly_system.add_var("phone_resource", var_attributes=["phone_id"])
game_resource = assembly_system.add_var("game_resource", var_attributes=["game_id"])

assembly_system.set_unobservable(token_attrs={'stock_chip': ['chip_id'], 
                                              'stock_phone_case': ['phone_case_id'], 
                                              'phone_resource': ['phone_id'],
                                              'game_resource': ['game_id']}) 

# Define the events that cause arrivals
def chip_arrival(arrival):
    # Use the instance's RNG, seeded with chip_id to get different values each call
    # but same sequence across instances (since chip_ids are the same)
    chip_id = arrival["chip_id"] + 1
    # Create a temporary RNG with seed based on chip_id for this specific call
    temp_rng = np.random.default_rng(42 + chip_id)
    delay = temp_rng.exponential(scale=3.0)
    new_arrival = {"chip_id": chip_id}

    return [SimToken(new_arrival, delay=delay), SimToken(new_arrival, delay=delay)]


def phone_case_arrival(arrival):
    # Use the instance's RNG, seeded with phone_case_id to get different values each call
    # but same sequence across instances (since phone_case_ids are the same)
    phone_case_id = arrival["phone_case_id"] + 1
    # Create a temporary RNG with seed based on phone_case_id for this specific call
    temp_rng = np.random.default_rng(42 + phone_case_id)
    delay = temp_rng.exponential(scale=5.0)
    new_arrival = {"phone_case_id": phone_case_id}

    return [SimToken(new_arrival, delay=delay), SimToken(new_arrival, delay=delay)]


assembly_system.add_event([arrival_chip], [arrival_chip, stock_chip],
                          behavior=chip_arrival,
                          name="chip_arrival")

assembly_system.add_event([arrival_phone_case], [arrival_phone_case, stock_phone_case],
                          behavior=phone_case_arrival,
                          name="phone_case_arrival")

assembly_system.add_action([stock_chip, game_resource], [game_resource],
                           behavior=lambda stock_chip, game_resource: [SimToken(game_resource, delay=1)],
                           reward_function=lambda stock_chip, game_resource: 1,
                           name="game_production")

assembly_system.add_action([stock_chip, stock_phone_case, phone_resource], [phone_resource],
                           behavior=lambda stock_chip, stock_phone_case, phone_resource: [SimToken(phone_resource, delay=2)],
                           reward_function=lambda stock_chip, stock_phone_case, phone_resource: 3,
                           name="phone_production")

# Describe the initial state of the system
arrival_chip.put({"chip_id": 0})
arrival_phone_case.put({"phone_case_id": 0})
game_resource.put({"game_id": 1})
phone_resource.put({"phone_id": 1})

# Define the heuristic policy
def choice_pattern(pn, actions_dict):
    active_stock_chip = HeuristicSolver.get_place_tokens_by_time(place_id='stock_chip', pn=pn)
    active_stock_phone_case = HeuristicSolver.get_place_tokens_by_time(place_id='stock_phone_case', pn=pn)
    active_game_resource = HeuristicSolver.get_place_tokens_by_time(place_id='game_resource', pn=pn)
    active_phone_resource = HeuristicSolver.get_place_tokens_by_time(place_id='phone_resource', pn=pn)

    if len(active_stock_phone_case) > 0 and len(active_stock_chip) > 1 and len(active_phone_resource) > 0:
        return {'phone_production': actions_dict['phone_production'][0]}

    # If we have 3 or more chips, and we cannot produce phones, produce games
    elif len(active_stock_phone_case) == 0 and len(active_stock_chip) >= 5 and len(active_game_resource) > 0:
        return {'game_production': actions_dict['game_production'][0]}

    else:  # If we cannot produce phones and we have less than 3 chips, postpone
        return 'postpone'

# Define the heuristic policy
def choice_pattern_pi3(pn, actions_dict):
    active_stock_chip = HeuristicSolver.get_place_tokens_by_time(place_id='stock_chip', pn=pn)
    active_stock_phone_case = HeuristicSolver.get_place_tokens_by_time(place_id='stock_phone_case', pn=pn)
    active_game_resource = HeuristicSolver.get_place_tokens_by_time(place_id='game_resource', pn=pn)
    active_phone_resource = HeuristicSolver.get_place_tokens_by_time(place_id='phone_resource', pn=pn)

    if len(active_stock_phone_case) > 0 and len(active_stock_chip) > 1 and len(active_phone_resource) > 0:
        return {'phone_production': actions_dict['phone_production'][0]}

    # If we have 3 or more chips, and we cannot produce phones, produce games
    elif len(active_stock_phone_case) == 0 and len(active_stock_chip) >= 2 and len(active_game_resource) > 0:
        return {'game_production': actions_dict['game_production'][0]}

    else:  # If we cannot produce phones and we have less than 3 chips, postpone
        return 'postpone'

# After simulation, print some statistics
train = False
test = True
visualize = False
conformance_analysis = False

w_p = "data/train/2026-03-14-15-50-20_run/best_policy.pth"

if train:
    print("Training the model...")
    assembly_system.training_run(length=100, args_dict={"open_tensorboard": True, "verbose": True})
    print("Model trained successfully")

if test:
    frozen_pn = copy.deepcopy(assembly_system)


    episodes = 20
    print("Testing the Heuristic model...")
    rewards_pi1 = []
    for episode in range(episodes):
        assembly_system = copy.deepcopy(frozen_pn)
        heuristic_model = HeuristicSolver(heuristic_function=choice_pattern)
        rewards = assembly_system.testing_run(solver=heuristic_model, length=100, visualize=visualize)
        rewards_pi1.append(rewards)

    print("Testing the Heuristic model (pi3)...")
    rewards_pi3 = []
    for episode in range(episodes):
        assembly_system = copy.deepcopy(frozen_pn)
        heuristic_model_pi3 = HeuristicSolver(heuristic_function=choice_pattern_pi3)
        rewards = assembly_system.testing_run(solver=heuristic_model_pi3, length=100, visualize=visualize)
        rewards_pi3.append(rewards)

    #print("Testing the Random model...")
    #random_model = RandomSolver()
    #rewards = assembly_system.testing_run(solver=random_model, length=100, visualize=visualize)
    #print("Random model rewards: ", rewards)

    print("Testing the PPO model...")
    rewards_ppo = []
    for episode in range(episodes):
        assembly_system = copy.deepcopy(frozen_pn)
        trained_model = GymSolver(weights_path=w_p, metadata= assembly_system.make_metadata())
        rewards = assembly_system.testing_run(solver=trained_model, length=100, visualize=visualize)
        rewards_ppo.append(rewards)
    
    print("Heuristic model (pi1) rewards: ", np.average(rewards_pi1))
    print("Heuristic model (pi1) rewards standard deviation: ", np.std(rewards_pi1))
    print(" ")
    
    print("Heuristic model (pi3) rewards: ", np.average(rewards_pi3))
    print("Heuristic model (pi3) rewards standard deviation: ", np.std(rewards_pi3))
    print(" ")
    
    print("PPO model rewards: ", np.average(rewards_ppo))
    print("PPO model rewards standard deviation: ", np.std(rewards_ppo))
    print(" ")

if conformance_analysis:
    print("Analyzing the conformance of the heuristic model to the PPO model...")
    frozen_pn = copy.deepcopy(assembly_system)
    assembly_system = copy.deepcopy(frozen_pn)

    heuristic_model = HeuristicSolver(heuristic_function=choice_pattern)
    heuristic_model_pi3 = HeuristicSolver(heuristic_function=choice_pattern_pi3)
    trained_model = GymSolver(weights_path=w_p, metadata=assembly_system.make_metadata())
    state_variables = [lambda stock_chip_queue_tokens, clock: len([token for token in stock_chip_queue_tokens if token.time <= clock]),
                       lambda stock_phone_case_queue_tokens, clock: len([token for token in stock_phone_case_queue_tokens if token.time <= clock])]
                       #lambda game_resource_queue_tokens, clock: len([token for token in game_resource_queue_tokens if token.time <= clock]),
                       #lambda phone_resource_queue_tokens, clock: len([token for token in phone_resource_queue_tokens if token.time <= clock])]
    
    conformance_analysis = PolicyConformance(assembly_system, trained_model, heuristic_model_pi3, state_variables,
                                              excluded_token_attrs=["chip_id", "phone_case_id", "game_id", "phone_id"])

    # Run N episodes, accumulating visit frequencies and action mappings across
    # all runs so that observations visited in any episode are covered.
    N_EPISODES = 5
    for episode in range(N_EPISODES):
        print(f"Episode {episode + 1}/{N_EPISODES}...")
        conformance_analysis.compute_state_action_mapping(horizon=100)

    # Use the accumulated instance dicts (union of all episodes)
    heuristic_o_visits         = conformance_analysis.p1_state_visit_frequency
    gym_o_visits               = conformance_analysis.p2_state_visit_frequency
    heuristic_o_action_mapping = conformance_analysis.p1_state_action_mapping
    gym_o_action_mapping       = conformance_analysis.p2_state_action_mapping

    # Compute the action probability for the heuristic and the gym policy
    heuristic_pa = conformance_analysis.compute_action_probability(heuristic_o_action_mapping)
    gym_pa = conformance_analysis.compute_action_probability(gym_o_action_mapping)

    # Compute the Earth Mover's Distance between the heuristic and the gym policy
    o_action_emd = conformance_analysis.compute_state_action_emd(heuristic_pa,gym_pa)

    # Reward rollouts
    expected_return_p1, expected_return_p2, delta_expected_return = conformance_analysis.expected_reward_run(heuristic_model, 
                                                                                                            trained_model, 
                                                                                                            rho=0.4, 
                                                                                                            eta=0, 
                                                                                                            num_rollouts=10, 
                                                                                                            num_steps=20)

    print("\n" + "=" * 80)
    print("Conformance analysis complete!")
    print("=" * 80 + "\n")

    # =====================================================================
    # PER-POLICY BREAKDOWN TABLES
    # =====================================================================

    def fmt_obs(obs):
        """Format observation tuple as a readable string."""
        labels = ["$m_c$", "$m_d$"]
        parts = [f"{labels[i]}={v}" for i, v in enumerate(obs)]
        return "(" + ", ".join(parts) + ")"

    def build_policy_table(visit_freq, action_prob, expected_return,
                           all_actions, obs_labels):
        """
        Build a DataFrame with one row per observation.

        Columns:
          Observation     — formatted observation label
          Visit Freq      — raw visit count for this policy
          P(<action>)     — probability of each action (0 if not present)
          Exp. Return     — expected return from rollouts (NaN if not evaluated)
        """
        all_obs = sorted(
            set(visit_freq.keys()) | set(action_prob.keys())
        )
        rows = []
        for obs in all_obs:
            row = {"Observation": obs_labels(obs),
                   "Visit Freq": visit_freq.get(obs, 0)}
            probs = action_prob.get(obs, {})
            for action in all_actions:
                row[f"P({action})"] = round(probs.get(action, 0.0), 4)
            er = expected_return.get(obs, np.nan)
            row["Exp. Return"] = round(er, 4) if not np.isnan(er) else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    # Collect all action names seen by either policy (strip binding suffix after '|')
    def short_action(key):
        base = key.split("|")[0]
        # 'None' arises when a postpone binding's transition is not Python None
        # but str()-evaluates to "None"; normalise to the canonical name.
        return "postpone" if base == "None" else base

    all_action_keys = sorted(
        {short_action(a)
         for obs_probs in list(heuristic_pa.values()) + list(gym_pa.values())
         for a in obs_probs}
    )

    # Remap probability dicts to short action names
    def remap_probs(pa):
        remapped = {}
        for obs, probs in pa.items():
            remapped[obs] = {}
            for a, p in probs.items():
                short = short_action(a)
                remapped[obs][short] = remapped[obs].get(short, 0.0) + p
        return remapped

    heuristic_pa_short = remap_probs(heuristic_pa)
    gym_pa_short       = remap_probs(gym_pa)

    df_heuristic = build_policy_table(heuristic_o_visits, heuristic_pa_short,
                                      expected_return_p1,
                                      all_action_keys, fmt_obs)
    df_gym       = build_policy_table(gym_o_visits,       gym_pa_short,
                                      expected_return_p2,
                                      all_action_keys, fmt_obs)

    for label, df, file_suffix in [
        ("HEURISTIC POLICY", df_heuristic, "heuristic"),
        ("GYM (PPO) POLICY", df_gym,       "gym"),
    ]:
        print(f"\n--- {label} ---")
        print(df.to_string(index=False))

        latex = df.to_latex(
            index=False,
            float_format="%.4f",
            na_rep="-",
            caption=f"Per-observation visit frequency and action probabilities — {label.lower()}.",
            label=f"tab:policy_{file_suffix}_resources",
        )
        name = f"policy_table_{file_suffix}_resources"
        df.to_csv(f"{name}.csv", index=False)
        with open(f"{name}.tex", "w") as f:
            f.write(latex)
        print(f"Saved: {name}.csv, {name}.tex")

    # =====================================================================
    # RESULTS TABLE  (Observation | Max Visit Ratio | EMAC | PERG)
    # =====================================================================
    # Retrieve the same max visit ratio used by expected_reward_run
    max_visit_ratios = conformance_analysis.compute_max_visit_ratio()

    all_obs = sorted(max_visit_ratios.keys())

    rows = []
    for obs in all_obs:
        emac = o_action_emd.get(obs, np.nan)
        perg = delta_expected_return.get(obs, np.nan)
        rows.append({
            "Observation": fmt_obs(obs),
            "$N_{ratio}$": round(max_visit_ratios[obs], 4),
            "$EMAC$": round(emac, 4) if not np.isnan(emac) else np.nan,
            "$PERG$": round(perg, 4) if not np.isnan(perg) else np.nan,
        })

    df_results = pd.DataFrame(rows)
    
    print("\n" + "=" * 80)
    print("CONFORMANCE RESULTS TABLE")
    print("=" * 80)

    print(df_results.to_string(index=False, na_rep="-"))

    # LaTeX table
    latex_table = df_results.to_latex(
        index=False,
        float_format="%.4f",
        na_rep="-",
        caption="Per-observation policy conformance metrics for the production system. ",
        label="tab:conformance_choice",
    )
    #print("\n--- LaTeX ---")
    #print(latex_table)

    # Save artefacts
    name = "conformance_results_choice_resources"
    df_results.to_csv(f"{name}.csv", index=False)
    with open(f"{name}.tex", "w") as f:
        f.write(latex_table)
    print(f"\nSaved: {name}.csv, {name}.tex")
