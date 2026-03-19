
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import copy
import numpy as np
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

assembly_system.set_unobservable(
    token_attrs={'stock_chip': ['chip_id'], 'stock_phone_case': ['phone_case_id'], 'phone_resource': ['phone_id'],
                 'game_resource': [
                     'game_id']})  # set_unobservable(token_attrs={"stock_chip": ["chip_id"], "stock_phone_case": ["phone_case_id"], "phone_resource": ["phone_id"], "game_resource": ["game_resource"]})


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
                           behavior=lambda stock_chip, stock_phone_case, phone_resource: [
                               SimToken(phone_resource, delay=2)],
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


# After simulation, print some statistics
train = False
test = False
visualize = False
conformance_analysis = True

w_p = "C:/Users/20183272/OneDrive - TU Eindhoven/Documents/PhD IS/Papers/Policy conformance analysis/data/train/2026-03-09-17-01-54_run/best_policy.pth"

if train:
    print("Training the model...")
    assembly_system.training_run(length=100, args_dict={"open_tensorboard": True, "verbose": True})
    print("Model trained successfully")

if test:
    frozen_pn = copy.deepcopy(assembly_system)

    print("Testing the Heuristic model...")
    assembly_system = copy.deepcopy(frozen_pn)
    heuristic_model = HeuristicSolver(heuristic_function=choice_pattern)
    rewards = assembly_system.testing_run(solver=heuristic_model, length=100, visualize=visualize)
    print("Heuristic model rewards: ", rewards)

    # print("Testing the Random model...")
    # random_model = RandomSolver()
    # rewards = assembly_system.testing_run(solver=random_model, length=100, visualize=visualize)
    # print("Random model rewards: ", rewards)

    # print("Testing the PPO model...")
    # assembly_system = copy.deepcopy(frozen_pn)
    # trained_model = GymSolver(weights_path=w_p, metadata= assembly_system.make_metadata())
    # rewards = assembly_system.testing_run(solver=trained_model, length=100, visualize=visualize)
    # print("PPO model rewards: ", rewards)

if conformance_analysis:
    print("Analyzing the conformance of the heuristic model to the PPO model...")
    frozen_pn = copy.deepcopy(assembly_system)
    assembly_system = copy.deepcopy(frozen_pn)

    heuristic_model = HeuristicSolver(heuristic_function=choice_pattern)
    trained_model = GymSolver(weights_path=w_p, metadata=assembly_system.make_metadata())
    state_variables = [
        lambda stock_chip_queue_tokens, clock: len([token for token in stock_chip_queue_tokens if token.time <= clock]),
        lambda stock_phone_case_queue_tokens, clock: len(
            [token for token in stock_phone_case_queue_tokens if token.time <= clock])]
    # lambda game_resource_queue_tokens, clock: len([token for token in game_resource_queue_tokens if token.time <= clock]),
    # lambda phone_resource_queue_tokens, clock: len([token for token in phone_resource_queue_tokens if token.time <= clock])]

    conformance_analysis = PolicyConformance(assembly_system, heuristic_model, trained_model, state_variables,
                                              excluded_token_attrs=["chip_id", "phone_case_id", "game_id", "phone_id"])

    # Analyze state-visit frequency between the heuristic and the gym policy
    heuristic_state_visits, gym_state_visits = conformance_analysis.compute_state_visit_frequency(
        horizon=100)  # dictionary with state as key and number of visits as value

    # Analyze state-action mapping between the heuristic and the gym policy
    heuristic_state_action_mapping, gym_state_action_mapping = conformance_analysis.compute_state_action_mapping(
        horizon=100)  # dictionary with state as key and actions as value

    # Compute the action probability for the heuristic and the gym policy
    heuristic_action_probability = conformance_analysis.compute_action_probability(heuristic_state_action_mapping)
    gym_action_probability = conformance_analysis.compute_action_probability(gym_state_action_mapping)

    # Compute the Earth Mover's Distance between the heuristic and the gym policy
    state_action_emd = conformance_analysis.compute_state_action_emd(heuristic_action_probability,
                                                                     gym_action_probability)

    # Reward rollouts
    expected_reward_p1, expected_reward_p2, delta_expected_reward = conformance_analysis.expected_reward_run(
        heuristic_model, trained_model, tau=0.4, gamma=0.95, num_rollouts=10, num_steps=10)

    print(" ")
    print("Heuristic state visit frequency: ", dict(sorted(heuristic_state_visits.items())))
    print("Gym state visit frequency: ", dict(sorted(gym_state_visits.items())))
    print(" ")
    print("Heuristic action probability: ", dict(sorted(heuristic_action_probability.items())))
    print("Gym action probability: ", dict(sorted(gym_action_probability.items())))
    print(" ")
    print("State-action EMD: ", dict(sorted(state_action_emd.items())))
    print(" ")
    print("States in observation: ",
          dict(sorted({key: len(value) for key, value in conformance_analysis.states_in_observation.items()}.items())))
    print("Total states observed: ", sum({len(value) for value in conformance_analysis.states_in_observation.values()}))
    print(" ")
    print("Expected reward P1: ", dict(sorted(conformance_analysis.expected_reward_p1.items())))
    print("Expected reward P2: ", dict(sorted(conformance_analysis.expected_reward_p2.items())))
    print("Delta expected reward: ", dict(sorted(conformance_analysis.delta_expected_reward.items())))

    # =====================================================================
    # CONFORMANCE ANALYSIS SUMMARY
    # =====================================================================
    print("\n" + "=" * 80)
    print("CONFORMANCE ANALYSIS SUMMARY")
    print("=" * 80)

    # --- State Visit Conformance ---
    all_states = set(heuristic_state_visits.keys()) | set(gym_state_visits.keys())
    overlapping_states = set(heuristic_state_visits.keys()) & set(gym_state_visits.keys())
    overlap_ratio = len(overlapping_states) / len(all_states) if all_states else 0

    print(f"\nState Visit Conformance:")
    print(f"  - Heuristic unique states: {len(heuristic_state_visits)}")
    print(f"  - Trained unique states:   {len(gym_state_visits)}")
    print(f"  - Overlapping states:      {len(overlapping_states)}")
    print(f"  - State overlap ratio:     {overlap_ratio:.2%}")
    if overlap_ratio > 0.7:
        print(f"    -> GOOD: Both policies explore similar state space")
    elif overlap_ratio > 0.4:
        print(f"    -> MEDIUM: Policies have moderate state space overlap")
    else:
        print(f"    -> LOW: Policies explore different state spaces")

    # --- Action Agreement Conformance ---
    common_states = set(heuristic_action_probability.keys()) & set(gym_action_probability.keys())
    action_agreement = 0
    for state in common_states:
        h_actions = heuristic_action_probability[state]
        g_actions = gym_action_probability[state]
        h_best = max(h_actions, key=h_actions.get)
        g_best = max(g_actions, key=g_actions.get)
        if h_best == g_best:
            action_agreement += 1
    agreement_ratio = action_agreement / len(common_states) if common_states else 0

    print(f"\nAction Agreement Conformance:")
    print(f"  - Common states:           {len(common_states)}")
    print(f"  - Same best action:        {action_agreement}/{len(common_states)}")
    print(f"  - Action agreement ratio:  {agreement_ratio:.2%}")
    if agreement_ratio > 0.7:
        print(f"    -> GOOD: Policies often choose same actions")
    elif agreement_ratio > 0.4:
        print(f"    -> MEDIUM: Policies sometimes choose same actions")
    else:
        print(f"    -> LOW: Policies have very different action distributions")

    # --- EMD Conformance ---
    emd_values = list(state_action_emd.values())
    if emd_values:
        avg_emd = np.mean(emd_values)
        max_emd = np.max(emd_values)
        states_with_diff = sum(1 for v in emd_values if v > 0)
        print(f"\nEarth Mover's Distance (EMD) Conformance:")
        print(f"  - Average EMD:             {avg_emd:.4f}")
        print(f"  - Max EMD:                 {max_emd:.4f}")
        print(f"  - States with EMD > 0:     {states_with_diff}/{len(emd_values)}")
        if avg_emd < 0.1:
            print(f"    -> GOOD: Action distributions are very similar")
        elif avg_emd < 0.3:
            print(f"    -> MEDIUM: Action distributions moderately differ")
        else:
            print(f"    -> LOW: Action distributions are substantially different")

    # --- Expected Reward Conformance ---
    if conformance_analysis.delta_expected_reward:
        delta_values = list(conformance_analysis.delta_expected_reward.values())
        avg_delta = np.mean(delta_values)
        p1_total = sum(conformance_analysis.expected_reward_p1.values())
        p2_total = sum(conformance_analysis.expected_reward_p2.values())
        n_states_evaluated = len(delta_values)

        print(f"\nExpected Reward Conformance (rollout-based):")
        print(f"  - States evaluated:        {n_states_evaluated}")
        print(f"  - Heuristic total reward:  {p1_total:.2f}")
        print(f"  - Trained total reward:    {p2_total:.2f}")
        print(f"  - Avg reward delta (P1-P2):{avg_delta:.4f}")
        if avg_delta > 0:
            print(f"    -> Heuristic outperforms trained policy on average")
        elif avg_delta < 0:
            print(f"    -> Trained policy outperforms heuristic on average")
        else:
            print(f"    -> Policies perform equally on average")

    print("\n" + "=" * 80)
    print("Conformance analysis complete!")
    print("=" * 80 + "\n")
