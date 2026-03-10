"""
Policy Conformance Analysis: Trained BPI_2012 Policy vs Random Policy

This script compares the trained PPO policy for BPI Challenge 2012 task assignment
against a random baseline policy using policy conformance metrics:
- State visit frequency (how often are the same states visited?)
- Action overlap (how often do they choose the same action in the same state?)
- Expected reward gap (how much better/worse is one policy?)
"""

import sys
import os
import copy
import numpy as np
import torch

sys.path.append("C:/Users/20183272/OneDrive - TU Eindhoven/Documents/GitHub/gympn")
#sys.path.append("C:/Users/lobia/PycharmProjects/policy_comparison/Policy_Conformance")

from gympn.solvers import GymSolver, RandomSolver
from gympn_problem_from_json import load_parameters, create_task_assignment_problem
from gympn_problem_disjoint import create_disjoint_task_assignment_problem
from policy_conformance import PolicyConformance

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def dprint(msg):
    """Print with flushing"""
    print(msg, flush=True)


def create_state_variables(disjoint_actions=False, parameters=None):
    """
    Define state variables for conformance analysis based on the actual
    observable network marking (token counts per place).

    The parameter names MUST match the place names in the GymProblem with
    "_queue" suffix.  The evaluate() method in policy_conformance.py strips
    that suffix to look up the SimVar.

    With the sequential case-flow model, both variants have:
      - resource_pool
      - waiting_{act}  (one per mapped activity)
      - busy_{act}     (one per mapped activity)
    """

    # ── Resource pool (always present) ──────────────────────────────────
    def n_available(resource_pool_queue):
        """Exact number of idle resources."""
        return len(resource_pool_queue) if resource_pool_queue else 0

    state_vars = [n_available]

    if parameters is not None:
        activity_resource_mapping = parameters.get('activity_resource_mapping', {})
        activities = parameters.get('activities', [])
        mapped_activities = [a for a in activities if a in activity_resource_mapping]

        for activity_name in mapped_activities:
            safe = activity_name.replace(" ", "_").replace(".", "_")

            # waiting_{act} count
            w_place = f"waiting_{safe}"
            w_code = (
                f"def n_waiting_{safe}({w_place}_queue):\n"
                f"    return len({w_place}_queue) if {w_place}_queue else 0\n"
            )
            w_ns = {}
            exec(w_code, {}, w_ns)
            state_vars.append(w_ns[f"n_waiting_{safe}"])

            # busy_{act} count
            b_place = f"busy_{safe}"
            b_code = (
                f"def n_busy_{safe}({b_place}_queue):\n"
                f"    return len({b_place}_queue) if {b_place}_queue else 0\n"
            )
            b_ns = {}
            exec(b_code, {}, b_ns)
            state_vars.append(b_ns[f"n_busy_{safe}"])

    return state_vars


if __name__ == "__main__":

    dprint("=" * 80)
    dprint("POLICY CONFORMANCE ANALYSIS: BPI_2012 Trained Policy vs Random Policy")
    dprint("=" * 80)

    # Configuration
    disjoint_actions = False  # True: one action per activity type; False: single "assign" action
    simulation_length = 100  # Length of simulation runs
    num_episodes = 10  # Number of episodes for conformance analysis

    variant = "disjoint" if disjoint_actions else "joint"
    model_save_dir = f"data/train/bpi_2012_ppo_{variant}"
    weights_path = os.path.join(model_save_dir, "best_policy.pth")

    dprint(f"\n[CONFIG] Problem variant: {'DISJOINT (one action per activity)' if disjoint_actions else 'JOINT (single assign action)'}")
    dprint(f"[CONFIG] Model path: {weights_path}")
    dprint(f"[CONFIG] Simulation length: {simulation_length}")
    dprint(f"[CONFIG] Number of episodes: {num_episodes}")

    # Step 1: Load the BPI_2012 task assignment problem
    dprint("\n[STEP 1] Loading problem from JSON parameters...")
    parameters = load_parameters("simulation_parameters.json")
    if disjoint_actions:
        problem = create_disjoint_task_assignment_problem(parameters)
    else:
        problem = create_task_assignment_problem(parameters)
    dprint(f"[SUCCESS] Problem created with {len(parameters['activities'])} activities and "
           f"{len([r for r in parameters['resources'] if r is not None and not (isinstance(r, float) and np.isnan(r))])} resources")

    # Step 2: Load the trained policy
    dprint("\n[STEP 2] Loading trained policy...")
    if not os.path.exists(weights_path):
        dprint(f"[ERROR] Trained model not found at {weights_path}")
        dprint("[ERROR] Please run BPI_2012_task_assignment_training.py with train=True first")
        sys.exit(1)

    try:
        # Create metadata from problem
        metadata = problem.make_metadata()
        trained_solver = GymSolver(weights_path=weights_path, metadata=metadata)
        dprint("[SUCCESS] Trained policy loaded successfully")
    except Exception as e:
        dprint(f"[ERROR] Failed to load trained policy: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 3: Create random solver baseline
    dprint("\n[STEP 3] Creating random policy baseline...")
    random_solver = RandomSolver()
    dprint("[SUCCESS] Random policy solver created")

    # Step 4: Define state variables for conformance analysis
    dprint("\n[STEP 4] Defining state variables for analysis...")
    state_variables = create_state_variables(disjoint_actions=disjoint_actions, parameters=parameters)
    dprint(f"[SUCCESS] Defined {len(state_variables)} state variables")

    # Step 5: Create PolicyConformance analyzer
    dprint("\n[STEP 5] Initializing PolicyConformance analyzer...")
    try:
        conformance = PolicyConformance(
            gym_problem=problem,
            heuristic_solver=random_solver,  # Using random as the "heuristic"
            gym_solver=trained_solver,        # Using trained policy as the "gym" policy
            state_variables=state_variables
        )
        dprint("[SUCCESS] PolicyConformance analyzer initialized")
    except Exception as e:
        dprint(f"[ERROR] Failed to initialize PolicyConformance: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 6: Run frequency analysis (state visit frequency for each policy)
    dprint("\n[STEP 6] Running state visit frequency analysis...")
    dprint(f"[INFO] Running {num_episodes} episodes with trained policy...")

    try:
        trained_frequencies = {}
        for episode in range(num_episodes):
            dprint(f"  Episode {episode + 1}/{num_episodes}...")
            freq = conformance.frequency_run(trained_solver, simulation_length)
            for state, count in freq.items():
                trained_frequencies[state] = trained_frequencies.get(state, 0) + count

        dprint(f"[SUCCESS] Trained policy visited {len(trained_frequencies)} unique states")

        dprint(f"[INFO] Running {num_episodes} episodes with random policy...")
        random_frequencies = {}
        for episode in range(num_episodes):
            dprint(f"  Episode {episode + 1}/{num_episodes}...")
            freq = conformance.frequency_run(random_solver, simulation_length)
            for state, count in freq.items():
                random_frequencies[state] = random_frequencies.get(state, 0) + count

        dprint(f"[SUCCESS] Random policy visited {len(random_frequencies)} unique states")

        # Calculate state overlap
        all_states = set(trained_frequencies.keys()) | set(random_frequencies.keys())
        overlapping_states = set(trained_frequencies.keys()) & set(random_frequencies.keys())
        overlap_ratio = len(overlapping_states) / len(all_states) if all_states else 0

        dprint(f"\n[ANALYSIS] State Visit Frequency:")
        dprint(f"  - Trained policy visited states: {len(trained_frequencies)}")
        dprint(f"  - Random policy visited states: {len(random_frequencies)}")
        dprint(f"  - Overlapping states: {len(overlapping_states)}")
        dprint(f"  - State overlap ratio: {overlap_ratio:.2%}")

    except Exception as e:
        dprint(f"[ERROR] Failed during frequency analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 7: Run action mapping analysis
    dprint("\n[STEP 7] Running action mapping analysis...")
    dprint("[INFO] Recording state-action pairs for both policies...")

    try:
        # Reset conformance object for action analysis
        conformance_action = PolicyConformance(
            gym_problem=problem,
            heuristic_solver=random_solver,
            gym_solver=trained_solver,
            state_variables=state_variables,
            excluded_token_attrs=["case_id", "task_id"]
        )

        # Run action episodes
        for episode in range(num_episodes):
            dprint(f"  Episode {episode + 1}/{num_episodes}...")
            conformance_action.action_run(trained_solver, random_solver, length=simulation_length)

        # Analyze action mappings
        p1_states = set(conformance_action.p1_state_action_mapping.keys())
        p2_states = set(conformance_action.p2_state_action_mapping.keys())
        common_states = p1_states & p2_states

        dprint(f"[SUCCESS] Analyzed action mappings")
        dprint(f"  - Trained policy action states: {len(p1_states)}")
        dprint(f"  - Random policy action states: {len(p2_states)}")
        dprint(f"  - Common states: {len(common_states)}")

        # Diagnostic: Print action diversity
        dprint(f"\n[DEBUG] Action diversity analysis:")
        all_trained_actions = set()
        all_random_actions = set()
        for state in p1_states:
            all_trained_actions.update(conformance_action.p1_state_action_mapping[state].keys())
        for state in p2_states:
            all_random_actions.update(conformance_action.p2_state_action_mapping[state].keys())

        dprint(f"  - Unique actions (trained policy): {len(all_trained_actions)}")
        dprint(f"  - Unique actions (random policy): {len(all_random_actions)}")
        dprint(f"  - Most common trained actions: {sorted([(a, sum(conformance_action.p1_state_action_mapping[s].get(a, 0) for s in p1_states)) for a in all_trained_actions], key=lambda x: x[1], reverse=True)[:5]}")
        dprint(f"  - Most common random actions: {sorted([(a, sum(conformance_action.p2_state_action_mapping[s].get(a, 0) for s in p2_states)) for a in all_random_actions], key=lambda x: x[1], reverse=True)[:5]}")

        # Calculate action agreement
        action_agreement = 0
        total_actions = 0

        # Debug: Store disagreement examples
        disagreement_examples = []

        for state in common_states:
            p1_actions = conformance_action.p1_state_action_mapping[state]
            p2_actions = conformance_action.p2_state_action_mapping[state]

            # Get most frequent actions
            p1_best_action = max(p1_actions, key=p1_actions.get)
            p2_best_action = max(p2_actions, key=p2_actions.get)

            if p1_best_action == p2_best_action:
                action_agreement += 1
            else:
                if len(disagreement_examples) < 5:  # Store first 5 disagreements
                    disagreement_examples.append({
                        'state': state,
                        'trained_action': p1_best_action,
                        'random_action': p2_best_action,
                        'trained_freq': p1_actions[p1_best_action],
                        'random_freq': p2_actions[p2_best_action]
                    })
            total_actions += 1

        agreement_ratio = action_agreement / total_actions if total_actions > 0 else 0
        dprint(f"\n[ANALYSIS] Action Agreement:")
        dprint(f"  - Common states with same best action: {action_agreement}/{total_actions}")
        dprint(f"  - Action agreement ratio: {agreement_ratio:.2%}")

        # Debug output
        if disagreement_examples:
            dprint(f"\n[DEBUG] Examples of action disagreement:")
            for i, example in enumerate(disagreement_examples, 1):
                dprint(f"  State {i}: {example['state']}")
                dprint(f"    - Trained: '{example['trained_action']}' (freq: {example['trained_freq']})")
                dprint(f"    - Random: '{example['random_action']}' (freq: {example['random_freq']})")
        else:
            dprint(f"\n[DEBUG] All common states have same best action - states may be too coarse")

    except Exception as e:
        dprint(f"[ERROR] Failed during action mapping analysis: {e}")
        import traceback
        traceback.print_exc()
        # Continue to next step

    # Step 8: Run performance comparison
    dprint("\n[STEP 8] Running performance comparison...")

    try:
        trained_rewards = []
        random_rewards = []

        for episode in range(5):  # Fewer episodes for performance runs
            dprint(f"  Episode {episode + 1}/5...")

            # Test trained policy
            test_problem_trained = copy.deepcopy(problem)
            test_problem_trained.set_solver(trained_solver)
            test_problem_trained.length = simulation_length
            active = True
            while test_problem_trained.clock <= test_problem_trained.length and active:
                _, active = test_problem_trained.step()
            trained_rewards.append(test_problem_trained.reward)

            # Test random policy
            test_problem_random = copy.deepcopy(problem)
            test_problem_random.set_solver(random_solver)
            test_problem_random.length = simulation_length
            active = True
            while test_problem_random.clock <= test_problem_random.length and active:
                _, active = test_problem_random.step()
            random_rewards.append(test_problem_random.reward)

        trained_avg = np.mean(trained_rewards)
        trained_std = np.std(trained_rewards)
        random_avg = np.mean(random_rewards)
        random_std = np.std(random_rewards)
        reward_diff = trained_avg - random_avg
        reward_ratio = trained_avg / random_avg if random_avg > 0 else 0

        dprint(f"\n[ANALYSIS] Performance Comparison:")
        dprint(f"  - Trained policy: {trained_avg:.2f} ± {trained_std:.2f}")
        dprint(f"  - Random policy: {random_avg:.2f} ± {random_std:.2f}")
        dprint(f"  - Reward difference: {reward_diff:.2f}")
        dprint(f"  - Reward ratio (trained/random): {reward_ratio:.2f}x")

    except Exception as e:
        dprint(f"[ERROR] Failed during performance comparison: {e}")
        import traceback
        traceback.print_exc()

    # Step 9: Summary and conclusions
    dprint("\n" + "=" * 80)
    dprint("CONFORMANCE ANALYSIS SUMMARY")
    dprint("=" * 80)

    dprint(f"\nState Visit Conformance:")
    dprint(f"  - State overlap: {overlap_ratio:.2%}")
    if overlap_ratio > 0.7:
        dprint(f"    → GOOD: Both policies explore similar state space")
    elif overlap_ratio > 0.4:
        dprint(f"    → MEDIUM: Policies have moderate state space overlap")
    else:
        dprint(f"    → LOW: Policies explore different state spaces")

    try:
        dprint(f"\nAction Agreement Conformance:")
        dprint(f"  - Action agreement: {agreement_ratio:.2%}")
        if agreement_ratio > 0.7:
            dprint(f"    → GOOD: Policies often choose same actions")
        elif agreement_ratio > 0.4:
            dprint(f"    → MEDIUM: Policies sometimes choose same actions")
        else:
            dprint(f"    → LOW: Policies have very different action distributions")

        # Diagnostic explanation for 100% agreement with low state overlap
        if agreement_ratio > 0.9 and overlap_ratio < 0.5:
            dprint(f"\n[IMPORTANT] High action agreement + low state overlap suggests:")
            dprint(f"  1. The state variables may be too coarse-grained")
            dprint(f"  2. Most actions visit the same few states (action bottleneck)")
            dprint(f"  3. The task assignment problem has many valid actions in each state")
            dprint(f"  4. Consider using finer-grained state buckets or activity-specific metrics")
    except:
        pass

    try:
        dprint(f"\nPerformance Conformance:")
        dprint(f"  - Trained policy reward: {trained_avg:.2f}")
        dprint(f"  - Random policy reward: {random_avg:.2f}")
        dprint(f"  - Improvement: {reward_ratio:.2f}x")
        if reward_ratio > 2.0:
            dprint(f"    → GOOD: Trained policy significantly outperforms random")
        elif reward_ratio > 1.2:
            dprint(f"    → FAIR: Trained policy moderately better than random")
        else:
            dprint(f"    → POOR: Trained policy not much better than random")

        # Diagnostic explanation for marginal improvement
        if reward_ratio < 1.5 and agreement_ratio > 0.7:
            dprint(f"\n[IMPORTANT] Marginal improvement despite high action agreement suggests:")
            dprint(f"  1. The training may have converged to a heuristic similar to random")
            dprint(f"  2. The reward signal may not be discriminative enough")
            dprint(f"  3. Need to increase training epochs or adjust hyperparameters")
            dprint(f"  4. Consider analyzing which states/actions differ between policies")
    except:
        pass

    dprint("\n" + "=" * 80)
    dprint("Analysis complete!")
    dprint("=" * 80 + "\n")

