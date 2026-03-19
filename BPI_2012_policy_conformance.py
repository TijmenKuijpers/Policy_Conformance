"""
Policy Conformance Analysis: Trained BPI_2012 Policy vs Random Policy

This script compares the trained PPO policy for BPI Challenge 2012 task assignment
against a random baseline policy using policy conformance metrics:
- State visit frequency (how often are the same states visited?)
- Action overlap (how often do they choose the same action in the same state?)
- Expected reward gap (how much better/worse is one policy?)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import copy
import numpy as np
import torch

#sys.path.append("C:/Users/20183272/OneDrive - TU Eindhoven/Documents/GitHub/gympn")
sys.path.append("C:/Users/lobia/PycharmProjects/policy_comparison/Policy_Conformance/gympn")

from gympn.solvers import GymSolver, RandomSolver, HeuristicSolver
from gympn_problem_from_json import load_parameters, create_task_assignment_problem, derive_problem_metadata
from gympn_problem_disjoint import create_disjoint_task_assignment_problem
from policy_conformance import PolicyConformance

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def dprint(msg):
    """Print with flushing"""
    print(msg, flush=True)


def make_spt_heuristic(parameters, valid_resources, time_scale,
                       min_resource_activities=1, max_resources=None,
                       max_activities=None):
    """
    Build a Shortest Processing Time heuristic function compatible with
    ``HeuristicSolver(heuristic_function=...)``.

    The heuristic inspects every available binding, computes the expected
    processing time for the (activity, resource) pair, and returns the
    binding with the shortest duration.

    Parameters
    ----------
    parameters : dict
        The full simulation_parameters dict (needs activity_resource_mapping,
        activities, resources).
    valid_resources : list[str]
        Ordered list of resource names actually present in the problem.
        ``resource_idx`` in token values indexes into this list.
    time_scale : float
        Divisor applied during problem creation to convert raw durations
        (seconds) into simulation time units.  Must match the value used
        by ``create_task_assignment_problem``.

    Returns
    -------
    callable
        ``spt_heuristic(observable_net, tokens_comb)`` suitable for
        ``HeuristicSolver``.
    """
    activity_resource_mapping = parameters['activity_resource_mapping']
    activities = parameters['activities']

    # Build idx -> activity name lookup (must match gympn_problem_from_json)
    idx_to_activity = {i: a for i, a in enumerate(activities)}

    n_act = len(activities)
    n_res = len(valid_resources)

    def spt_heuristic(observable_net, tokens_comb):
        """
        Shortest Processing Time: pick the (activity, resource) binding
        whose expected service duration is the smallest.
        """
        best_key = None
        best_binding = None
        best_duration = float('inf')

        for action_name, bindings_list in tokens_comb.items():
            for binding in bindings_list:
                # binding is a list of (place, token) tuples
                # For the task assignment problem:
                #   binding[0] = (waiting_place, task_token)
                #   binding[1] = (resource_pool, resource_token)
                task_token = binding[0][1]  # SimToken
                resource_token = binding[1][1]  # SimToken

                task_val = task_token.value if hasattr(task_token, 'value') else task_token
                res_val = resource_token.value if hasattr(resource_token, 'value') else resource_token

                # Extract activity index from boolean flags is_activity_i
                activity_idx = None
                for i in range(n_act):
                    if task_val.get(f'is_activity_{i}', False):
                        activity_idx = i
                        break
                if activity_idx is None:
                    # If no activity flag found, skip this binding
                    continue

                # Extract resource index from boolean flags is_resource_j
                resource_idx = None
                for j in range(n_res):
                    if res_val.get(f'is_resource_{j}', False):
                        resource_idx = j
                        break
                if resource_idx is None or resource_idx < 0 or resource_idx >= len(valid_resources):
                    continue

                # Look up duration
                act_name = idx_to_activity.get(activity_idx)

                res_name = valid_resources[resource_idx]
                rmap = activity_resource_mapping.get(act_name, {})
                raw_dur = rmap.get(res_name)

                if raw_dur is None or (isinstance(raw_dur, float) and raw_dur != raw_dur):
                    dur = 1.0  # fallback, same as problem creation
                else:
                    dur = max(0.01, abs(raw_dur) / time_scale)

                if dur < best_duration:
                    best_duration = dur
                    best_key = action_name
                    best_binding = binding

        if best_key is not None:
            return {best_key: best_binding}

        # Fallback: return first available binding
        for action_name, bindings_list in tokens_comb.items():
            if bindings_list:
                return {action_name: bindings_list[0]}
        return 'postpone'

    return spt_heuristic


def create_state_variables(disjoint_actions=False, parameters=None, filtered_activities=None):
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

    :param filtered_activities: If provided, only create state vars for these activities.
        Should match the activities actually present in the problem (after max_activities filtering).
    """

    # ── Resource pool (always present) ──────────────────────────────────
    def n_available(resource_pool_queue):
        """Exact number of idle resources."""
        return len(resource_pool_queue) if resource_pool_queue else 0

    state_vars = [n_available]

    if parameters is not None:
        if filtered_activities is not None:
            mapped_activities = filtered_activities
        else:
            activity_resource_mapping = parameters.get('activity_resource_mapping', {})
            activities = parameters.get('activities', [])
            mapped_activities = [a for a in activities if a in activity_resource_mapping]

        for activity_name in mapped_activities:
            safe = activity_name.replace(" ", "_").replace(".", "_")

            # single presence indicator per activity: 1 if waiting or busy tokens exist, else 0
            w_place = f"waiting_{safe}"
            b_place = f"busy_{safe}"
            pres_code = (
                f"def n_presence_{safe}({w_place}_queue, {b_place}_queue):\n"
                f"    # 1 if there is at least one token in waiting OR busy queue for this activity, else 0\n"
                f"    return 1 if (({w_place}_queue and len({w_place}_queue) > 0) or ({b_place}_queue and len({b_place}_queue) > 0)) else 0\n"
            )
            pres_ns = {}
            exec(pres_code, {}, pres_ns)
            state_vars.append(pres_ns[f"n_presence_{safe}"])

    return state_vars


if __name__ == "__main__":

    dprint("=" * 80)
    dprint("POLICY CONFORMANCE ANALYSIS: BPI_2012 Trained Policy vs Random Policy")
    dprint("=" * 80)

    # Configuration
    disjoint_actions = False  # True: one action per activity type; False: single "assign" action
    simulation_length = 10  # Length of simulation runs — should match training length
    num_episodes = 2  # Number of episodes for conformance analysis

    # Problem simplification: MUST MATCH training script settings exactly!
    min_resource_activities = 2   # Only keep resources used in >= N activities
    max_resources = None          # Cap on total resources (None=no cap) — MUST match training!
    max_activities = None         # Cap on total activities (None=no cap) — MUST match training!
    remove_self_loops = True      # Remove self-loop transitions
    arrival_rate_factor = 2.0     # >1 = lighter load (fewer arrivals)

    variant = "disjoint" if disjoint_actions else "joint"
    model_save_dir = f"data/train/bpi_2012_ppo_{variant}"
    weights_path = os.path.join(model_save_dir, "best_policy.pth")

    dprint(f"\n[CONFIG] Problem variant: {'DISJOINT (one action per activity)' if disjoint_actions else 'JOINT (single assign action)'}")
    dprint(f"[CONFIG] Resource filter: min_activities={min_resource_activities}, max={max_resources}")
    dprint(f"[CONFIG] Activity filter: max_activities={max_activities}")
    dprint(f"[CONFIG] Model path: {weights_path}")
    dprint(f"[CONFIG] Simulation length: {simulation_length}")
    dprint(f"[CONFIG] Number of episodes: {num_episodes}")

    # Step 1: Load the BPI_2012 task assignment problem
    dprint("\n[STEP 1] Loading problem from JSON parameters...")
    parameters = load_parameters("simulation_parameters.json")
    if disjoint_actions:
        problem = create_disjoint_task_assignment_problem(
            parameters,
            min_resource_activities=min_resource_activities,
            max_resources=max_resources,
            max_activities=max_activities,
            remove_self_loops=remove_self_loops,
            arrival_rate_factor=arrival_rate_factor)
    else:
        problem = create_task_assignment_problem(
            parameters,
            min_resource_activities=min_resource_activities,
            max_resources=max_resources,
            max_activities=max_activities,
            remove_self_loops=remove_self_loops,
            arrival_rate_factor=arrival_rate_factor)
    dprint(f"[SUCCESS] Problem created")

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

    # Step 3: Create baseline solvers (Random + SPT heuristic)
    dprint("\n[STEP 3] Creating baseline solvers...")
    random_solver = RandomSolver()
    dprint("[SUCCESS] Random policy solver created")

    # Build SPT heuristic using derived problem metadata
    meta = derive_problem_metadata(
        parameters,
        min_resource_activities=min_resource_activities,
        max_resources=max_resources,
        max_activities=max_activities,
    )
    spt_function = make_spt_heuristic(
        parameters,
        valid_resources=meta['valid_resources'],
        time_scale=meta['time_scale'],
    )
    spt_solver = HeuristicSolver(heuristic_function=spt_function)
    dprint(f"[SUCCESS] SPT heuristic solver created "
           f"({len(meta['valid_resources'])} resources, "
           f"time_scale={meta['time_scale']:.0f}s)")

    # Step 4: Define state variables for conformance analysis
    # Determine which activities actually exist in the problem by checking place names
    dprint("\n[STEP 4] Defining state variables for analysis...")
    all_activities = parameters.get('activities', [])
    arm = parameters.get('activity_resource_mapping', {})
    # Re-derive filtered activities using the same logic as create_task_assignment_problem
    if max_activities is not None and max_activities < len(all_activities):
        from collections import deque
        tp = parameters.get('transition_probabilities', {})
        sp = parameters.get('start_activity_probabilities', {})
        reachable_score = {}
        queue = deque()
        for act, prob in sp.items():
            if prob > 0 and act in set(all_activities):
                queue.append((act, prob))
                reachable_score[act] = reachable_score.get(act, 0) + prob
        visited = set()
        while queue:
            act, score = queue.popleft()
            if act in visited:
                continue
            visited.add(act)
            if act in tp:
                for tgt, p in tp[act].items():
                    if p > 0 and tgt in set(all_activities):
                        reachable_score[tgt] = reachable_score.get(tgt, 0) + score * p
                        if tgt not in visited:
                            queue.append((tgt, score * p))
        ranked = sorted(reachable_score.keys(), key=lambda a: reachable_score[a], reverse=True)
        kept_act_set = set(ranked[:max_activities])
        filtered_acts = [a for a in all_activities if a in kept_act_set and a in arm]
    else:
        filtered_acts = [a for a in all_activities if a in arm]
    dprint(f"[DEBUG] Filtered activities for state vars: {[a for a in filtered_acts]}")

    state_variables = create_state_variables(
        disjoint_actions=disjoint_actions,
        parameters=parameters,
        filtered_activities=filtered_acts
    )
    dprint(f"[SUCCESS] Defined {len(state_variables)} state variables")

    # Token attributes that are irrelevant for conformance comparison:
    #   - case_id:      instance-specific identifier, doesn't affect the decision
    #   - counter:      arrival counter, bookkeeping only
    # Note: activities and resources are encoded as boolean flags (is_activity_i, is_resource_j)
    # resource flags ARE relevant — they represent the actual assignment choice.
    excluded_attrs = ["case_id", "counter"]

    # Step 5: Create PolicyConformance analyzer
    dprint("\n[STEP 5] Initializing PolicyConformance analyzer...")
    dprint(f"[CONFIG] Excluded token attributes: {excluded_attrs}")
    try:
        conformance = PolicyConformance(
            gym_problem=problem,
            heuristic_solver=random_solver,  # Using random as the "heuristic"
            gym_solver=trained_solver,        # Using trained policy as the "gym" policy
            state_variables=state_variables,
            excluded_token_attrs=excluded_attrs
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

        dprint(f"[INFO] Running {num_episodes} episodes with SPT policy...")
        spt_frequencies = {}
        for episode in range(num_episodes):
            dprint(f"  Episode {episode + 1}/{num_episodes}...")
            freq = conformance.frequency_run(spt_solver, simulation_length)
            for state, count in freq.items():
                spt_frequencies[state] = spt_frequencies.get(state, 0) + count

        dprint(f"[SUCCESS] SPT policy visited {len(spt_frequencies)} unique states")

        # Calculate state overlap (trained vs random)
        all_states = set(trained_frequencies.keys()) | set(random_frequencies.keys())
        overlapping_states = set(trained_frequencies.keys()) & set(random_frequencies.keys())
        overlap_ratio = len(overlapping_states) / len(all_states) if all_states else 0

        # Calculate state overlap (trained vs SPT)
        all_states_spt = set(trained_frequencies.keys()) | set(spt_frequencies.keys())
        overlapping_states_spt = set(trained_frequencies.keys()) & set(spt_frequencies.keys())
        overlap_ratio_spt = len(overlapping_states_spt) / len(all_states_spt) if all_states_spt else 0

        dprint(f"\n[ANALYSIS] State Visit Frequency:")
        dprint(f"  - Trained policy visited states: {len(trained_frequencies)}")
        dprint(f"  - Random policy visited states: {len(random_frequencies)}")
        dprint(f"  - SPT policy visited states: {len(spt_frequencies)}")
        dprint(f"  - Overlapping states (trained vs random): {len(overlapping_states)}")
        dprint(f"  - State overlap ratio (trained vs random): {overlap_ratio:.2%}")
        dprint(f"  - Overlapping states (trained vs SPT): {len(overlapping_states_spt)}")
        dprint(f"  - State overlap ratio (trained vs SPT): {overlap_ratio_spt:.2%}")

    except Exception as e:
        dprint(f"[ERROR] Failed during frequency analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 7: Run action mapping analysis
    dprint("\n[STEP 7] Running action mapping analysis...")
    dprint("[INFO] Recording state-action pairs for both policies...")

    try:
        # Reset conformance object for action analysis (trained vs random)
        conformance_action = PolicyConformance(
            gym_problem=problem,
            heuristic_solver=random_solver,
            gym_solver=trained_solver,
            state_variables=state_variables,
            excluded_token_attrs=excluded_attrs
        )

        # Run action episodes (trained vs random)
        for episode in range(num_episodes):
            dprint(f"  Episode {episode + 1}/{num_episodes} (trained vs random)...")
            conformance_action.action_run(trained_solver, random_solver, length=simulation_length)

        # Conformance object for trained vs SPT
        conformance_action_spt = PolicyConformance(
            gym_problem=problem,
            heuristic_solver=spt_solver,
            gym_solver=trained_solver,
            state_variables=state_variables,
            excluded_token_attrs=excluded_attrs
        )

        for episode in range(num_episodes):
            dprint(f"  Episode {episode + 1}/{num_episodes} (trained vs SPT)...")
            conformance_action_spt.action_run(trained_solver, spt_solver, length=simulation_length)

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

        # ── Trained vs SPT action agreement ───────────────────────────
        p1_states_spt = set(conformance_action_spt.p1_state_action_mapping.keys())
        p2_states_spt = set(conformance_action_spt.p2_state_action_mapping.keys())
        common_states_spt = p1_states_spt & p2_states_spt

        dprint(f"\n[SUCCESS] Analyzed action mappings (trained vs SPT)")
        dprint(f"  - Trained policy action states: {len(p1_states_spt)}")
        dprint(f"  - SPT policy action states: {len(p2_states_spt)}")
        dprint(f"  - Common states: {len(common_states_spt)}")

        action_agreement_spt = 0
        total_actions_spt = 0
        disagreement_examples_spt = []

        for state in common_states_spt:
            p1_actions = conformance_action_spt.p1_state_action_mapping[state]
            p2_actions = conformance_action_spt.p2_state_action_mapping[state]

            p1_best = max(p1_actions, key=p1_actions.get)
            p2_best = max(p2_actions, key=p2_actions.get)

            if p1_best == p2_best:
                action_agreement_spt += 1
            else:
                if len(disagreement_examples_spt) < 3:
                    disagreement_examples_spt.append({
                        'state': state,
                        'trained_action': p1_best,
                        'spt_action': p2_best,
                    })
            total_actions_spt += 1

        agreement_ratio_spt = action_agreement_spt / total_actions_spt if total_actions_spt > 0 else 0
        dprint(f"\n[ANALYSIS] Action Agreement (trained vs SPT):")
        dprint(f"  - Common states with same best action: {action_agreement_spt}/{total_actions_spt}")
        dprint(f"  - Action agreement ratio: {agreement_ratio_spt:.2%}")

        if disagreement_examples_spt:
            dprint(f"\n[DEBUG] Examples of trained vs SPT disagreement:")
            for i, ex in enumerate(disagreement_examples_spt, 1):
                dprint(f"  State {i}: {ex['state']}")
                dprint(f"    - Trained: '{ex['trained_action']}'")
                dprint(f"    - SPT:     '{ex['spt_action']}'")
        else:
            dprint(f"\n[DEBUG] Trained and SPT agree on all common states")

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
        spt_rewards = []

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

            # Test SPT policy
            test_problem_spt = copy.deepcopy(problem)
            test_problem_spt.set_solver(spt_solver)
            test_problem_spt.length = simulation_length
            active = True
            while test_problem_spt.clock <= test_problem_spt.length and active:
                _, active = test_problem_spt.step()
            spt_rewards.append(test_problem_spt.reward)

        trained_avg = np.mean(trained_rewards)
        trained_std = np.std(trained_rewards)
        random_avg = np.mean(random_rewards)
        random_std = np.std(random_rewards)
        spt_avg = np.mean(spt_rewards)
        spt_std = np.std(spt_rewards)
        reward_diff = trained_avg - random_avg
        reward_ratio = trained_avg / random_avg if random_avg > 0 else 0
        reward_ratio_spt = trained_avg / spt_avg if spt_avg > 0 else 0

        dprint(f"\n[ANALYSIS] Performance Comparison:")
        dprint(f"  - Trained policy: {trained_avg:.2f} ± {trained_std:.2f}")
        dprint(f"  - Random policy:  {random_avg:.2f} ± {random_std:.2f}")
        dprint(f"  - SPT policy:     {spt_avg:.2f} ± {spt_std:.2f}")
        dprint(f"  - Reward difference (trained - random): {reward_diff:.2f}")
        dprint(f"  - Reward ratio (trained/random): {reward_ratio:.2f}x")
        dprint(f"  - Reward ratio (trained/SPT): {reward_ratio_spt:.2f}x")

    except Exception as e:
        dprint(f"[ERROR] Failed during performance comparison: {e}")
        import traceback
        traceback.print_exc()

    # Step 9: Summary and conclusions
    dprint("\n" + "=" * 80)
    dprint("CONFORMANCE ANALYSIS SUMMARY")
    dprint("=" * 80)

    dprint(f"\nState Visit Conformance:")
    dprint(f"  - State overlap (trained vs random): {overlap_ratio:.2%}")
    dprint(f"  - State overlap (trained vs SPT):    {overlap_ratio_spt:.2%}")

    try:
        dprint(f"\nAction Agreement Conformance:")
        dprint(f"  - Action agreement (trained vs random): {agreement_ratio:.2%}")
        dprint(f"  - Action agreement (trained vs SPT):    {agreement_ratio_spt:.2%}")
    except:
        pass

    try:
        dprint(f"\nPerformance Conformance:")
        dprint(f"  - Trained policy reward: {trained_avg:.2f}")
        dprint(f"  - Random policy reward:  {random_avg:.2f}")
        dprint(f"  - SPT policy reward:     {spt_avg:.2f}")
        dprint(f"  - Improvement vs random: {reward_ratio:.2f}x")
        dprint(f"  - Improvement vs SPT:    {reward_ratio_spt:.2f}x")
        if reward_ratio > 2.0:
            dprint(f"    → GOOD: Trained policy significantly outperforms random")
        elif reward_ratio > 1.2:
            dprint(f"    → FAIR: Trained policy moderately better than random")
        else:
            dprint(f"    → POOR: Trained policy not much better than random")
    except:
        pass

    dprint("\n" + "=" * 80)
    dprint("Analysis complete!")
    dprint("=" * 80 + "\n")

