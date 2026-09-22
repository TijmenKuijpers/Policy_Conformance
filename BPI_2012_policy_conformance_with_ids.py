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
import random
# torch not needed for heuristic-only comparison

# If gympn is not installed in your environment:
# 1) git clone https://github.com/bpogroup/gympn.git
# 2) add your local path below, e.g. sys.path.append("C:/path/to/gympn")
# sys.path.append("C:/path/to/gympn")

from gympn.solvers import HeuristicSolver
from gympn_problem_from_json_with_ids import load_parameters, create_task_assignment_problem, derive_problem_metadata
from gympn_problem_disjoint import create_disjoint_task_assignment_problem
from policy_conformance import PolicyConformance

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def dprint(msg):
    """Print with flushing"""
    print(msg, flush=True)


def make_spt_heuristic(parameters, valid_resources, time_scale,
                       min_resource_activities=1, max_resources=None,
                       max_activities=None, normalize=False):
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

    # Precompute per-activity average (mean) expected duration (in simulation time
    # units) across valid resources. Used when normalize=True to compare a
    # candidate's duration relative to the activity's typical duration.
    activity_avg = {}
    for act_name, rmap in activity_resource_mapping.items():
        vals = []
        for res, raw_dur in rmap.items():
            try:
                if res in valid_resources and raw_dur is not None and not (isinstance(raw_dur, float) and raw_dur != raw_dur):
                    vals.append(max(0.01, abs(raw_dur) / time_scale))
            except Exception:
                continue
        if vals:
            activity_avg[act_name] = float(sum(vals)) / len(vals)
        else:
            activity_avg[act_name] = 1.0

    def spt_heuristic(observable_net, tokens_comb):
        """
        Shortest Processing Time: pick the (activity, resource) binding
        whose expected service duration is the smallest.
        """
        # Collect all candidate bindings first, then choose the one with
        # the smallest estimated duration. This guarantees we inspect all
        # possible combinations across every action_name (no early return).
        candidates = []  # list of (est_duration, action_name, binding)

        for action_name, bindings_list in tokens_comb.items():
            for binding in bindings_list:
                # binding is typically a list of (place, token) tuples
                # but its shape can vary for disjoint/joint variants —
                # robustly find the task token (has 'activity_idx') and
                # the resource token (has 'resource_idx').
                task_token = None
                resource_token = None
                for place, token in binding:
                    val = token.value if hasattr(token, 'value') else token
                    if isinstance(val, dict):
                        if task_token is None and 'activity_idx' in val:
                            task_token = token
                        if resource_token is None and 'resource_idx' in val:
                            resource_token = token

                # fall back to positional assumption if not discovered
                if task_token is None:
                    try:
                        task_token = binding[0][1]
                    except Exception:
                        continue
                if resource_token is None:
                    try:
                        resource_token = binding[1][1]
                    except Exception:
                        # some actions may not include a resource (postpone etc.)
                        continue

                task_val = task_token.value if hasattr(task_token, 'value') else task_token
                res_val = resource_token.value if hasattr(resource_token, 'value') else resource_token

                try:
                    activity_idx = int(task_val.get('activity_idx', -1))
                except Exception:
                    activity_idx = -1
                try:
                    resource_idx = int(res_val.get('resource_idx', -1))
                except Exception:
                    resource_idx = -1

                act_name = idx_to_activity.get(activity_idx)
                if act_name is None or resource_idx < 0 or resource_idx >= len(valid_resources):
                    # skip bindings for which we cannot compute a duration
                    continue

                res_name = valid_resources[resource_idx]
                rmap = activity_resource_mapping.get(act_name, {})
                raw_dur = rmap.get(res_name)

                if raw_dur is None or (isinstance(raw_dur, float) and raw_dur != raw_dur):
                    dur = 1.0
                else:
                    dur = max(0.01, abs(raw_dur) / time_scale)

                # If normalization requested, compare relative to activity average
                if normalize:
                    avg = activity_avg.get(act_name, 1.0) or 1.0
                    metric = dur / avg
                else:
                    metric = dur

                # store metric first so selection uses it
                candidates.append((metric, dur, action_name, binding))

        if not candidates:
            # No valid duration candidates found; fall back to the first
            # available binding across action_names, or 'postpone' if none.
            for action_name, bindings_list in tokens_comb.items():
                if bindings_list:
                    return {action_name: bindings_list[0]}
            return 'postpone'

        # Choose the candidate with the smallest estimated duration
        # candidates tuples are (metric, dur, action_name, binding) when normalize
        best = min(candidates, key=lambda x: x[0])
        _, _, action_name, binding = best
        return {action_name: binding}

    return spt_heuristic


def make_fifo_heuristic(parameters, valid_resources, time_scale, tie_break='spt'):
    """FIFO heuristic with configurable tie-break among resources for the oldest case.
    Behavior:
    - find the oldest case among all available bindings (smallest case_id)
    - among bindings that belong to that case, either:
        * 'random' -> pick a random resource assignment among that case's candidates
        * 'spt'    -> pick the binding with the smallest estimated duration (SPT) among that case

    Parameters
    ----------
    tie_break : str
        One of 'random' or 'spt'. Default is 'random'.
    """
    activity_resource_mapping = parameters['activity_resource_mapping']
    activities = parameters['activities']
    idx_to_activity = {i: a for i, a in enumerate(activities)}

    def fifo(observable_net, tokens_comb):
        candidates = []  # (case_id_num, est_dur, action_name, binding)
        for action_name, bindings_list in tokens_comb.items():
            for binding in bindings_list:
                try:
                    task_token = binding[0][1]
                    resource_token = binding[1][1]
                except Exception:
                    continue
                task_val = task_token.value if hasattr(task_token, 'value') else task_token
                res_val = resource_token.value if hasattr(resource_token, 'value') else resource_token

                # Extract case identifier to derive chronological ordering (older -> smaller value).
                # Many event logs use non-numeric case IDs (e.g. 'Case 123'), which previously caused
                # float(case_id) to fail and set case_id_num==inf for all candidates. When that
                # happens, FIFO reduces to choosing the globally shortest expected duration (SPT),
                # making both heuristics behave identically. To avoid that, prefer an explicit
                # numeric 'counter' if present (arrival order), then try a numeric case_id or
                # extract digits from a case_id string, and finally try common timestamp fields.
                case_id = task_val.get('case_id')
                counter = task_val.get('counter')
                case_id_num = float('inf')

                # Prefer explicit arrival counter if available
                if counter is not None:
                    try:
                        case_id_num = float(counter)
                    except Exception:
                        pass

                # Next, try to interpret case_id as a number or extract digits from it
                if case_id_num == float('inf') and case_id is not None:
                    try:
                        case_id_num = float(case_id)
                    except Exception:
                        # try extracting the first sequence of digits
                        try:
                            import re
                            m = re.search(r"(\d+)", str(case_id))
                            if m:
                                case_id_num = float(m.group(1))
                        except Exception:
                            pass

                # Estimate duration for SPT tie-break
                activity_idx = int(task_val.get('activity_idx', -1))
                resource_idx = int(res_val.get('resource_idx', -1))
                act_name = idx_to_activity.get(activity_idx)
                est = 1.0
                if act_name is not None and 0 <= resource_idx < len(valid_resources):
                    res_name = valid_resources[resource_idx]
                    rmap = activity_resource_mapping.get(act_name, {})
                    raw = rmap.get(res_name)
                    if raw is None or (isinstance(raw, float) and raw != raw):
                        est = 1.0
                    else:
                        est = max(0.01, abs(raw) / time_scale)

                candidates.append((case_id_num, est, action_name, binding))

        if not candidates:
            return 'postpone'

        # Find oldest case (smallest case_id_num)
        min_case = min(c[0] for c in candidates)
        # Filter to candidates for that case
        filt = [c for c in candidates if c[0] == min_case]
        # Tie-break among candidates for the oldest case
        try:
            if tie_break == 'random':
                # preserve case-level FIFO but randomize resource selection
                import random as _rand
                chosen = _rand.choice(filt)
                _, _, action_name, binding = chosen
                return {action_name: binding}
            elif tie_break == 'spt':
                # choose the candidate with smallest estimated duration among the oldest case
                chosen = min(filt, key=lambda x: x[1])
                _, _, action_name, binding = chosen
                return {action_name: binding}
            else:
                # unknown tie_break: fallback to random
                import random as _rand
                chosen = _rand.choice(filt)
                _, _, action_name, binding = chosen
                return {action_name: binding}
        except Exception:
            # Fallback: pick the first
            _, _, action_name, binding = filt[0]
            return {action_name: binding}

    return fifo


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
        """Presence indicator for idle resources (reduced state):
        return 1 if there is at least one idle resource, else 0.

        This reduces the state-space dimensionality by converting the
        raw count into a binary indicator so downstream observations
        become boolean tuples (0/1) per slot.
        """
        return 1 if (resource_pool_queue and len(resource_pool_queue) > 0) else 0

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
            # function accepts both queues; PolicyConformance.evaluate will pass the corresponding markings
            pres_code = (
                f"def n_presence_{safe}({w_place}_queue, {b_place}_queue):\n"
                f"    # 1 if there is at least one token in waiting OR busy queue for this activity, else 0\n"
                f"    return 1 if (({w_place}_queue and len({w_place}_queue) > 0) or ({b_place}_queue and len({b_place}_queue) > 0)) else 0\n"
            )
            pres_ns = {}
            exec(pres_code, {}, pres_ns)
            state_vars.append(pres_ns[f"n_presence_{safe}"])

    return state_vars


def make_random_heuristic():
    """Return a heuristic that picks a random available binding uniformly."""
    def random_h(observable_net, tokens_comb):
        # Flatten all bindings into list of (action_name, binding)
        choices = []
        for action_name, bl in tokens_comb.items():
            for binding in bl:
                choices.append((action_name, binding))
        if not choices:
            return 'postpone'
        import random as _rand
        a, b = _rand.choice(choices)
        return {a: b}
    return random_h


def per_step_trace(problem, fifo_solver, spt_solver, conformance_obj, seed=0, sim_length=10, max_steps=500):
    """
    Run a paired, per-step trace of FIFO vs SPT on copies of `problem`.
    Prints available bindings, solver choices and immediate reward deltas
    for each step. Helpful to diagnose why policies visit different states
    but produce equal rewards.
    """
    import copy, random, numpy as np

    dprint("\n[TRACE] Starting per-step trace (seed=%s, sim_length=%s)" % (seed, sim_length))

    random.seed(seed)
    np.random.seed(seed)

    p_fifo = copy.deepcopy(problem)
    p_spt = copy.deepcopy(problem)

    p_fifo.set_solver(fifo_solver)
    p_spt.set_solver(spt_solver)
    p_fifo.length = sim_length
    p_spt.length = sim_length

    step_count = 0
    cum_fifo = 0.0
    cum_spt = 0.0

    def dump_aug(aug):
        for k, bl in aug.items():
            dprint(f"  action '{k}':")
            for b in bl:
                try:
                    task = b[0][1]
                    task_val = task.value if hasattr(task, 'value') else task
                    res = b[1][1] if len(b) > 1 else None
                    res_val = res.value if (res is not None and hasattr(res, 'value')) else res
                    dprint(f"    binding -> case_id={task_val.get('case_id')}, counter={task_val.get('counter')}, activity_idx={task_val.get('activity_idx')}, resource_idx={res_val.get('resource_idx') if res_val else None}")
                except Exception as e:
                    dprint(f"    binding -> (could not extract): {e}")

    # Step both problems until they finish or we hit max_steps
    while step_count < max_steps:
        step_count += 1

        # _get_solver_action expects a GymProblem-like state (with .bindings()),
        # so pass the GymProblem instances p_fifo / p_spt directly.
        act_key_fifo = conformance_obj._get_solver_action(p_fifo, fifo_solver)
        act_key_spt = conformance_obj._get_solver_action(p_spt, spt_solver)

        bindings_f, _ = p_fifo.bindings()
        bindings_s, _ = p_spt.bindings()
        aug_f = p_fifo._augment_bindings_with_postpone(bindings_f)
        aug_s = p_spt._augment_bindings_with_postpone(bindings_s)

        dprint(f"\n[TRACE] STEP {step_count}: clocks FIFO={p_fifo.clock:.3f} SPT={p_spt.clock:.3f}")
        dprint(" FIFO available bindings:")
        dump_aug(aug_f)
        dprint(" SPT available bindings:")
        dump_aug(aug_s)

        dprint(f" Solver choices: FIFO -> {act_key_fifo} ; SPT -> {act_key_spt}")

        prev_r_f = p_fifo.reward
        prev_r_s = p_spt.reward

        timed_binding_f, alive_f = p_fifo.step()
        timed_binding_s, alive_s = p_spt.step()

        delta_f = p_fifo.reward - prev_r_f
        delta_s = p_spt.reward - prev_r_s
        cum_fifo += delta_f
        cum_spt += delta_s

        # Show the executed binding keys (what actually fired) and whether the
        # solver-predicted best action matched the executed binding.
        try:
            fired_key_f = conformance_obj._binding_key(timed_binding_f) if timed_binding_f is not None else 'postpone'
        except Exception:
            fired_key_f = str(timed_binding_f)
        try:
            fired_key_s = conformance_obj._binding_key(timed_binding_s) if timed_binding_s is not None else 'postpone'
        except Exception:
            fired_key_s = str(timed_binding_s)

        dprint(f" After step: FIFO delta={delta_f:.3f} cum={cum_fifo:.3f} ; SPT delta={delta_s:.3f} cum={cum_spt:.3f}")
        dprint(f"  Predicted best action keys: FIFO_pred={act_key_fifo} ; SPT_pred={act_key_spt}")
        dprint(f"  Executed firing keys:      FIFO_fired={fired_key_f} ; SPT_fired={fired_key_s}")
        dprint(f"  Prediction==Fired? FIFO: {act_key_fifo == fired_key_f} ; SPT: {act_key_spt == fired_key_s}")

        # termination when both finished
        if (p_fifo.clock >= p_fifo.length and not alive_f) and (p_spt.clock >= p_spt.length and not alive_s):
            break

    dprint(f"\n[TRACE] Finished trace after {step_count} steps. Final rewards: FIFO={p_fifo.reward} SPT={p_spt.reward}\n")



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
    max_resources = 50          # Cap on total resources (None=no cap) — MUST match training!
    max_activities = None         # Cap on total activities (None=no cap) — MUST match training!
    remove_self_loops = True      # Remove self-loop transitions
    arrival_rate_factor = 0.1     # >1 = lighter load (fewer arrivals)

    variant = "disjoint" if disjoint_actions else "joint"
    dprint(f"\n[CONFIG] Problem variant: {'DISJOINT (one action per activity)' if disjoint_actions else 'JOINT (single assign action)'}")
    dprint(f"[CONFIG] Resource filter: min_activities={min_resource_activities}, max={max_resources}")
    dprint(f"[CONFIG] Activity filter: max_activities={max_activities}")
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

    # Step 2: Create baseline solvers (FIFO + SPT heuristic)
    dprint("\n[STEP 2] Creating baseline solvers (FIFO + SPT)...")

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
        normalize=True,
    )
    spt_solver = HeuristicSolver(heuristic_function=spt_function)
    # Now create FIFO using the derived metadata. You can configure how FIFO
    # breaks ties among resources for the oldest case: 'random' or 'spt'.
    fifo_tie_break = 'random'  # change to 'spt' to pick SPT among oldest-case candidates
    dprint(f"[CONFIG] FIFO tie-break strategy: {fifo_tie_break}")
    fifo_function = make_fifo_heuristic(
        parameters,
        valid_resources=meta['valid_resources'],
        time_scale=meta['time_scale'],
        tie_break=fifo_tie_break,
    )
    fifo_solver = HeuristicSolver(heuristic_function=fifo_function)
    # Random baseline solver
    random_function = make_random_heuristic()
    random_solver = HeuristicSolver(heuristic_function=random_function)
    dprint("[SUCCESS] FIFO heuristic solver created")
    dprint("[SUCCESS] Random heuristic solver created")
    dprint(f"[SUCCESS] SPT heuristic solver created "
           f"({len(meta['valid_resources'])} resources, "
           f"time_scale={meta['time_scale']:.0f}s)")

    # Step 3: Define state variables for conformance analysis
    # Determine which activities actually exist in the problem by checking place names
    dprint("\n[STEP 3] Defining state variables for analysis...")
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

    # Initialize metrics with safe defaults in case of earlier failure
    overlap_ratio = 0.0
    agreement_ratio = 0.0
    fifo_avg = 0.0
    fifo_std = 0.0
    spt_avg = 0.0
    spt_std = 0.0
    reward_ratio = 0.0

    # Token attributes that are irrelevant for conformance comparison:
    #   - case_id:      instance-specific identifier, doesn't affect the decision
    #   - counter:      arrival counter, bookkeeping only
    # resource_idx IS relevant — it represents the actual assignment choice.
    # NOTE: Do NOT exclude 'activity_idx' when using a joint-action problem,
    # because the activity identity is needed to distinguish actions.
    excluded_attrs = ["case_id", "counter"]

    # Step 4: Create PolicyConformance analyzer comparing FIFO vs SPT heuristics
    dprint("\n[STEP 4] Initializing PolicyConformance analyzer (FIFO vs SPT)...")
    dprint(f"[CONFIG] Excluded token attributes: {excluded_attrs}")
    try:
        conformance = PolicyConformance(
            gym_problem=problem,
            heuristic_solver=fifo_solver,
            gym_solver=spt_solver,
            state_variables=state_variables,
            excluded_token_attrs=excluded_attrs
        )
        dprint("[SUCCESS] PolicyConformance analyzer initialized")
    except Exception as e:
        dprint(f"[ERROR] Failed to initialize PolicyConformance: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Optional: run an interactive per-step trace to debug solver decisions
    enable_per_step_trace = False
    if enable_per_step_trace:
        try:
            per_step_trace(problem, fifo_solver, spt_solver, conformance_obj=conformance, seed=0, sim_length=20, max_steps=200)
        except Exception as e:
            dprint(f"[TRACE ERROR] per_step_trace failed: {e}")
            import traceback
            traceback.print_exc()

    # Step 6: Run action mapping analysis
    dprint("\n[STEP 6] Running action mapping analysis...")
    dprint("[INFO] Recording state-action pairs for both policies...")

    try:
        for episode in range(num_episodes):
            dprint(f"  Episode {episode + 1}/{num_episodes}...")
            conformance.compute_state_action_mapping(horizon=simulation_length)

        mapping_fifo = conformance.p1_state_action_mapping
        mapping_spt = conformance.p2_state_action_mapping

        p1_states = set(mapping_fifo.keys())
        p2_states = set(mapping_spt.keys())
        common_states = p1_states & p2_states

        dprint(f"[SUCCESS] Collected action mappings")
        dprint(f"  - FIFO (p1) action states: {len(p1_states)}; SPT (p2) action states: {len(p2_states)}")

        # State visit overlap (replaces old Step 6 diagnostics)
        fifo_freq_states = set(conformance.p1_state_visit_frequency.keys())
        spt_freq_states = set(conformance.p2_state_visit_frequency.keys())
        all_freq_states = fifo_freq_states | spt_freq_states
        overlapping_states = fifo_freq_states & spt_freq_states
        overlap_ratio = len(overlapping_states) / len(all_freq_states) if all_freq_states else 0

        dprint(f"\n[ANALYSIS] State Visit Frequency:")
        dprint(f"  - FIFO policy visited states: {len(fifo_freq_states)}")
        dprint(f"  - SPT policy visited states: {len(spt_freq_states)}")
        dprint(f"  - Overlapping states (FIFO vs SPT): {len(overlapping_states)}")
        dprint(f"  - State overlap ratio (FIFO vs SPT): {overlap_ratio:.2%}")

        dprint(f"\n[DEBUG] Action diversity analysis:")
        all_spt_actions = set(a for m in mapping_spt.values() for a in m.keys())
        all_fifo_actions = set(a for m in mapping_fifo.values() for a in m.keys())

        dprint(f"  - Unique actions (SPT policy): {len(all_spt_actions)}")
        dprint(f"  - Unique actions (FIFO policy): {len(all_fifo_actions)}")

        def top_actions(mapping, states):
            counts = {}
            for s in states:
                for a, c in mapping.get(s, {}).items():
                    counts[a] = counts.get(a, 0) + c
            return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]

        dprint(f"  - Most common SPT actions: {top_actions(mapping_spt, p2_states)}")
        dprint(f"  - Most common FIFO actions: {top_actions(mapping_fifo, p1_states)}")

        action_agreement = 0
        total_actions = 0
        disagreement_examples = []
        name_agreement = 0
        name_total = 0

        for state in common_states:
            snapshot = None
            if state in conformance.states_in_observation and conformance.states_in_observation[state]:
                snapshot = conformance.states_in_observation[state][0]
            if snapshot is None:
                continue

            act_spt = conformance._get_solver_action(snapshot, spt_solver)
            act_fifo = conformance._get_solver_action(snapshot, fifo_solver)

            a1 = act_spt if act_spt is not None else 'postpone'
            a2 = act_fifo if act_fifo is not None else 'postpone'

            if a1 == a2:
                action_agreement += 1
            else:
                if len(disagreement_examples) < 5:
                    disagreement_examples.append({'state': state, 'spt_action': a1, 'fifo_action': a2})
            total_actions += 1
            try:
                name1 = str(a1).split('|', 1)[0]
                name2 = str(a2).split('|', 1)[0]
            except Exception:
                name1 = a1
                name2 = a2
            if name1 == name2:
                name_agreement += 1
            name_total += 1

        agreement_ratio = action_agreement / total_actions if total_actions > 0 else 0
        dprint(f"\n[ANALYSIS] Action Agreement (queried on same snapshots):")
        dprint(f"  - Common states compared: {total_actions}")
        dprint(f"  - Common states with same best action: {action_agreement}/{total_actions}")
        dprint(f"  - Action agreement ratio: {agreement_ratio:.2%}")
        if name_total > 0:
            dprint(f"  - Action-name agreement (ignore tokens): {name_agreement}/{name_total} ({name_agreement/name_total:.2%})")

        if disagreement_examples:
            dprint(f"\n[DEBUG] Examples of action disagreement (queried snapshots):")
            for ex in disagreement_examples:
                st = ex['state']
                dprint(f"  State: {st} -> SPT: {ex['spt_action']} ; FIFO: {ex['fifo_action']}")
                dprint(f"    SPT actions: {mapping_spt.get(st)}")
                dprint(f"    FIFO actions: {mapping_fifo.get(st)}")
                state_snap = None
                if st in conformance.states_in_observation:
                    state_snap = conformance.states_in_observation[st][0]
                if state_snap is not None:
                    act_spt = conformance._get_solver_action(state_snap, spt_solver)
                    act_fifo = conformance._get_solver_action(state_snap, fifo_solver)
                    dprint(f"    Queried on same snapshot: SPT -> {act_spt} ; FIFO -> {act_fifo}")

        try:
            union_states = set(conformance.states_in_observation.keys())
            union_total = 0
            union_agree = 0
            union_disagree_examples = []
            weighted_total = 0.0
            weighted_agree = 0.0

            for st in union_states:
                snap = None
                if st in conformance.states_in_observation and conformance.states_in_observation[st]:
                    snap = conformance.states_in_observation[st][0]
                if snap is None:
                    continue

                act_spt_q = conformance._get_solver_action(snap, spt_solver)
                act_fifo_q = conformance._get_solver_action(snap, fifo_solver)
                a1 = act_spt_q if act_spt_q is not None else 'postpone'
                a2 = act_fifo_q if act_fifo_q is not None else 'postpone'
                union_total += 1
                if a1 == a2:
                    union_agree += 1

                vcount = (conformance.p1_state_visit_frequency.get(st, 0)
                          + conformance.p2_state_visit_frequency.get(st, 0))
                weighted_total += vcount
                if a1 == a2:
                    weighted_agree += vcount

                if a1 != a2 and len(union_disagree_examples) < 5:
                    union_disagree_examples.append({'state': st, 'spt': a1, 'fifo': a2})

            union_ratio = union_agree / union_total if union_total > 0 else 0
            weighted_ratio = weighted_agree / weighted_total if weighted_total > 0 else 0

            dprint(f"\n[DIAGNOSTIC] Global action agreement across UNION of observed states:")
            dprint(f"  - Union states compared: {union_total}")
            dprint(f"  - Union agreement: {union_agree}/{union_total} ({union_ratio:.2%})")
            dprint(f"  - Visit-weighted agreement (by state frequency): {weighted_ratio:.2%}")
            if union_disagree_examples:
                dprint("  - Example disagreements on union states:")
                for ex in union_disagree_examples:
                    dprint(f"    State: {ex['state']} -> SPT: {ex['spt']} ; FIFO: {ex['fifo']}")
        except Exception as e:
            dprint(f"[DIAGNOSTIC ERROR] failed to compute union agreement: {e}")

        if disagreement_examples:
            dprint(f"\n[DEBUG] Examples of action disagreement:")
            for i, example in enumerate(disagreement_examples, 1):
                dprint(f"  State {i}: {example['state']}")
                dprint(f"    - SPT: '{example['spt_action']}'")
                dprint(f"    - FIFO: '{example['fifo_action']}'")
        else:
            dprint(f"\n[DEBUG] All common states have same best action - states may be too coarse")

    except Exception as e:
        dprint(f"[ERROR] Failed during action mapping analysis: {e}")
        import traceback
        traceback.print_exc()

    # =====================================================================
    # ADDITIONAL OUTPUT: Produce per-observation policy tables and results
    # Matching the output structure in `choice_resources_gym.py`
    # =====================================================================
    try:
        dprint("\n[STEP X] Generating per-observation policy tables and conformance results...")

        # Accumulate mappings across multiple episodes so union of visited observations
        N_EPISODES = 5
        for episode in range(N_EPISODES):
            dprint(f"  Accumulation Episode {episode + 1}/{N_EPISODES}...")
            conformance.compute_state_action_mapping(horizon=simulation_length)

        # Extract visit frequencies and action mappings
        heuristic_o_visits = conformance.p1_state_visit_frequency
        gym_o_visits = conformance.p2_state_visit_frequency
        heuristic_o_action_mapping = conformance.p1_state_action_mapping
        gym_o_action_mapping = conformance.p2_state_action_mapping

        # Compute per-observation action probabilities
        heuristic_pa = conformance.compute_action_probability(heuristic_o_action_mapping)
        gym_pa = conformance.compute_action_probability(gym_o_action_mapping)

        # Compute Earth Mover's Action Conformance (EMAC)
        o_action_emd = conformance.compute_state_action_emd(heuristic_pa, gym_pa)

        # Expected reward rollouts (PERG) — using FIFO as p1 and SPT as p2
        expected_return_p1, expected_return_p2, delta_expected_return = conformance.expected_reward_run(
            fifo_solver, spt_solver, rho=0.95, eta=0, num_rollouts=5, num_steps=5)

        # Helper formatters and table builders (copied from choice_resources_gym)
        def fmt_obs(obs):
            return str(obs)

        def build_policy_table(visit_freq, action_prob, expected_return, all_actions, obs_labels):
            import pandas as _pd
            import numpy as _np
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
                er = expected_return.get(obs, float('nan'))
                row["Exp. Return"] = round(er, 4) if er == er else float('nan')
                rows.append(row)
            return _pd.DataFrame(rows)

        def short_action(key):
            base = str(key).split("|")[0]
            return "postpone" if base == "None" else base

        # Collect all action keys seen by either policy
        all_action_keys = sorted(
            {short_action(a)
             for obs_probs in list(heuristic_pa.values()) + list(gym_pa.values())
             for a in obs_probs}
        )

        def remap_probs(pa):
            remapped = {}
            for obs, probs in pa.items():
                remapped[obs] = {}
                for a, p in probs.items():
                    short = short_action(a)
                    remapped[obs][short] = remapped[obs].get(short, 0.0) + p
            return remapped

        heuristic_pa_short = remap_probs(heuristic_pa)
        gym_pa_short = remap_probs(gym_pa)

        import pandas as pd
        df_heuristic = build_policy_table(heuristic_o_visits, heuristic_pa_short,
                                          expected_return_p1,
                                          all_action_keys, fmt_obs)
        df_gym = build_policy_table(gym_o_visits, gym_pa_short,
                                    expected_return_p2,
                                    all_action_keys, fmt_obs)

        for label, df, file_suffix in [
            ("HEURISTIC POLICY", df_heuristic, "heuristic"),
            ("GYM (SPT) POLICY", df_gym, "gym"),
        ]:
            dprint(f"\n--- {label} ---")
            dprint(df.to_string(index=False))

            latex = df.to_latex(
                index=False,
                float_format="%.4f",
                na_rep="-",
                caption=f"Per-observation visit frequency and action probabilities — {label.lower()}.",
                label=f"tab:policy_{file_suffix}_bpi",
            )
            name = f"policy_table_{file_suffix}_bpi"
            df.to_csv(f"{name}.csv", index=False)
            with open(f"{name}.tex", "w") as f:
                f.write(latex)
            dprint(f"Saved: {name}.csv, {name}.tex")

        # RESULTS TABLE
        max_visit_ratios = conformance.compute_max_visit_ratio()
        all_obs = sorted(max_visit_ratios.keys())
        rows = []
        import math
        for obs in all_obs:
            emac = o_action_emd.get(obs, float('nan'))
            perg = delta_expected_return.get(obs, float('nan'))
            rows.append({
                "Observation": fmt_obs(obs),
                "$N_{ratio}$": round(max_visit_ratios[obs], 4),
                "$EMAC$": round(emac, 4) if emac == emac else float('nan'),
                "$PERG$": round(perg, 4) if perg == perg else float('nan'),
            })

        df_results = pd.DataFrame(rows)
        dprint("\n" + "=" * 80)
        dprint("CONFORMANCE RESULTS TABLE")
        dprint("=" * 80)
        dprint(df_results.to_string(index=False, na_rep="-"))

        latex_table = df_results.to_latex(
            index=False,
            float_format="%.4f",
            na_rep="-",
            caption="Per-observation policy conformance metrics.",
            label="tab:conformance_bpi",
        )
        name = "conformance_results_bpi"
        df_results.to_csv(f"{name}.csv", index=False)
        with open(f"{name}.tex", "w") as f:
            f.write(latex_table)
        dprint(f"\nSaved: {name}.csv, {name}.tex")

    except Exception as e:
        dprint(f"[ERROR] Failed to generate per-observation tables: {e}")
        import traceback
        traceback.print_exc()
    
    # =====================================================================
    # PLOTTING: generate heatmaps and metric bar charts similar to choice_experiment.py
    # =====================================================================
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import seaborn as sns

        dprint("\n[STEP Y] Generating plots (heatmaps + metrics)...")

        # Remap back to original (short) action probs if necessary
        pa_heur = heuristic_pa_short
        pa_gym = gym_pa_short

        # Build list of all observations seen by either policy
        all_obs = sorted(set(pa_heur.keys()) | set(pa_gym.keys()))
        if not all_obs:
            dprint("[PLOT] No observations to plot — skipping plots.")
        else:
            # Action ordering
            ACTIONS = all_action_keys
            ACTION_LABELS = ACTIONS

            def make_matrix(pa):
                mat = np.zeros((len(all_obs), len(ACTIONS)))
                for i, obs in enumerate(all_obs):
                    probs = pa.get(obs, {})
                    for j, act in enumerate(ACTIONS):
                        mat[i, j] = probs.get(act, 0.0)
                return mat

            mat_h = make_matrix(pa_heur)
            mat_g = make_matrix(pa_gym)

            # Heatmap: two side-by-side subplots (heuristic vs gym)
            fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(all_obs) * 0.35 + 1)), sharey=True)
            cmap = sns.color_palette("Blues", as_cmap=True)
            policy_data = [mat_h, mat_g]
            policy_labels = [r"FIFO (heuristic)", r"SPT (gym)"]

            obs_labels = [str(o) for o in all_obs]
            for ax, mat, plabel in zip(axes, policy_data, policy_labels):
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
                    # Do not show state identifiers on the row axis
                    yticklabels=[],
                    cbar=plabel.endswith('SPT'),
                    cbar_kws={"label": "Action Probability", "shrink": 0.8} if plabel.endswith('SPT') else {},
                    annot_kws={"size": 8},
                )
                ax.set_title(plabel, fontsize=12, fontweight="bold", pad=8)
                ax.set_xlabel("Action", fontsize=10)
                # Remove all ticks on the y-axis (no indices) and keep x labels minimal
                ax.set_yticks([])
                ax.tick_params(axis='x', labelsize=9, rotation=0)

            plt.tight_layout()
            plt.savefig("policy_heatmaps_bpi.png", dpi=150, bbox_inches="tight")
            dprint("Saved: policy_heatmaps_bpi.png")
            plt.close(fig)

            # --- Metric bar charts: visit ratio, EMAC, PERG ---
            # Visit ratios
            def merge_freq_dicts(a, b):
                out = {}
                for d in (a, b):
                    for k, v in d.items():
                        out[k] = out.get(k, 0) + v
                return out

            def visit_ratio(freq_dict):
                total = sum(freq_dict.values()) or 1
                return {obs: cnt / total for obs, cnt in freq_dict.items()}

            vr_h = visit_ratio(heuristic_o_visits)
            vr_g = visit_ratio(gym_o_visits)

            emac_hg = o_action_emd  # EMAC between heuristic and gym
            perg_hg = delta_expected_return

            metrics_obs = sorted(set(vr_h) | set(vr_g) | set(emac_hg) | set(perg_hg))
            x = np.arange(len(metrics_obs))
            w = 0.3
            x_lbl = [str(o) for o in metrics_obs]

            COLORS = {"FIFO": "#4C72B0", "SPT": "#DD8452", "FIFO vs SPT": "#55A868"}

            fig_w = max(4, len(metrics_obs) * 0.42 + 1)

            # Visit ratio figure
            fig_vr, ax_vr = plt.subplots(figsize=(fig_w, 4))
            vals_h = [vr_h.get(o, 0.0) for o in metrics_obs]
            vals_g = [vr_g.get(o, 0.0) for o in metrics_obs]
            # Place bars so they touch with no gap: widths 0.5 placed at x-0.5 and x
            ax_vr.bar(x - 0.5, vals_h, width=0.5, label='FIFO', color=COLORS['FIFO'], alpha=0.95, edgecolor='white')
            ax_vr.bar(x,         vals_g, width=0.5, label='SPT',  color=COLORS['SPT'],  alpha=0.95, edgecolor='white')
            ax_vr.set_ylabel("Visit ratio", fontsize=11)
            ax_vr.set_title("Visit ratio per observation", fontsize=12, fontweight="bold")
            ax_vr.legend(fontsize=9)
            ax_vr.set_ylim(0, None)
            ax_vr.yaxis.grid(True, linestyle='--', alpha=0.5)
            ax_vr.set_axisbelow(True)
            # Remove axis ticks/labels (keep axes empty as requested)
            ax_vr.set_xticks([])
            ax_vr.set_yticks([])
            fig_vr.tight_layout()
            fig_vr.savefig("metrics_visit_ratio_bpi.png", dpi=150, bbox_inches="tight")
            dprint("Saved: metrics_visit_ratio_bpi.png")
            plt.close(fig_vr)

            # EMAC figure
            fig_emac, ax_emac = plt.subplots(figsize=(fig_w, 4))
            vals_emac = [emac_hg.get(o, 0.0) for o in metrics_obs]
            # Single bar per observation — make bars adjacent by using width=1.0 at x-0.5
            ax_emac.bar(x - 0.5, vals_emac, width=1.0, color=COLORS['FIFO vs SPT'], alpha=0.95, edgecolor='white')
            ax_emac.set_ylabel("EMAC", fontsize=11)
            ax_emac.set_title("EMAC per observation", fontsize=12, fontweight='bold')
            ax_emac.set_ylim(0, 1)
            ax_emac.yaxis.grid(True, linestyle='--', alpha=0.5)
            ax_emac.set_axisbelow(True)
            # Remove axis ticks/labels
            ax_emac.set_xticks([])
            ax_emac.set_yticks([])
            fig_emac.tight_layout()
            fig_emac.savefig("metrics_emac_bpi.png", dpi=150, bbox_inches="tight")
            dprint("Saved: metrics_emac_bpi.png")
            plt.close(fig_emac)

            # PERG figure
            fig_perg, ax_perg = plt.subplots(figsize=(fig_w, 4))
            vals_perg = [perg_hg.get(o, 0.0) for o in metrics_obs]
            ax_perg.bar(x - 0.5, vals_perg, width=1.0, color=COLORS['FIFO vs SPT'], alpha=0.95, edgecolor='white')
            ax_perg.axhline(0, color='black', linewidth=0.8)
            ax_perg.set_ylabel("PERG", fontsize=11)
            ax_perg.set_title("PERG per observation", fontsize=12, fontweight='bold')
            ax_perg.yaxis.grid(True, linestyle='--', alpha=0.5)
            ax_perg.set_axisbelow(True)
            ax_perg.set_xticks([])
            ax_perg.set_yticks([])
            fig_perg.tight_layout()
            fig_perg.savefig("metrics_perg_bpi.png", dpi=150, bbox_inches="tight")
            dprint("Saved: metrics_perg_bpi.png")
            plt.close(fig_perg)

    except Exception as e:
        dprint(f"[PLOT ERROR] Failed to generate plots: {e}")
        import traceback
        traceback.print_exc()

    # Step 8: Run performance comparison
    dprint("\n[STEP 8] Running performance comparison...")

    try:
        fifo_rewards = []
        spt_rewards = []
        random_rewards = []

        performance_episodes = 10
        for episode in range(performance_episodes):
            seed = episode
            dprint(f"  Paired Episode {episode + 1}/{performance_episodes} (seed={seed})...")

            # Test FIFO policy on a fresh copy with fixed seed
            random.seed(seed)
            np.random.seed(seed)
            test_problem_fifo = copy.deepcopy(problem)
            test_problem_fifo.set_solver(fifo_solver)
            test_problem_fifo.length = simulation_length
            active = True
            while test_problem_fifo.clock <= test_problem_fifo.length and active:
                _, active = test_problem_fifo.step()
            fifo_rewards.append(test_problem_fifo.reward)
            dprint(f"    FIFO episode reward: {test_problem_fifo.reward}")

            # Test SPT policy on a fresh copy with the same seed
            random.seed(seed)
            np.random.seed(seed)
            test_problem_spt = copy.deepcopy(problem)
            test_problem_spt.set_solver(spt_solver)
            test_problem_spt.length = simulation_length
            active = True
            while test_problem_spt.clock <= test_problem_spt.length and active:
                _, active = test_problem_spt.step()
            spt_rewards.append(test_problem_spt.reward)
            dprint(f"    SPT episode reward: {test_problem_spt.reward}")

            # Test RANDOM policy on a fresh copy with the same seed
            random.seed(seed)
            np.random.seed(seed)
            test_problem_rand = copy.deepcopy(problem)
            test_problem_rand.set_solver(random_solver)
            test_problem_rand.length = simulation_length
            active = True
            while test_problem_rand.clock <= test_problem_rand.length and active:
                _, active = test_problem_rand.step()
            random_rewards.append(test_problem_rand.reward)
            dprint(f"    RANDOM episode reward: {test_problem_rand.reward}")

        fifo_avg = np.mean(fifo_rewards)
        fifo_std = np.std(fifo_rewards)
        spt_avg = np.mean(spt_rewards)
        spt_std = np.std(spt_rewards)
        rand_avg = np.mean(random_rewards)
        rand_std = np.std(random_rewards)
        reward_diff = spt_avg - fifo_avg
        reward_ratio = spt_avg / fifo_avg if fifo_avg > 0 else 0

        dprint(f"\n[ANALYSIS] Performance Comparison:")
        dprint(f"  - FIFO policy:  {fifo_avg:.2f} ± {fifo_std:.2f}")
        dprint(f"  - SPT policy:     {spt_avg:.2f} ± {spt_std:.2f}")
        dprint(f"  - RANDOM policy:  {rand_avg:.2f} ± {rand_std:.2f}")
        dprint(f"  - Reward difference (SPT - FIFO): {reward_diff:.2f}")
        dprint(f"  - Reward ratio (SPT/FIFO): {reward_ratio:.2f}x")

    except Exception as e:
        dprint(f"[ERROR] Failed during performance comparison: {e}")
        import traceback
        traceback.print_exc()

    # Step 9: Summary and conclusions
    dprint("\n" + "=" * 80)
    dprint("CONFORMANCE ANALYSIS SUMMARY")
    dprint("=" * 80)

    dprint(f"\nState Visit Conformance:")
    dprint(f"  - State overlap (FIFO vs SPT): {overlap_ratio:.2%}")

    try:
        dprint(f"\nAction Agreement Conformance:")
        dprint(f"  - Action agreement (SPT vs FIFO): {agreement_ratio:.2%}")
    except:
        pass

    try:
        dprint(f"\nPerformance Conformance:")
        dprint(f"  - FIFO policy reward:  {fifo_avg:.2f}")
        dprint(f"  - SPT policy reward:     {spt_avg:.2f}")
        dprint(f"  - Improvement (SPT vs FIFO): {reward_ratio:.2f}x")
        if reward_ratio > 1.2:
            dprint(f"    → GOOD: SPT policy outperforms FIFO")
        else:
            dprint(f"    → POOR: SPT not substantially better than FIFO")
    except:
        pass

    dprint("\n" + "=" * 80)
    dprint("Analysis complete!")
    dprint("=" * 80 + "\n")

