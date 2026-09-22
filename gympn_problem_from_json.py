"""
Create a GymPN task assignment problem from JSON parameters with SEQUENTIAL case flow.

Activities within a case happen sequentially according to transition_probabilities.
Each case starts with a start activity (from start_activity_probabilities), then
after each activity completes, the next activity is chosen probabilistically.

Structure per activity:
  waiting_{act} --[assign_{act} (ACTION)]--> busy_{act} --[complete_{act} (EVENT)]--> route_{act}

Routing event uses transition_probabilities to send the token to the next
activity's waiting place or to 'done' (case finished).

A single shared resource_pool is used across all activities.

ALL token attributes are NUMERIC (floats) for GymPN compatibility.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import numpy as np
import sys
# If gympn is not installed in your environment:
# 1) git clone https://github.com/bpogroup/gympn.git
# 2) add your local path below, e.g. sys.path.append("C:/path/to/gympn")
# sys.path.append("C:/path/to/gympn")

from simpn.simulator import SimToken
from gympn.simulator import GymProblem


def dprint(msg):
    print(msg, flush=True)


def load_parameters(json_file):
    dprint(f"[DEBUG] Loading parameters from {json_file}...")
    with open(json_file, 'r') as f:
        parameters = json.load(f)
    dprint(f"[DEBUG] Parameters loaded successfully")
    dprint(f"[DEBUG] - Activities: {len(parameters['activities'])}")
    dprint(f"[DEBUG] - Resources: {len(parameters['resources'])}")
    dprint(f"[DEBUG] - Activity-resource mappings: {len(parameters['activity_resource_mapping'])}")
    if 'start_activity_probabilities' not in parameters:
        dprint("[WARN] start_activity_probabilities not found in JSON. "
               "Re-run parameters_extraction.py to generate it.")
    if 'transition_probabilities' not in parameters:
        dprint("[WARN] transition_probabilities not found in JSON. "
               "Re-run parameters_extraction.py to generate it.")
    return parameters


def _safe(name):
    """Sanitise an activity name for use as a SimVar / transition identifier."""
    return name.replace(" ", "_").replace(".", "_")


def derive_problem_metadata(parameters, time_scale=None,
                            min_resource_activities=1,
                            max_resources=None,
                            max_activities=None):
    """
    Derive the ``valid_resources`` list and ``time_scale`` that
    ``create_task_assignment_problem`` would compute for the same arguments.

    This is useful for building heuristics (e.g. SPT) that need to map
    ``resource_idx`` back to resource names and know the duration scaling.

    Returns
    -------
    dict with keys:
        valid_resources : list[str]   – ordered resource names (idx → name)
        time_scale      : float       – seconds-per-time-unit divisor
        activities      : list[str]   – activities kept after filtering
    """
    from collections import Counter

    activities = list(parameters['activities'])
    resources = list(parameters['resources'])
    activity_resource_mapping = dict(parameters['activity_resource_mapping'])

    # ── Activity filtering (same BFS logic) ───────────────────────────
    if max_activities is not None and max_activities < len(activities):
        from collections import deque
        transition_probs = parameters.get('transition_probabilities', {})
        start_probs = parameters.get('start_activity_probabilities', {})
        if not start_probs:
            start_probs = {a: 1.0 / len(activities) for a in activities}
        total = sum(start_probs.values())
        start_probs = {a: p / total for a, p in start_probs.items()}

        reachable_score = {}
        queue = deque()
        for act, prob in start_probs.items():
            if prob > 0 and act in set(activities):
                queue.append((act, prob))
                reachable_score[act] = reachable_score.get(act, 0) + prob
        visited = set()
        while queue:
            act, score = queue.popleft()
            if act in visited:
                continue
            visited.add(act)
            if act in transition_probs:
                for tgt, p in transition_probs[act].items():
                    if p > 0 and tgt in set(activities):
                        reachable_score[tgt] = reachable_score.get(tgt, 0) + score * p
                        if tgt not in visited:
                            queue.append((tgt, score * p))
        ranked = sorted(reachable_score.keys(),
                        key=lambda a: reachable_score[a], reverse=True)
        kept_activities = set(ranked[:max_activities])
        activities = [a for a in activities if a in kept_activities]
        activity_resource_mapping = {
            act: rmap for act, rmap in activity_resource_mapping.items()
            if act in kept_activities
        }

    # ── Resource filtering ────────────────────────────────────────────
    valid_resources = [r for r in resources
                       if r is not None and not (isinstance(r, float) and np.isnan(r))]
    resource_to_idx = {r: i for i, r in enumerate(valid_resources)}

    resource_act_count = Counter()
    for act_name, rmap in activity_resource_mapping.items():
        for res in rmap:
            if res in resource_to_idx:
                resource_act_count[res] += 1

    kept_resources = [r for r in valid_resources
                      if resource_act_count.get(r, 0) >= min_resource_activities]
    if max_resources is not None and len(kept_resources) > max_resources:
        kept_resources = sorted(kept_resources,
                                key=lambda r: resource_act_count.get(r, 0),
                                reverse=True)[:max_resources]
    valid_resources = kept_resources
    resource_set = set(valid_resources)

    # Prune mapping to kept resources
    activity_resource_mapping = {
        act: {res: dur for res, dur in rmap.items() if res in resource_set}
        for act, rmap in activity_resource_mapping.items()
    }
    activity_resource_mapping = {
        act: rmap for act, rmap in activity_resource_mapping.items() if rmap
    }

    # ── Time scale ────────────────────────────────────────────────────
    all_durations_sec = []
    for act_name, rmap in activity_resource_mapping.items():
        for res, dur in rmap.items():
            if dur is not None and not (isinstance(dur, float) and np.isnan(dur)):
                all_durations_sec.append(abs(dur))

    if time_scale is None and all_durations_sec:
        time_scale = float(np.median(all_durations_sec))
        if time_scale <= 0:
            time_scale = 1.0
    elif time_scale is None:
        time_scale = 1.0

    return {
        'valid_resources': valid_resources,
        'time_scale': time_scale,
        'activities': activities,
    }


def create_task_assignment_problem(parameters, time_scale=None,
                                    min_resource_activities=1,
                                    max_resources=None,
                                    max_activities=None,
                                    remove_self_loops=True,
                                    arrival_rate_factor=None):
    """
    Create a GymPN task assignment problem with sequential case flow.

    :param parameters: dict loaded from simulation_parameters.json
    :param time_scale: Divide all durations by this value. If None, auto-compute
        so median activity duration = 1 time unit.
    :param min_resource_activities: Only keep resources appearing in >= N activities.
    :param max_resources: Cap on total resources (most-used first). None = no cap.
    :param max_activities: Cap on total activities (most reachable from start first).
        None = no cap.  Reducing activities dramatically shrinks the Petri net.
    :param remove_self_loops: If True, remove self-loop transitions and renormalize.
        This prevents cases from looping on the same activity dozens of times.
    :param arrival_rate_factor: Multiply the scaled inter-arrival time by this factor.
        If None, auto-compute so that arrival rate ≈ service capacity (n_resources * avg_dur).
        Use values > 1 for lighter load, < 1 for heavier load.
    """
    dprint("[DEBUG] Creating SEQUENTIAL task assignment problem...")

    # ── Extract parameters ───────────────────────────────────────────────
    activities = parameters['activities']
    resources = parameters['resources']
    activity_resource_mapping = parameters['activity_resource_mapping']
    avg_inter_arrival_time = parameters.get('avg_inter_arrival_time', 5.0)
    transition_probs = parameters.get('transition_probabilities', {})
    start_probs = parameters.get('start_activity_probabilities', {})

    # Fallback: if no start probs, assume uniform over all activities
    if not start_probs:
        dprint("[WARN] No start_activity_probabilities found, using uniform")
        start_probs = {a: 1.0 / len(activities) for a in activities}

    # Normalise start probs (safety)
    total = sum(start_probs.values())
    start_probs = {a: p / total for a, p in start_probs.items()}

    # ── Filter activities by reachability (BFS from start) ───────────────
    if max_activities is not None and max_activities < len(activities):
        # BFS/rank activities by reachability from start activities
        from collections import deque
        reachable_score = {}  # activity -> cumulative probability of being reached
        queue = deque()
        for act, prob in start_probs.items():
            if prob > 0 and act in set(activities):
                queue.append((act, prob))
                reachable_score[act] = reachable_score.get(act, 0) + prob
        visited = set()
        while queue:
            act, score = queue.popleft()
            if act in visited:
                continue
            visited.add(act)
            if act in transition_probs:
                for tgt, p in transition_probs[act].items():
                    if p > 0 and tgt in set(activities):
                        new_score = score * p
                        reachable_score[tgt] = reachable_score.get(tgt, 0) + new_score
                        if tgt not in visited:
                            queue.append((tgt, new_score))

        # Keep top max_activities by reachability score
        ranked = sorted(reachable_score.keys(),
                        key=lambda a: reachable_score[a], reverse=True)
        kept_activities = set(ranked[:max_activities])
        activities = [a for a in activities if a in kept_activities]

        # Re-filter start_probs to kept activities
        start_probs = {a: p for a, p in start_probs.items() if a in kept_activities}
        if not start_probs:
            # If no start activities remain, use uniform
            start_probs = {a: 1.0 / len(activities) for a in activities}
        total = sum(start_probs.values())
        start_probs = {a: p / total for a, p in start_probs.items()}

        # Re-filter transition_probs
        transition_probs = {
            src: {tgt: p for tgt, p in targets.items() if tgt in kept_activities}
            for src, targets in transition_probs.items()
            if src in kept_activities
        }
        # Re-filter activity_resource_mapping
        activity_resource_mapping = {
            act: rmap for act, rmap in activity_resource_mapping.items()
            if act in kept_activities
        }

        dprint(f"[DEBUG] Activity filtering: kept {len(activities)} of "
               f"{len(parameters['activities'])} (max_activities={max_activities})")
        for a in activities:
            dprint(f"[DEBUG]   - {a} (reachability={reachable_score.get(a, 0):.4f})")

    # Numeric mappings
    activity_to_idx = {a: i for i, a in enumerate(activities)}
    idx_to_activity = {i: a for a, i in activity_to_idx.items()}

    valid_resources = [r for r in resources
                       if r is not None and not (isinstance(r, float) and np.isnan(r))]
    resource_to_idx = {r: i for i, r in enumerate(valid_resources)}

    # ── Filter resources by usage frequency ──────────────────────────────
    from collections import Counter
    resource_act_count = Counter()
    for act_name, rmap in activity_resource_mapping.items():
        for res in rmap:
            if res in resource_to_idx:
                resource_act_count[res] += 1

    kept_resources = [r for r in valid_resources
                      if resource_act_count.get(r, 0) >= min_resource_activities]
    if max_resources is not None and len(kept_resources) > max_resources:
        kept_resources = sorted(kept_resources,
                                key=lambda r: resource_act_count.get(r, 0),
                                reverse=True)[:max_resources]

    if len(kept_resources) < len(valid_resources):
        dprint(f"[DEBUG] Resource filtering: {len(valid_resources)} -> {len(kept_resources)} "
               f"(min_act={min_resource_activities}, max={max_resources})")
    valid_resources = kept_resources
    resource_to_idx = {r: i for i, r in enumerate(valid_resources)}
    resource_set = set(valid_resources)

    # Prune activity_resource_mapping to kept resources only
    activity_resource_mapping = {
        act: {res: dur for res, dur in rmap.items() if res in resource_set}
        for act, rmap in activity_resource_mapping.items()
    }
    activity_resource_mapping = {
        act: rmap for act, rmap in activity_resource_mapping.items() if rmap
    }

    # Only keep activities that have resource mappings (can actually be assigned)
    mapped_activities = [a for a in activities if a in activity_resource_mapping]
    dprint(f"[DEBUG] - {len(activities)} activities, {len(valid_resources)} resources")
    dprint(f"[DEBUG] - {len(mapped_activities)} mapped activities")

    # ── Compute time scale ───────────────────────────────────────────────
    all_durations_sec = []
    for act_name, rmap in activity_resource_mapping.items():
        for res, dur in rmap.items():
            if dur is not None and not (isinstance(dur, float) and np.isnan(dur)):
                all_durations_sec.append(abs(dur))

    if time_scale is None and all_durations_sec:
        time_scale = float(np.median(all_durations_sec))
        if time_scale <= 0:
            time_scale = 1.0
    elif time_scale is None:
        time_scale = 1.0

    scaled_inter_arrival = max(0.01, (avg_inter_arrival_time * 3600) / time_scale)
    dprint(f"[DEBUG] Time scale: {time_scale:.0f}s (median dur), "
           f"raw inter-arrival={scaled_inter_arrival:.3f} time units")

    # ── Build the transition matrix as numpy array for fast sampling ─────
    # transition_matrix[i] is the probability distribution over next activities
    # An extra column (index len(activities)) represents "case done" (no successor)
    n_act = len(activities)
    mapped_set = set(mapped_activities)
    transition_matrix = np.zeros((n_act, n_act + 1))  # +1 for "done"
    for src_act, targets in transition_probs.items():
        if src_act not in activity_to_idx:
            continue
        i = activity_to_idx[src_act]
        row_sum = 0.0
        for tgt_act, prob in targets.items():
            if tgt_act in activity_to_idx and tgt_act in mapped_set:
                # Only route to activities that have resource mappings
                j = activity_to_idx[tgt_act]
                transition_matrix[i, j] = prob
                row_sum += prob
            else:
                # Probability to unmapped activities counts as "done"
                transition_matrix[i, n_act] += prob
        # Remaining probability mass (from rounding, etc.) goes to "done"
        transition_matrix[i, n_act] += max(0.0, 1.0 - row_sum - transition_matrix[i, n_act])
    # Activities without transition data: 100% done
    for i in range(n_act):
        if transition_matrix[i].sum() == 0:
            transition_matrix[i, n_act] = 1.0
    # Safety: renormalise each row
    for i in range(n_act):
        s = transition_matrix[i].sum()
        if s > 0:
            transition_matrix[i] /= s

    # ── Remove self-loops from transition matrix ─────────────────────────
    if remove_self_loops:
        for i in range(n_act):
            self_prob = transition_matrix[i, i]
            if self_prob > 0:
                act_name = idx_to_activity.get(i, f"act_{i}")
                dprint(f"[DEBUG] Removing self-loop for {act_name}: {self_prob:.4f}")
                transition_matrix[i, i] = 0.0
                # Renormalise remaining probabilities
                remaining = transition_matrix[i].sum()
                if remaining > 0:
                    transition_matrix[i] /= remaining
                else:
                    # If only self-loop existed, go to "done"
                    transition_matrix[i, n_act] = 1.0

    dprint("[DEBUG] Transition matrix built")
    # Diagnostic: show transition stats for key activities
    for act in mapped_activities:
        i = activity_to_idx[act]
        successors = [(idx_to_activity[j], transition_matrix[i, j])
                      for j in range(n_act) if transition_matrix[i, j] > 0.01]
        done_prob = transition_matrix[i, n_act]
        dprint(f"[DEBUG]   {act}: -> done={done_prob:.2f}, successors={successors[:3]}")

    # ── Compute expected case length (avg activities per case) ───────────
    # Solve for expected visits: E = (I - T)^{-1} * 1
    # where T is the activity-to-activity sub-matrix (excluding "done" column)
    T_sub = transition_matrix[:n_act, :n_act]
    avg_activities_per_case = None
    try:
        # Check that each row of T_sub sums to strictly less than 1
        # (i.e. every activity has a nonzero probability of eventually finishing).
        row_sums = T_sub.sum(axis=1)
        max_row_sum = row_sums.max()
        if max_row_sum >= 1.0 - 1e-9:
            dprint(f"[WARN] Transition sub-matrix has absorbing cycles "
                   f"(max row sum = {max_row_sum:.6f}). Using simulation estimate.")
        else:
            expected_visits = np.linalg.solve(
                np.eye(n_act) - T_sub,
                np.ones(n_act)
            )
            # Validate: all expected visits should be positive and finite
            if np.all(expected_visits > 0) and np.all(np.isfinite(expected_visits)):
                start_vec = np.zeros(n_act)
                for act, prob in start_probs.items():
                    if act in activity_to_idx:
                        start_vec[activity_to_idx[act]] = prob
                avg_activities_per_case = float(start_vec @ expected_visits)
                if avg_activities_per_case <= 0 or avg_activities_per_case > 1000:
                    dprint(f"[WARN] Computed avg_activities_per_case={avg_activities_per_case:.1f} "
                           f"seems unreasonable. Using simulation estimate.")
                    avg_activities_per_case = None
                else:
                    dprint(f"[DEBUG] Expected activities per case: {avg_activities_per_case:.1f}")
            else:
                dprint("[WARN] Expected visits has invalid values. Using simulation estimate.")
    except np.linalg.LinAlgError:
        dprint("[WARN] Singular matrix in expected case length. Using simulation estimate.")

    # Fallback: Monte Carlo estimate of expected case length
    if avg_activities_per_case is None:
        dprint("[DEBUG] Estimating avg activities per case via Monte Carlo (1000 samples)...")
        total_steps = 0
        n_samples = 1000
        max_steps_per_sample = 200  # safety cap per simulated case
        for _ in range(n_samples):
            # Pick start activity
            acts_list = list(start_probs.keys())
            probs_list = [start_probs[a] for a in acts_list]
            current = np.random.choice(acts_list, p=probs_list)
            steps = 1
            while steps < max_steps_per_sample:
                if current not in activity_to_idx:
                    break
                idx = activity_to_idx[current]
                row = transition_matrix[idx]
                next_idx = int(np.random.choice(len(row), p=row))
                if next_idx >= n_act:
                    break  # case done
                current = idx_to_activity.get(next_idx)
                if current is None or current not in mapped_set:
                    break
                steps += 1
            total_steps += steps
        avg_activities_per_case = total_steps / n_samples
        dprint(f"[DEBUG] Monte Carlo estimate: {avg_activities_per_case:.1f} activities per case")

    # ── Auto-compute inter-arrival rate ──────────────────────────────────
    # Goal: arrival rate should be sustainable given the number of resources.
    # A case needs ~avg_activities_per_case resource-time-units to complete.
    # Service capacity = n_resources (each can do 1 activity per time unit on average).
    # Sustainable arrival rate = n_resources / avg_activities_per_case
    # Inter-arrival time = 1 / arrival_rate = avg_activities_per_case / n_resources
    n_resources = len(valid_resources)
    sustainable_iat = max(0.01, avg_activities_per_case / n_resources)
    if arrival_rate_factor is not None:
        scaled_inter_arrival = sustainable_iat * arrival_rate_factor
    else:
        # Auto: aim for ~67% utilization
        scaled_inter_arrival = sustainable_iat * 1.5

    # Safety: ensure inter-arrival is always positive and reasonable
    scaled_inter_arrival = max(0.01, scaled_inter_arrival)

    dprint(f"[DEBUG] Adjusted inter-arrival: {scaled_inter_arrival:.3f} time units "
           f"(sustainable={sustainable_iat:.3f}, "
           f"n_resources={n_resources}, avg_acts/case={avg_activities_per_case:.1f})")

    # ── Create the GymProblem ────────────────────────────────────────────
    problem = GymProblem(causal_rl=False)

    # Shared resource pool represented as one-hot resource flags
    n_res = len(valid_resources)
    resource_flag_names = [f"is_resource_{i}" for i in range(n_res)]
    resource_pool = problem.add_var("resource_pool", var_attributes=resource_flag_names)
    for idx in range(n_res):
        flags = {f"is_resource_{j}": (1.0 if (j == idx) else 0.0) for j in range(n_res)}
        resource_pool.put(flags)
    dprint(f"[DEBUG] - Placed {n_res} resource tokens")

    # (case_id removed - not used)

    # ── Per-activity places ──────────────────────────────────────────────
    # waiting_{act}: tasks waiting to be assigned for this activity
    # busy_{act}:    tasks being executed
    # routed_{act}:  tasks that completed and await routing to next activity
    waiting_places = {}
    busy_places = {}
    routed_places = {}

    # Replace numeric activity index with one-hot boolean flags per activity
    # plus an `is_done` flag to indicate terminal/done tokens.
    activity_flag_names = [f"is_activity_{i}" for i in range(n_act)]
    token_attrs_waiting = activity_flag_names + ['is_done']
    token_attrs_busy = activity_flag_names + ['is_done'] + resource_flag_names
    token_attrs_routed = activity_flag_names + ['is_done']

    for act in mapped_activities:
        s = _safe(act)
        waiting_places[act] = problem.add_var(f"waiting_{s}", var_attributes=token_attrs_waiting)
        busy_places[act] = problem.add_var(f"busy_{s}", var_attributes=token_attrs_busy)
        routed_places[act] = problem.add_var(f"routed_{s}", var_attributes=token_attrs_routed)

    dprint(f"[DEBUG] Created places for {len(mapped_activities)} activities")

    # ── Arrival event ────────────────────────────────────────────────────
    arrival = problem.add_var("arrival", var_attributes=['counter'])
    arrival.put({'counter': 0.0})

    # We need an arrival output that can fan out to any start activity's
    # waiting place.  Since the start activity is probabilistic, we route
    # through a single "new_case" place, then a routing event distributes.
    new_case = problem.add_var("new_case", var_attributes=activity_flag_names + ['is_done'])

    def _make_activity_flags(idx):
        """Return a dict of numeric activity flags (1.0/0.0) with only idx True and is_done 0.0."""
        flags = {f"is_activity_{j}": (1.0 if (j == idx) else 0.0) for j in range(n_act)}
        flags['is_done'] = 0.0
        return flags

    def arrive(arrival_token):
        # Pick start activity
        acts = list(start_probs.keys())
        probs = [start_probs[a] for a in acts]
        start_act = np.random.choice(acts, p=probs)
        a_idx = activity_to_idx[start_act]
        delay = max(0.01, np.random.exponential(scaled_inter_arrival))
        return [
            SimToken({**{f"is_activity_{j}": (1.0 if (j == a_idx) else 0.0) for j in range(n_act)}, 'is_done': 0.0}),   # new case token (no case_id)
            SimToken(arrival_token, delay=delay),                  # re-trigger
        ]

    problem.add_event([arrival], [new_case, arrival], arrive, name="arrive")
    dprint("[DEBUG] Arrival event added")

    # ── Route new_case to the correct waiting place ──────────────────────
    # One routing event per mapped activity with a guard on activity boolean flags.
    for act in mapped_activities:
        a_idx = activity_to_idx[act]
        s = _safe(act)

        def make_guard_new(target):
            def guard(tok):
                val = tok.value if hasattr(tok, 'value') else tok
                return bool(val.get(f"is_activity_{target}", 0))
            return guard

        def make_route_new():
            def route(tok):
                val = tok.value if hasattr(tok, 'value') else tok
                return [SimToken(val)]
            return route

        problem.add_event(
            [new_case], [waiting_places[act]],
            make_route_new(),
            guard=make_guard_new(a_idx),
            name=f"route_new_{s}"
        )

    dprint("[DEBUG] New-case routing events added")

    # ── Per-activity: assign action + complete event + routing event ─────
    # "done" place: cases that finished all activities (tokens carry activity flags)
    done = problem.add_var("done", var_attributes=activity_flag_names + ['is_done'])

    # Shared intermediate routing place: tokens here carry activity boolean flags
    routed_next = problem.add_var("routed_next", var_attributes=activity_flag_names + ['is_done'])

    for act in mapped_activities:
        a_idx = activity_to_idx[act]
        s = _safe(act)
        res_mapping = activity_resource_mapping[act]

        # ── ASSIGN action (RL decision) ──────────────────────────────
        def make_assign(act_name, act_idx, rmap, tscale):
            def assign(task, resource):
                task_val = task.value if hasattr(task, 'value') else task
                resource_val = resource.value if hasattr(resource, 'value') else resource
                # resource_val is a dict of is_resource_i flags; find the active one
                res_idx = None
                for j in range(n_res):
                    if resource_val.get(f"is_resource_{j}", 0):
                        res_idx = j
                        break
                if res_idx is None:
                    # fallback: pick first resource
                    res_idx = 0
                res_name = valid_resources[res_idx]
                if res_name in rmap:
                    raw_sec = rmap[res_name]
                    if raw_sec is None or (isinstance(raw_sec, float) and np.isnan(raw_sec)):
                        dur = 1.0
                    else:
                        dur = max(0.01, abs(raw_sec) / tscale)
                else:
                    dur = 1.0
                # Build numeric flags with only this activity True
                flags = {f"is_activity_{j}": (1.0 if (j == act_idx) else 0.0) for j in range(n_act)}
                flags['is_done'] = 0.0
                # attach numeric resource flags to the busy token
                res_flags = {f"is_resource_{j}": (1.0 if (j == res_idx) else 0.0) for j in range(n_res)}
                return [SimToken({**flags, **res_flags}, delay=dur)]
            return assign

        problem.add_action(
            [waiting_places[act], resource_pool],
            [busy_places[act]],
            behavior=make_assign(act, a_idx, res_mapping, time_scale),
            name=f"assign_{s}"
        )

        # ── COMPLETE event: free resource, move token to routed place ─
        def make_complete():
            def complete(tok):
                tok_val = tok.value if hasattr(tok, 'value') else tok
                # Return resource flags back to pool (preserve which resource)
                res_flags = {f"is_resource_{j}": (1.0 if tok_val.get(f"is_resource_{j}", 0) else 0.0) for j in range(n_res)}
                resource_token = SimToken(res_flags)
                # Extract activity flags and is_done from the busy token (numeric)
                flags = {f"is_activity_{j}": (1.0 if tok_val.get(f"is_activity_{j}", 0) else 0.0) for j in range(n_act)}
                flags['is_done'] = (1.0 if tok_val.get('is_done', 0) else 0.0)
                routed_token = SimToken(flags)
                return [resource_token, routed_token]
            return complete

        problem.add_event(
            [busy_places[act]], [resource_pool, routed_places[act]],
            make_complete(),
            name=f"complete_{s}"
        )

        # ── ROUTING event: use transition_probs to pick next activity ─
        # Instead of one fan-out event with N+1 outputs (mostly None), we use
        # a two-step approach: (1) a "sample" event that picks the next activity
        # and writes it to `routed_next`, (2) guarded dispatch events (1-to-1).
        src_idx = a_idx
        probs_vec = transition_matrix[src_idx]  # length n_act+1

        def make_sample_route(prob_vec, n, idx2act, ma_set):
            """Sample next activity and encode using boolean flags (or is_done)."""
            def route(tok):
                # tok may be a SimToken or dict; unwrap if needed
                tok_val = tok.value if hasattr(tok, 'value') else tok
                next_idx = int(np.random.choice(len(prob_vec), p=prob_vec))
                next_act = idx2act.get(next_idx)
                if next_idx >= n or next_act not in ma_set:
                    # Mark as done using numeric flags
                    flags = {f"is_activity_{j}": 0.0 for j in range(n)}
                    flags['is_done'] = 1.0
                else:
                    flags = {f"is_activity_{j}": (1.0 if (j == next_idx) else 0.0) for j in range(n)}
                    flags['is_done'] = 0.0
                return [SimToken(flags)]
            return route

        problem.add_event(
            [routed_places[act]], [routed_next],
            make_sample_route(probs_vec, n_act, idx_to_activity, mapped_set),
            name=f"route_{s}"
        )

    dprint(f"[DEBUG] Created assign/complete/route for {len(mapped_activities)} activities")

    # ── Guarded dispatch from routed_next to individual waiting places ───
    for act in mapped_activities:
        a_idx = activity_to_idx[act]
        s = _safe(act)

        def make_guard_dispatch(target_idx):
            def guard(tok):
                val = tok.value if hasattr(tok, 'value') else tok
                return bool(val.get(f"is_activity_{target_idx}", 0))
            return guard

        def make_dispatch():
            def dispatch(tok):
                val = tok.value if hasattr(tok, 'value') else tok
                return [SimToken(val)]
            return dispatch

        problem.add_event(
            [routed_next], [waiting_places[act]],
            make_dispatch(),
            guard=make_guard_dispatch(a_idx),
            name=f"dispatch_{s}"
        )

    # Dispatch for "done" (activity_idx == n_act sentinel)
    # 'done' is represented by the 'is_done' flag being True

    def guard_done(tok):
        val = tok.value if hasattr(tok, 'value') else tok
        return bool(val.get('is_done', 0))

    def dispatch_done(tok):
        val = tok.value if hasattr(tok, 'value') else tok
        return [SimToken(val)]

    problem.add_event(
        [routed_next], [done],
        dispatch_done,
        guard=guard_done,
        name="dispatch_done"
    )
    dprint("[DEBUG] Guarded dispatch events added")

    # ── Reward: completing a case (token in done) ────────────────────────
    # Add a "sink" event that consumes from done and gives reward
    def case_done(tok):
        return []  # consume the token, nothing produced

    problem.add_event(
        [done], [],
        case_done,
        name="case_done",
        reward_function=lambda x: 1
    )
    dprint("[DEBUG] Case-done reward event added")

    dprint("[DEBUG] Sequential task assignment problem created successfully!")
    return problem


if __name__ == "__main__":
    dprint("[DEBUG] ========== STARTING PROBLEM CREATION ==========")

    dprint("[DEBUG] Step 1: Loading JSON parameters...")
    parameters = load_parameters("simulation_parameters.json")

    dprint("[DEBUG] Step 2: Creating GymPN problem...")
    problem = create_task_assignment_problem(parameters)

    mapped = [a for a in parameters['activities'] if a in parameters['activity_resource_mapping']]
    dprint(f"\n[SUCCESS] Sequential task assignment problem created!")
    dprint(f"[SUCCESS] Activities (mapped): {len(mapped)}")
    dprint(f"[SUCCESS] Resources: {len([r for r in parameters['resources'] if r is not None and not (isinstance(r, float) and np.isnan(r))])}")
    dprint(f"[SUCCESS] Avg inter-arrival time: {parameters.get('avg_inter_arrival_time', 'N/A')}")
    dprint(f"[SUCCESS] Start activities: {list(parameters.get('start_activity_probabilities', {}).keys())[:5]}...")
    dprint("[DEBUG] ========== PROBLEM CREATION COMPLETED ==========\n")

