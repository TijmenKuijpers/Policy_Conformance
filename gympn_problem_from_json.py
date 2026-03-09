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

import json
import numpy as np
import sys
sys.path.append("C:/Users/lobia/PycharmProjects/policy_comparison/Policy_Conformance/gympn")

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


def create_task_assignment_problem(parameters):
    """
    Create a GymPN task assignment problem with sequential case flow.

    Cases are generated at the arrival event.  Each case follows a path
    through activities governed by ``transition_probabilities``.  At every
    activity step, an RL action decides which (task, resource) pair to bind.

    Returns a ``GymProblem`` instance ready for training or conformance analysis.
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

    # Numeric mappings
    activity_to_idx = {a: i for i, a in enumerate(activities)}
    idx_to_activity = {i: a for a, i in activity_to_idx.items()}

    valid_resources = [r for r in resources
                       if r is not None and not (isinstance(r, float) and np.isnan(r))]
    resource_to_idx = {r: i for i, r in enumerate(valid_resources)}

    # Only keep activities that have resource mappings (can actually be assigned)
    mapped_activities = [a for a in activities if a in activity_resource_mapping]
    dprint(f"[DEBUG] - {len(activities)} activities, {len(valid_resources)} resources")
    dprint(f"[DEBUG] - {len(mapped_activities)} activities with resource mappings")

    # ── Build the transition matrix as numpy array for fast sampling ─────
    # transition_matrix[i] is the probability distribution over next activities
    # An extra column (index len(activities)) represents "case done" (no successor)
    n_act = len(activities)
    transition_matrix = np.zeros((n_act, n_act + 1))  # +1 for "done"
    for src_act, targets in transition_probs.items():
        if src_act not in activity_to_idx:
            continue
        i = activity_to_idx[src_act]
        row_sum = 0.0
        for tgt_act, prob in targets.items():
            if tgt_act in activity_to_idx:
                j = activity_to_idx[tgt_act]
                transition_matrix[i, j] = prob
                row_sum += prob
        # Remaining probability mass goes to "done"
        transition_matrix[i, n_act] = max(0.0, 1.0 - row_sum)
    # Activities without transition data: 100% done
    for i in range(n_act):
        if transition_matrix[i].sum() == 0:
            transition_matrix[i, n_act] = 1.0

    dprint("[DEBUG] Transition matrix built")

    # ── Create the GymProblem ────────────────────────────────────────────
    problem = GymProblem(causal_rl=False)

    # Shared resource pool
    resource_pool = problem.add_var("resource_pool", var_attributes=['resource_idx'])
    for idx in range(len(valid_resources)):
        resource_pool.put({'resource_idx': float(idx)})
    dprint(f"[DEBUG] - Placed {len(valid_resources)} resource tokens")

    # Case counter for unique case IDs
    case_counter = [0]

    # ── Per-activity places ──────────────────────────────────────────────
    # waiting_{act}: tasks waiting to be assigned for this activity
    # busy_{act}:    tasks being executed
    # routed_{act}:  tasks that completed and await routing to next activity
    waiting_places = {}
    busy_places = {}
    routed_places = {}

    token_attrs_waiting = ['case_id', 'activity_idx']
    token_attrs_busy = ['case_id', 'activity_idx', 'resource_idx']
    token_attrs_routed = ['case_id', 'activity_idx']

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
    new_case = problem.add_var("new_case", var_attributes=['case_id', 'activity_idx'])

    def arrive(arrival_token):
        # Pick start activity
        acts = list(start_probs.keys())
        probs = [start_probs[a] for a in acts]
        start_act = np.random.choice(acts, p=probs)
        a_idx = float(activity_to_idx[start_act])

        cid = float(case_counter[0])
        case_counter[0] += 1

        delay = max(0.01, np.random.exponential(avg_inter_arrival_time))
        return [
            SimToken({'case_id': cid, 'activity_idx': a_idx}),   # new case token
            SimToken(arrival_token, delay=delay),                  # re-trigger
        ]

    problem.add_event([arrival], [new_case, arrival], arrive, name="arrive")
    dprint("[DEBUG] Arrival event added")

    # ── Route new_case to the correct waiting place ──────────────────────
    # One routing event per mapped activity with a guard on activity_idx.
    for act in mapped_activities:
        a_idx = float(activity_to_idx[act])
        s = _safe(act)

        def make_guard_new(target):
            def guard(tok):
                return tok['activity_idx'] == target
            return guard

        def make_route_new():
            def route(tok):
                return [SimToken(tok)]
            return route

        problem.add_event(
            [new_case], [waiting_places[act]],
            make_route_new(),
            guard=make_guard_new(a_idx),
            name=f"route_new_{s}"
        )

    dprint("[DEBUG] New-case routing events added")

    # ── Per-activity: assign action + complete event + routing event ─────
    # "done" place: cases that finished all activities
    done = problem.add_var("done", var_attributes=['case_id', 'activity_idx'])

    for act in mapped_activities:
        a_idx = activity_to_idx[act]
        s = _safe(act)
        res_mapping = activity_resource_mapping[act]

        # ── ASSIGN action (RL decision) ──────────────────────────────
        def make_assign(act_name, act_idx, rmap):
            def assign(task, resource):
                resource_idx = int(resource['resource_idx'])
                res_name = valid_resources[resource_idx]
                if res_name in rmap:
                    dur = max(0.01, rmap[res_name] / 3600)
                else:
                    dur = 0.1
                return [SimToken({
                    'case_id': task['case_id'],
                    'activity_idx': float(act_idx),
                    'resource_idx': float(resource_idx),
                }, delay=dur)]
            return assign

        problem.add_action(
            [waiting_places[act], resource_pool],
            [busy_places[act]],
            behavior=make_assign(act, a_idx, res_mapping),
            name=f"assign_{s}"
        )

        # ── COMPLETE event: free resource, move token to routed place ─
        def make_complete():
            def complete(tok):
                return [
                    SimToken({'resource_idx': tok['resource_idx']}),        # back to pool
                    SimToken({'case_id': tok['case_id'],
                              'activity_idx': tok['activity_idx']}),        # to routing
                ]
            return complete

        problem.add_event(
            [busy_places[act]], [resource_pool, routed_places[act]],
            make_complete(),
            name=f"complete_{s}"
        )

        # ── ROUTING event: use transition_probs to pick next activity ─
        # Build the probability vector for this source activity
        src_idx = a_idx
        probs_vec = transition_matrix[src_idx]  # length n_act+1

        # Collect valid targets (mapped activities + done)
        target_indices = list(range(n_act + 1))  # 0..n_act, last = done

        def make_route(prob_vec, n, idx2act, wp, done_place, ma_set):
            """Factory capturing per-activity routing data."""
            def route(tok):
                # Sample next activity index (or n_act = done)
                next_idx = int(np.random.choice(len(prob_vec), p=prob_vec))
                new_tok = {'case_id': tok['case_id'], 'activity_idx': float(next_idx) if next_idx < n else tok['activity_idx']}

                if next_idx >= n:
                    # Case is done
                    result = [None] * len(ma_set) + [SimToken(new_tok)]
                else:
                    next_act = idx2act.get(next_idx)
                    result = []
                    for target_act in ma_set:
                        if target_act == next_act:
                            result.append(SimToken(new_tok))
                        else:
                            result.append(None)
                    result.append(None)  # done slot = None
                return result
            return route

        # Outflow: one slot per mapped_activity waiting place + done
        outflow = [waiting_places[a] for a in mapped_activities] + [done]

        problem.add_event(
            [routed_places[act]], outflow,
            make_route(probs_vec, n_act, idx_to_activity, waiting_places, done, mapped_activities),
            name=f"route_{s}"
        )

    dprint(f"[DEBUG] Created assign/complete/route for {len(mapped_activities)} activities")

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

