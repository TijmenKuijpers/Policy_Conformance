
import sys
import copy
import inspect
import random
import numpy as np
sys.path.append("C:/Users/lobia/PycharmProjects/policy_comparison/Policy_Conformance/gympn")
import torch

from gympn.solvers import BaseSolver
from gympn.simulator import GymProblem
from gympn.solvers import GymSolver

class PolicyConformance(GymProblem):
    """
    Measures conformance between a heuristic policy (P1) and a reference policy (P2)
    on a shared GymProblem, across three per-observation metrics:

    - **Max Visit Ratio** — relative visit frequency; used as a reliability filter.
    - **EMAC** — Earth Mover's Action Conformance; total-variation distance between
      the policies' action distributions: EMAC(s) = 1 - sum_a min(P1(a|s), P2(a|s)).
    - **PERG** — Policy Expected Reward Gap; E[R|P1,o] - E[R|P2,o] estimated via
      rollouts, computed only where EMAC > rho and MaxVisitRatio > eta.
    """

    def __init__(self, gym_problem, heuristic_solver, gym_solver, state_variables, excluded_token_attrs=None):
        super().__init__()
        # Initialize the heuristic and gym solvers
        self.heuristic_solver = heuristic_solver
        self.gym_solver = gym_solver
        self.gym_problem = gym_problem
        
        # Store a frozen copy of the initial state to reset between runs
        self.frozen_gym_problem = copy.deepcopy(gym_problem)

        # Initialize the state variables
        self.state_variables = state_variables

        # Token attributes to exclude when building binding keys for action
        # comparison.  Two bindings that differ only in excluded attributes
        # are treated as the same action.
        # Format: set of attribute names, e.g. {"chip_id", "case_id"}
        self.excluded_token_attrs = set(excluded_token_attrs) if excluded_token_attrs else set()

        # Observation-level dictionaries
        self.states_in_observation = {} # Which states are recorded in each observation 
        
        self.p1_state_visit_frequency = {} # How often are observations visited by the first policy
        self.p2_state_visit_frequency = {} # How often are observations visited by the second policy

        self.p1_state_action_mapping = {} # What actions are taken by the first policy in each observation
        self.p2_state_action_mapping = {} # What actions are taken by the second policy in each observation

        self.state_action_overlap = {} # How often are the actions taken by the first policy in the same observation as the actions taken by the second policy

        # Earth mover's action distance per observation (stochastic conformance)
        self.state_action_emd = {}

        # Initialize the reward gap per observation
        self.expected_reward_p1 = {} # Expected reward for the first policy
        self.expected_reward_p2 = {} # Expected reward for the second policy
        self.delta_expected_reward = {} # Expected reward gap between the two policies

    def evaluate(self, gym_problem, function):
        """
        Evaluated the given function, with the parameters of the function bounds to the corresponding values of the binding.
        """
        # Get the function signature
        sig = inspect.signature(function)
        # Create a dictionary of parameter names to values
        params = {}
        for k in sig.parameters:
            if k == "clock":
                params[k] = gym_problem.clock
            elif k.endswith("_queue"):
                queue_content = gym_problem.var(k[:-6]).marking
                token_values = [token.value for token in queue_content]
                params[k] = token_values
            elif k.endswith("_queue_tokens"):
                # Pass actual tokens instead of just values
                queue_content = gym_problem.var(k[:-13]).marking
                params[k] = list(queue_content)  # Pass tokens
            #else:
                #arams[k] = binding[k]
        # Call the function with the bound parameters
        return function(**params)
    
    def calculate_functions(self, gym_problem, functions):
        """
        Calculates the values of the functions for the given model bindings and log event.
        """
        function_results = [self.evaluate(gym_problem, f) for f in functions]

        return function_results

    def compute_state_visit_frequency(self, horizon=10):
        """
        Measure the frequency of state visits for both policies and store the
        results in ``self.p1_state_visit_frequency`` and
        ``self.p2_state_visit_frequency``.

        Counts are accumulated so the method can be called multiple times
        (e.g. over several episodes) without losing earlier results.
        """
        frequency_1 = self.frequency_run(solver=self.heuristic_solver, length=horizon)
        frequency_2 = self.frequency_run(solver=self.gym_solver, length=horizon)

        for obs, count in frequency_1.items():
            self.p1_state_visit_frequency[obs] = self.p1_state_visit_frequency.get(obs, 0) + count
        for obs, count in frequency_2.items():
            self.p2_state_visit_frequency[obs] = self.p2_state_visit_frequency.get(obs, 0) + count

        return frequency_1, frequency_2

    def compute_max_visit_ratio(self):
        """
        Compute the max visit ratio for every observation seen by either policy.

        For observation ``s``:

            MaxVisitRatio(s) = max( v_P1(s) / total_P1,  v_P2(s) / total_P2 )

        where ``total_Pi`` is the total number of action-step visits recorded
        for policy ``i``.  This measures the largest *relative* attention either
        policy pays to ``s``; observations with a low ratio were rarely visited
        and are statistically less reliable.

        :return: dict mapping observation tuple -> max visit ratio in (0, 1].
        :raises RuntimeError: if ``compute_state_visit_frequency`` has not been
                              called yet.
        """
        if not self.p1_state_visit_frequency and not self.p2_state_visit_frequency:
            raise RuntimeError(
                "compute_state_action_mapping() must be called before "
                "compute_max_visit_ratio()."
            )

        total_p1 = sum(self.p1_state_visit_frequency.values())
        total_p2 = sum(self.p2_state_visit_frequency.values())

        all_obs = (set(self.p1_state_visit_frequency.keys())
                   | set(self.p2_state_visit_frequency.keys()))

        max_visit_ratio = {}
        for obs in all_obs:
            r_p1 = self.p1_state_visit_frequency.get(obs, 0) / total_p1 if total_p1 > 0 else 0.0
            r_p2 = self.p2_state_visit_frequency.get(obs, 0) / total_p2 if total_p2 > 0 else 0.0
            max_visit_ratio[obs] = max(r_p1, r_p2)
        return max_visit_ratio

    def _binding_key(self, binding):
        """
        Build a hashable key that identifies a binding by its action name
        AND the tokens it consumes, so that two bindings of the same action
        but different resource/task choices are distinguished.

        Token attributes listed in ``self.excluded_token_attrs`` are stripped
        from the key so that bindings differing only in those attributes
        are treated as the same action.

        Handles multiple binding formats:
          - From bindings():            ([(place, token), ...], time, transition)
          - From get_graph_observation: ([(place, token), ...], time, transition)
          - Postpone pseudo-binding:    (['postpone'], time, None)
        """
        transition_name = str(binding[2])
        token_parts = []

        # Postpone pseudo-binding: transition is None, or the token list
        # contains a bare 'postpone' string (GymSolver may produce this when
        # the postpone action is selected from obs['actions_dict']).
        if binding[2] is None or (
            isinstance(binding[0], (list, tuple))
            and len(binding[0]) == 1
            and str(binding[0][0]) == "postpone"
        ):
            return "postpone"
        
        for item in binding[0]:
            if isinstance(item, (tuple, list)):
                # (place, token) pair — extract the token value
                raw = item[-1].value if hasattr(item[-1], 'value') else item[-1]
            elif hasattr(item, 'value'):
                # bare token
                raw = item.value
            else:
                # string like 'postpone' or other primitive
                raw = item

            # Strip excluded attributes from dict-valued tokens
            if isinstance(raw, dict) and self.excluded_token_attrs:
                raw = {k: v for k, v in raw.items() if k not in self.excluded_token_attrs}

            token_parts.append(str(raw))
        return transition_name + "|" + ";".join(token_parts)

    def _get_solver_action(self, state, solver):
        """
        Query the action a solver would take from the given state without executing it.
        Returns a hashable key that captures the full binding (action + tokens),
        or None for postpone.
        """
        bindings, _ = state.bindings()
        if not bindings or not state.network_tag.is_action():
            return None

        if isinstance(solver, GymSolver):
            obs = state.get_graph_observation()
            act_probs = solver.solve(obs)
            max_index = torch.argmax(act_probs).item()
            binding = obs['actions_dict'][max_index]
        else:
            obs = state.get_heuristic_observation()
            aug_bindings = state._augment_bindings_with_postpone(bindings)
            binding = solver.solve(obs, aug_bindings)
            if binding == "postpone":
                return "postpone"
        return self._binding_key(binding)

    def compute_state_action_mapping(self, horizon=10):
        """
        Record the actions each policy takes across both policies' trajectories.

        Two simulation runs are performed:

        1. Heuristic trajectory — heuristic executes, gym is queried counterfactually.
           Heuristic's actions  → p1_state_action_mapping
           Gym's actions        → p2_state_action_mapping

        2. Gym trajectory — gym executes, heuristic is queried counterfactually.
           Gym's actions        → p2_state_action_mapping  (mapping_1 is routed to p2)
           Heuristic's actions  → p1_state_action_mapping  (mapping_2 is routed to p1)

        Routing the second call's destination dicts explicitly prevents the
        cross-contamination where gym actions would otherwise accumulate in p1
        and heuristic actions in p2.
        """
        # Run 1: follow heuristic trajectory — also accumulate p1 visit frequencies
        self.action_run(
            solver_1=self.heuristic_solver,
            solver_2=self.gym_solver,
            length=horizon,
            mapping_1=self.p1_state_action_mapping,
            mapping_2=self.p2_state_action_mapping,
            freq_mapping=self.p1_state_visit_frequency,
        )
        # Run 2: follow gym trajectory — swap destination dicts, accumulate p2 visits
        self.action_run(
            solver_1=self.gym_solver,
            solver_2=self.heuristic_solver,
            length=horizon,
            mapping_1=self.p2_state_action_mapping,   # gym's executed actions → p2
            mapping_2=self.p1_state_action_mapping,   # heuristic's counterfactual → p1
            freq_mapping=self.p2_state_visit_frequency,
        )

        return self.p1_state_action_mapping, self.p2_state_action_mapping

    def compute_action_probability(self, action_mapping):
        """
        Compute the probability that the given action is taken in the given state.
        """
        # Compute the probability that the given action is taken in the given state
        action_probability = {}
        for state in action_mapping.keys():
            action_probability[state] = {}
            
            # The probability of an action in a state is the number of times the action is taken in the state divided by the total number of actions taken following pi_1 and pi_2.
            for action in action_mapping[state].keys():
                action_probability[state][action] = round(action_mapping[state][action] / sum(action_mapping[state].values()), 4)

        return action_probability

    def frequency_run(self, solver, length, reporter=None):

        """
        The frequency run measures the frequency of state visits for the given solver over a specified duration.
        :param solver: An instance of a solver class implementing the `BaseSolver` interface.
        :param length: The maximum duration of the testing run. The simulation will stop if the clock exceeds (or matches) this length.
        :param reporter: A reporter to log simulation events.
        :return: The frequency of state visits for the given solver.
        """

        if not isinstance(solver, BaseSolver):
            raise Exception(f"The provided solver {solver} does not extend BaseSolver")

        # Reset the gym_problem to initial state before each run
        gym_problem = copy.deepcopy(self.frozen_gym_problem)
        
        gym_problem.set_solver(solver)
        gym_problem.length = length
        active_model = True
        
        state_visit_frequency = None
        state_visit_frequency = {}

        while gym_problem.clock <= gym_problem.length and active_model:

            state_variables = self.calculate_functions(gym_problem, self.state_variables)
            binding, active_model = gym_problem.step(reporter, gym_problem.length)
            
            if gym_problem.clock > gym_problem.length:
                #print("Clock exceeds length")
                break
            
            # After each step, calculate the state variables and update the conformance measures  
            all_actions = gym_problem.actions + [None]
            if binding[2] in all_actions:
                if tuple(state_variables) not in state_visit_frequency.keys():
                    state_visit_frequency[tuple(state_variables)] = 1
                else:
                    state_visit_frequency[tuple(state_variables)] += 1

        return state_visit_frequency

    def action_run(self, solver_1, solver_2=None, length=100,
                   mapping_1=None, mapping_2=None, freq_mapping=None):
        """
        The action run measures the actions taken by the given solver over a specified duration.

        The trajectory follows solver_1.  At each action step:
          - solver_1's executed action is recorded into mapping_1,
          - solver_2's counterfactual action is recorded into mapping_2,
          - the visit count for that observation is incremented in freq_mapping.

        All three dicts accumulate across repeated calls.  mapping_1/mapping_2
        default to self.p1/p2_state_action_mapping; freq_mapping defaults to
        None (no frequency tracking) and is set explicitly by
        compute_state_action_mapping to keep visit counts aligned with the
        action recordings from the same simulation run.

        :param solver_1: Solver whose trajectory is followed (executes actions).
        :param solver_2: Solver whose counterfactual action is recorded at each step.
        :param length: Maximum simulation duration.
        :param mapping_1: Dict to accumulate solver_1's executed actions into.
        :param mapping_2: Dict to accumulate solver_2's counterfactual actions into.
        :param freq_mapping: Dict to accumulate per-observation visit counts for
                             solver_1's trajectory.  Pass None to skip tracking.
        """

        if not isinstance(solver_1, BaseSolver):
            raise Exception(f"The provided solver {solver_1} does not extend BaseSolver")

        if not isinstance(solver_2, BaseSolver):
            raise Exception(f"The provided solver {solver_2} does not extend BaseSolver")

        if mapping_1 is None:
            mapping_1 = self.p1_state_action_mapping
        if mapping_2 is None:
            mapping_2 = self.p2_state_action_mapping

        # Reset the gym_problem to initial state before each run
        gym_problem = copy.deepcopy(self.frozen_gym_problem)
        gym_problem.set_solver(solver_1)
        gym_problem.length = length
        
        active_model = True

        while gym_problem.clock <= gym_problem.length and active_model:
           
            bindings, active_model = gym_problem.bindings()
            if len(bindings) > 0 and gym_problem.network_tag.is_evolution():
                timed_binding = bindings[0]
                gym_problem.fire(timed_binding)
            
            elif len(bindings) > 0 and gym_problem.network_tag.is_action():
                state_variables = self.calculate_functions(gym_problem, self.state_variables)

                # Record the counterfactual action of solver_2 into mapping_2
                if solver_2 is not None:
                    action_2 = str(self._get_solver_action(gym_problem, solver_2))
                    state_key_2 = tuple(state_variables)
                    if state_key_2 not in mapping_2:
                        mapping_2[state_key_2] = {action_2: 1}
                    elif action_2 in mapping_2[state_key_2]:
                        mapping_2[state_key_2][action_2] += 1
                    else:
                        mapping_2[state_key_2][action_2] = 1

                # Execute the action using solver_1
                timed_binding, active_model = gym_problem.step(reporter=None, length=gym_problem.length)

                # Record solver_1's executed action into mapping_1
                if timed_binding is not None:
                    action_1 = self._binding_key(timed_binding)
                    state_key = tuple(state_variables)
                    if state_key not in mapping_1:
                        mapping_1[state_key] = {action_1: 1}
                    elif action_1 in mapping_1[state_key]:
                        mapping_1[state_key][action_1] += 1
                    else:
                        mapping_1[state_key][action_1] = 1

                    # Track visit frequency at action states (Option C)
                    if freq_mapping is not None:
                        freq_mapping[state_key] = freq_mapping.get(state_key, 0) + 1

                    # Record the states in the observation (only for the executing solver's trajectory)
                    if state_key not in self.states_in_observation:
                        self.states_in_observation[state_key] = [copy.deepcopy(gym_problem)]
                    else:
                        self.states_in_observation[state_key].append(copy.deepcopy(gym_problem))

    def expected_reward_run(self, solver_1, solver_2, rho, eta, num_rollouts, num_steps):
        """
        Compute the expected reward for each solver using rollouts.

        Only observations that pass **both** thresholds are evaluated:

        * ``EMAC(s) > rho``  — the action distributions differ enough to be
          worth investigating (stochastic conformance filter).
        * ``MaxVisitRatio(s) > eta``  — the observation was visited frequently
          enough by at least one policy to be worth investigating
          (visit-frequency filter).  Set ``eta=0`` to disable this filter.

        Rewards are averaged across rollout seeds and captured observations.  Only
        reward accumulated during the rollout window is counted (pre-rollout
        reward is subtracted) so results are comparable across observations.

        :param solver_1: First policy (P1).
        :param solver_2: Second policy (P2).
        :param rho: EMAC threshold; only observations with EMAC > rho pass.
        :param eta: Max visit ratio threshold; only observations where
                    MaxVisitRatio > eta pass.
        :param num_rollouts: Monte Carlo rollouts per state per solver.
        :param num_steps: Rollout window length added to ``state.clock``.
        :return: Tuple of (expected_reward_p1, expected_reward_p2, delta_expected_reward).
        """

        max_visit_ratio = self.compute_max_visit_ratio()

        overlap_obs = [
            key for key, value in self.state_action_emd.items()
            if value > rho and max_visit_ratio.get(key, 0.0) > eta
        ]
        print(f"Observations passing filters (EMAC>{rho}, MaxVisitRatio>{eta}): {overlap_obs}")

        for obs in overlap_obs:
            #print(f'Progress: {overlap_obs.index(obs)+1}/{len(overlap_obs)}')
            states = self.states_in_observation[obs]
            #print(f'Rollouts for observation: {obs}. States: {len(states)}, rollouts per state: {num_rollouts}')

            total_reward_pi_1 = 0.0
            total_reward_pi_2 = 0.0

            skipped = 0
            for state in states:
                rollout_horizon = state.clock + num_steps

                # Skip states where both solvers choose the same action
                action_1 = self._get_solver_action(state, solver_1)
                action_2 = self._get_solver_action(state, solver_2)
                if action_1 == action_2:
                    skipped += 1
                    continue

                # N rollouts for solver 1
                seed_reward_pi_1 = 0.0
                for k in range(num_rollouts):
                    random.seed(k)
                    np.random.seed(k)

                    rollout = copy.deepcopy(state)
                    rollout.set_solver(solver_1)
                    rollout.length = rollout_horizon
                    initial_reward = rollout.reward

                    active = True
                    while rollout.clock <= rollout.length and active:
                        bindings, active = rollout.bindings()
                        if len(bindings) > 0 and rollout.network_tag.is_evolution():
                            rollout.fire(bindings[0])
                        elif len(bindings) > 0 and rollout.network_tag.is_action():
                            _, active = rollout.step(reporter=None, length=rollout.length)

                    seed_reward_pi_1 += (rollout.reward - initial_reward) / num_rollouts

                total_reward_pi_1 += seed_reward_pi_1

                # N rollouts for solver 2
                seed_reward_pi_2 = 0.0
                for k in range(num_rollouts):
                    random.seed(k)
                    np.random.seed(k)

                    rollout = copy.deepcopy(state)
                    rollout.set_solver(solver_2)
                    rollout.length = rollout_horizon
                    initial_reward = rollout.reward

                    active = True
                    while rollout.clock <= rollout.length and active:
                        bindings, active = rollout.bindings()
                        if len(bindings) > 0 and rollout.network_tag.is_evolution():
                            rollout.fire(bindings[0])
                        elif len(bindings) > 0 and rollout.network_tag.is_action():
                            _, active = rollout.step(reporter=None, length=rollout.length)

                    seed_reward_pi_2 += (rollout.reward - initial_reward) / num_rollouts

                total_reward_pi_2 += seed_reward_pi_2

            evaluated = len(states) - skipped
            #print(f'Skipped {skipped}/{len(states)} states (same action). Evaluated: {evaluated}')
            if evaluated > 0:
                total_reward_pi_1 /= evaluated
                total_reward_pi_2 /= evaluated

            self.expected_reward_p1[obs] = total_reward_pi_1
            self.expected_reward_p2[obs] = total_reward_pi_2
            self.delta_expected_reward[obs] = total_reward_pi_1 - total_reward_pi_2

        return self.expected_reward_p1, self.expected_reward_p2, self.delta_expected_reward

    @staticmethod
    def earth_mover_distance_action_distributions(prob_1, prob_2):
        """
        Earth Mover's Distance between two action probability distributions for a single state.

        With unit cost between different actions (d(a,b)=0 if a==b, 1 if a!=b), the minimal
        cost to transform one distribution into the other is:
          EMD = 1 - sum over actions a of min(P(a), Q(a)).

        :param prob_1: dict action -> probability (policy 1 at this state)
        :param prob_2: dict action -> probability (policy 2 at this state)
        :return: EMD in [0, 1]; 0 = identical distributions, 1 = no overlap
        """
        all_actions = set(prob_1.keys()) | set(prob_2.keys())
        if not all_actions:
            return 0.0
        overlap = sum(min(prob_1.get(a, 0.0), prob_2.get(a, 0.0)) for a in all_actions)
        return round(1.0 - overlap, 4)
    
    def compute_state_action_emd(self, action_probability_1, action_probability_2):
        """
        Earth movers' stochastic conformance for state-action probabilities.

        For each state visited by both policies, computes the Earth Mover's Distance
        between the two action probability distributions. With unit cost between
        different actions, EMD = 1 - sum_a min(P(a), Q(a)) (total variation).

        :param action_probability_1: state -> {action -> prob} for policy 1
        :param action_probability_2: state -> {action -> prob} for policy 2
        :return: dict state -> EMD in [0, 1]; 0 = identical, 1 = no overlap
        """
        overlapping_states = set(action_probability_1.keys()) & set(action_probability_2.keys())

        for state in overlapping_states:
            self.state_action_emd[state] = self.earth_mover_distance_action_distributions(
                action_probability_1[state], action_probability_2[state]
            )
        return self.state_action_emd
