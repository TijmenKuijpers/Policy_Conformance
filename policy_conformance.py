
import sys
import copy
import inspect
import random
import numpy as np
sys.path.append("C:/Users/20183272/OneDrive - TU Eindhoven/Documents/GitHub/gympn")
import torch

from gympn.solvers import BaseSolver
from gympn.simulator import GymProblem
from gympn.solvers import GymSolver

class PolicyConformance(GymProblem):

    """
    This class is used to analyze the conformance of a given heuristic policy to a policy trained on a given GymProblem.
    The conformance between policies is measured by:

    1. The frequency of state visits for both policies.
    2. The similarity of actions taken by both policies in the same state.
    3. The loss of rewards between the two policies from the same state.
    """

    def __init__(self, gym_problem, heuristic_solver, gym_solver, state_variables):
        super().__init__()
        # Initialize the heuristic and gym solvers
        self.heuristic_solver = heuristic_solver
        self.gym_solver = gym_solver
        self.gym_problem = gym_problem
        
        # Store a frozen copy of the initial state to reset between runs
        self.frozen_gym_problem = copy.deepcopy(gym_problem)

        # Initialize the state variables
        self.state_variables = state_variables

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

    def action_run(self, solver_1, solver_2 = None, length=100):
        """
        The action run measures the actions taken by the given solver over a specified duration.

        :param solver: An instance of a solver class implementing the `BaseSolver` interface. 
                       Steps follow solver_1. If solver_2 is provided, action mapping p2 is also updated for visited states.
        :param length: The maximum duration of the testing run. The simulation will stop if the clock exceeds (or matches) this length.
        :param reporter: A reporter to log simulation events.
        :return: The total reward accumulated during the testing run.
        """

        if not isinstance(solver_1, BaseSolver):
            raise Exception(f"The provided solver {solver_1} does not extend BaseSolver")

        if not isinstance(solver_2, BaseSolver):
            raise Exception(f"The provided solver {solver_2} does not extend BaseSolver")

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

                # Record the action taken by the second solver
                if solver_2 is not None:
                    action_2 = str(self._get_solver_action(gym_problem, solver_2))

                    # Count the action taken in the observation
                    if tuple(state_variables) not in self.p2_state_action_mapping.keys():
                        self.p2_state_action_mapping[tuple(state_variables)] = {action_2: 1}
                    elif action_2 in self.p2_state_action_mapping[tuple(state_variables)]:
                        self.p2_state_action_mapping[tuple(state_variables)][action_2] += 1
                    else:
                        self.p2_state_action_mapping[tuple(state_variables)][action_2] = 1
                    
                # Execute the action using solver 1
                timed_binding, active_model = gym_problem.step(reporter=None, length=gym_problem.length)

                # Record the action taken by the first solver (skip if step returned None, i.e. clock exceeded length)
                if timed_binding is not None:
                    action_1 = str(timed_binding[2])
                    state_key = tuple(state_variables)
                    if state_key not in self.p1_state_action_mapping:
                        self.p1_state_action_mapping[state_key] = {action_1: 1}
                    elif action_1 in self.p1_state_action_mapping[state_key]:
                        self.p1_state_action_mapping[state_key][action_1] += 1
                    else:
                        self.p1_state_action_mapping[state_key][action_1] = 1

                    # Record the states in the observation
                    if state_key not in self.states_in_observation:
                        self.states_in_observation[state_key] = [copy.deepcopy(gym_problem)]
                    else:
                        self.states_in_observation[state_key].append(copy.deepcopy(gym_problem))

    def _get_solver_action(self, state, solver):
        """
        Query the action a solver would take from the given state without executing it.
        Returns the action name (binding[2]), or None for postpone.
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
                return None
        return binding[2]

    def expected_reward_run(self, solver_1, solver_2, tau, gamma, num_rollouts, num_steps):
        """
        Compute the expected reward for each solver using N Monte Carlo rollouts per state.

        For each observation with EMD > tau, all captured states are rolled out N times
        under different random seeds to sample stochastic transitions. Rewards are averaged
        across seeds and states to produce a per-observation expected reward.
        Only reward accumulated during the rollout window is counted (pre-rollout reward
        is subtracted), so results are comparable across observations.

        :param solver_1: First policy (P1).
        :param solver_2: Second policy (P2).
        :param tau: EMD threshold; only observations with EMD > tau are evaluated.
                    Also used as the rollout horizon (state.clock + tau).
        :param gamma: Discount factor (reserved for future discounted reward computation).
        :param N: Number of Monte Carlo rollouts per state per solver.
        :return: Tuple of (expected_reward_p1, expected_reward_p2, delta_expected_reward).
        """

        overlap_obs = [key for key, value in self.state_action_emd.items() if value > tau]
        print(f'Overlapping observations: {overlap_obs}')

        for obs in overlap_obs:
            print(f'Progress: {overlap_obs.index(obs)+1}/{len(overlap_obs)}')
            states = self.states_in_observation[obs]
            print(f'Rollouts for observation: {obs}. States: {len(states)}, rollouts per state: {num_rollouts}')

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
            print(f'Skipped {skipped}/{len(states)} states (same action). Evaluated: {evaluated}')
            if evaluated > 0:
                total_reward_pi_1 /= evaluated
                total_reward_pi_2 /= evaluated

            self.expected_reward_p1[obs] = total_reward_pi_1
            self.expected_reward_p2[obs] = total_reward_pi_2
            self.delta_expected_reward[obs] = total_reward_pi_1 - total_reward_pi_2

        return self.expected_reward_p1, self.expected_reward_p2, self.delta_expected_reward

    def compute_state_visit_frequency(self, horizon=10):
        """
        Measure the frequency of state visits for the heuristic policy.
        """
        # Computing state visit frequency for the heuristic
        frequency_1 = self.frequency_run(solver=self.heuristic_solver, length=horizon)
        # Computing state visit frequency for the gym solver
        frequency_2 = self.frequency_run(solver=self.gym_solver, length=horizon)
        
        return frequency_1, frequency_2
    
    def compute_state_action_mapping(self, horizon=10):
        """
        Measure the frequency of state visits for the gym policy.
        """
        # Computing state action mapping following the heuristic solver
        self.action_run(solver_1=self.heuristic_solver, solver_2=self.gym_solver, length=horizon)
        # Computing state action mapping following the gym solver
        self.action_run(solver_1=self.gym_solver, solver_2=self.heuristic_solver, length=horizon)
        
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
