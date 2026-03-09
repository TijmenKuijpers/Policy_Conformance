"""
Trains a PPO agent on a task assignment problem created from BPI Challenge 2012 data.
Uses functions from gympn_problem_from_json.py to create the GymPN problem.
"""

import sys
import os
import copy
import warnings
import signal

sys.path.append("C:/Users/lobia/PycharmProjects/policy_comparison/Policy_Conformance/gympn")

import numpy as np
from gympn.solvers import GymSolver, RandomSolver, HeuristicSolver
from gympn.visualisation import Visualisation
from gympn_problem_from_json import load_parameters, create_task_assignment_problem
from gympn_problem_disjoint import create_disjoint_task_assignment_problem

warnings.simplefilter(action='ignore', category=FutureWarning)


def timeout_handler(signum, frame):
    """Handle timeout by raising an exception"""
    raise TimeoutError("Training exceeded maximum time limit")


# Set signal handler for timeout (only on Unix systems)
if hasattr(signal, 'SIGALRM'):
    signal.signal(signal.SIGALRM, timeout_handler)


if __name__ == "__main__":

    ###########################################################################
    # Run configurations
    train = True  # Set to False to test a trained model
    disjoint_actions = False  # True: one action per activity type; False: single "assign" action
    visualize_random = False  # Set to True to visualize the random solver
    visualize_heuristic = False  # Set to True to visualize the heuristic solver
    visualize_ppo = True  # Set to True to visualize the PPO solver

    # Path to save/load the trained model (separate dirs for each problem variant)
    variant = "disjoint" if disjoint_actions else "joint"
    model_save_dir = f"data/train/bpi_2012_ppo_{variant}"
    weights_path = os.path.join(model_save_dir, "best_policy.pth")

    ###########################################################################

    print("[DEBUG] ========== TRAINING SCRIPT STARTED ==========", flush=True)
    print(f"[DEBUG] Problem variant: {'DISJOINT (one action per activity)' if disjoint_actions else 'JOINT (single assign action)'}", flush=True)

    # Create the GymProblem from JSON parameters
    print("[DEBUG] Step 1: Loading parameters from JSON...", flush=True)
    parameters = load_parameters("simulation_parameters.json")
    print(f"[DEBUG] - Loaded {len(parameters['activities'])} activities", flush=True)
    print(f"[DEBUG] - Loaded {len(parameters['resources'])} resources", flush=True)

    print("[DEBUG] Step 2: Creating GymProblem...", flush=True)
    if disjoint_actions:
        problem = create_disjoint_task_assignment_problem(parameters)
    else:
        problem = create_task_assignment_problem(parameters)
    print("[DEBUG] GymProblem created successfully", flush=True)


    ###########################################################################
    # Default training arguments (customize as needed)
    default_args = {
        # Algorithm Parameters
        "algorithm": "ppo-clip",
        "gam": 1.0,  # With finite horizon, use gam=1
        "lam": 0.99,
        "eps": 0.3,
        "c": 0.2,
        "ent_bonus": 0.0,
        "agent_seed": 42,  # Fixed seed for reproducibility

        # Policy Model
        "policy_model": "gnn",
        "policy_kwargs": {"hidden_layers": [32]},  # Reduced hidden layer size
        "policy_lr": 3e-4,
        "policy_updates": 1,  # Reduced from 2
        "policy_kld_limit": 0.01,
        "policy_weights": "",
        "policy_network": "",
        "score": False,
        "score_weight": 1e-3,

        # Value Model
        "value_model": "gnn",
        "value_kwargs": {"hidden_layers": [32]},  # Reduced hidden layer size
        "value_lr": 3e-4,
        "value_updates": 5,  # Reduced from 10
        "value_weights": "",

        # Training Parameters
        "episodes": 5,  # Reduced from 5 for faster testing
        "epochs": 100,  # Reduced from 50 for faster testing
        "max_episode_length": 100,  # Add max episode length to prevent infinite loops
        "batch_size": 32,  # Reduced from 64
        "sort_states": False,
        "use_gpu": False,
        "load_policy_network": False,
        "verbose": 1,

        # Saving Parameters
        "name": f"bpi_2012_ppo_{variant}",
        "datetag": False,
        "logdir": "data/train",
        "save_freq": 1,
        "open_tensorboard": False,

        # Wandb Parameters
        "use_wandb": False,
    }

    ###########################################################################

    def heuristic_solver_function(observable_net, tokens_comb):
        """
        Simple heuristic: assign tasks to available resources (first available).
        """
        for k, el in tokens_comb.items():
            if el:
                # Select the first valid binding
                return {k: el[0]}
        return None

    ###########################################################################

    if train:
        print("\n[DEBUG] Step 3: Starting training mode...", flush=True)
        print(f"[DEBUG] - Training for {default_args['episodes']} episodes, {default_args['epochs']} epochs", flush=True)
        print(f"[DEBUG] - Max episode length: {default_args['max_episode_length']} time units", flush=True)
        print(f"[DEBUG] - Policy model: {default_args['policy_model']} with hidden layers {default_args['policy_kwargs']['hidden_layers']}", flush=True)
        print("[DEBUG] - Initializing PPO training run...", flush=True)
        print("[DEBUG] ========== TRAINING STARTED ==========\n", flush=True)

        try:
            # Set timeout for training (30 minutes per epoch)
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(1800)

            problem.training_run(length=50, args_dict=default_args)

            # Cancel alarm if training completes
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

            print("\n[DEBUG] ========== TRAINING COMPLETED ==========")
            print("[SUCCESS] Training completed!")
        except TimeoutError as e:
            print(f"\n[ERROR] {e}")
            print("[ERROR] Training exceeded timeout limit")
        except Exception as e:
            print(f"\n[ERROR] Training failed with error: {e}")
            print(f"[ERROR] Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()

    else:
        print("\n[DEBUG] Step 3: Starting evaluation mode...")
        print("[DEBUG] ========== EVALUATION STARTED ==========\n")

        # Evaluate Random Solver
        random_rewards = []
        print("[DEBUG] Evaluating Random Solver...")
        if visualize_random:
            print("[DEBUG] - Visualization mode enabled")
            frozen_problem = copy.deepcopy(problem)
            frozen_problem.set_solver(RandomSolver())
            visual = Visualisation(frozen_problem)
            visual.show()
        else:
            for i in range(5):
                print(f"[DEBUG] - Random run {i+1}/5...")
                frozen_problem = copy.deepcopy(problem)
                res = frozen_problem.testing_run(length=50, solver=RandomSolver())
                random_rewards.append(res)
                print(f"  Run {i+1}: Reward = {res}")

            random_avg = np.mean(random_rewards)
            random_std = np.std(random_rewards)
            print(f"Random Solver - Average: {random_avg:.2f}, Std: {random_std:.2f}")

        # Evaluate Heuristic Solver
        heuristic_rewards = []
        print("\n[DEBUG] Evaluating Heuristic Solver...")
        if visualize_heuristic:
            print("[DEBUG] - Visualization mode enabled")
            frozen_problem = copy.deepcopy(problem)
            frozen_problem.set_solver(HeuristicSolver(heuristic_solver_function))
            visual = Visualisation(frozen_problem)
            visual.show()
        else:
            for i in range(5):
                print(f"[DEBUG] - Heuristic run {i+1}/5...")
                frozen_problem = copy.deepcopy(problem)
                solver = HeuristicSolver(heuristic_solver_function)
                res = frozen_problem.testing_run(length=50, solver=solver)
                heuristic_rewards.append(res)
                print(f"  Run {i+1}: Reward = {res}")

            heuristic_avg = np.mean(heuristic_rewards)
            heuristic_std = np.std(heuristic_rewards)
            print(f"Heuristic Solver - Average: {heuristic_avg:.2f}, Std: {heuristic_std:.2f}")

        # Evaluate Trained DRL Solver
        ppo_rewards = []
        print("\n[DEBUG] Evaluating Trained PPO Solver...")
        if not os.path.exists(weights_path):
            print(f"[DEBUG] Warning: Trained model not found at {weights_path}")
            print("[DEBUG] Please train the model first by setting train=True")
        else:
            if visualize_ppo:
                print("[DEBUG] - Visualization mode enabled")
                frozen_problem = copy.deepcopy(problem)
                frozen_problem.set_solver(GymSolver(weights_path=weights_path, metadata=problem.make_metadata()))
                visual = Visualisation(frozen_problem)
                visual.show()
            else:
                for i in range(5):
                    print(f"[DEBUG] - PPO run {i+1}/5...")
                    frozen_problem = copy.deepcopy(problem)
                    solver = GymSolver(weights_path=weights_path, metadata=problem.make_metadata())
                    res = frozen_problem.testing_run(length=50, solver=solver)
                    ppo_rewards.append(res)
                    print(f"  Run {i+1}: Reward = {res}")

                ppo_avg = np.mean(ppo_rewards)
                ppo_std = np.std(ppo_rewards)
                print(f"PPO Solver - Average: {ppo_avg:.2f}, Std: {ppo_std:.2f}")

        # Summary comparison
        if not visualize_random and not visualize_heuristic and not visualize_ppo and ppo_rewards:
            print("\n" + "="*50)
            print("PERFORMANCE SUMMARY")
            print("="*50)
            print(f"Random Solver   - Avg: {random_avg:>7.2f}, Std: {random_std:>6.2f}")
            print(f"Heuristic Solver - Avg: {heuristic_avg:>7.2f}, Std: {heuristic_std:>6.2f}")
            print(f"PPO Solver      - Avg: {ppo_avg:>7.2f}, Std: {ppo_std:>6.2f}")
            print("="*50)

        print("\n[DEBUG] ========== EVALUATION COMPLETED ==========")

