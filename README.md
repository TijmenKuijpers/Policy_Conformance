# Decision Policy Conformance in Process Models

This repository contains the Python implementation accompanying the paper **Decision Policy Conformance in Process Models**.

The code evaluates how closely alternative decision policies conform in process-model simulations. It builds sequential task-assignment models from event-log data, runs heuristic and learned policies, and compares their behavior at the level of observed process states.

## What the code does

The implementation:

- Extracts activities, resources, durations, and transition probabilities from the BPI Challenge 2012 event log.
- Builds sequential GymPN/SimpN process simulations with activity queues, shared resources, routing, and completion rewards.
- Implements FIFO, shortest-processing-time (SPT), random, and PPO-based policies.
- Records the states visited and actions selected by each policy.
- Computes policy-conformance measures, including:
  - **Max Visit Ratio** — the relative frequency with which an observation is visited.
  - **EMAC** — Earth Mover's Action Conformance, measuring differences between action distributions.
  - **PERG** — Policy Expected Reward Gap, comparing expected rewards through rollouts.
- Produces policy tables, conformance-result CSV files, LaTeX tables, heatmaps, histograms, and metric plots.

## Repository structure

```text
BPI_2012_conformance.py
  Detailed BPI 2012 FIFO-versus-SPT conformance experiment.

BPI_2012_policy_conformance_with_ids.py
  Heuristic policy definitions and broader BPI 2012 analysis.

policy_conformance.py
  Core PolicyConformance implementation and conformance metrics.

parameters_extraction.py
  Extracts simulation parameters from the BPI 2012 XES event log.

gympn_problem_from_json.py
  Builds the sequential GymPN task-assignment model using one-hot attributes.

gympn_problem_from_json_with_ids.py
  Builds the numeric-ID variant of the sequential task-assignment model.

gympn_problem_disjoint.py
  Backward-compatible wrapper around the JSON-based problem builder.

choice_gym.py
  Assembly-system simulation and policy-conformance analysis.

choice_gym_experiments.py
  Compares multiple assembly-system policies and generates visualizations.

data/
  BPI Challenge 2012 event log, training runs, TensorBoard data, and policy weights.

results/
  Generated policy tables, conformance metrics, plots, and histograms.

simulation_parameters.json
  Extracted process activities, resources, timings, and transition probabilities.

requirements.txt
  Python dependencies.
```

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The project also depends on [GymPN](https://github.com/bpogroup/gympn), which is installed from its Git repository through `requirements.txt`. If it is installed separately, make sure it is available on the Python import path.

## Running the experiments

Extract simulation parameters from the included BPI Challenge 2012 event log:

```bash
python parameters_extraction.py
```

Create the sequential task-assignment model:

```bash
python gympn_problem_from_json_with_ids.py
```

Run the detailed FIFO-versus-SPT analysis:

```bash
python BPI_2012_conformance.py
```

Run the broader BPI 2012 conformance analysis:

```bash
python BPI_2012_policy_conformance_with_ids.py
```

Run the assembly-system experiments:

```bash
python choice_gym.py
python choice_gym_experiments.py
```

The experiment scripts write their generated tables and figures to the working directory. Example result files are also included under `results/`.

## Data and models

The main process-mining example uses `data/BPI_Challenge_2012.xes`. The extracted `simulation_parameters.json` file contains activity-resource mappings, average durations, inter-arrival information, start-activity probabilities, and activity-transition probabilities.

The repository also includes trained-policy artifacts and experiment logs under `data/train/` and `.wandb_offline/`.

## Citation

This implementation accompanies the paper:

> **Decision Policy Conformance in Process Models**

Please add the paper's complete bibliographic citation when using this code in academic work.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
