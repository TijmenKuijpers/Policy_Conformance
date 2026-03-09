"""
Create a GymPN task assignment problem with DISJOINT ACTIONS from JSON parameters.

With the sequential case-flow model in gympn_problem_from_json.py, every activity
already has its own action transition (assign_{activity}).  This module re-exports
that function under its original name for backward compatibility with scripts
that import from gympn_problem_disjoint.

If you need the old flat single-queue model, see git history.
"""

from gympn_problem_from_json import (
    load_parameters,
    create_task_assignment_problem as create_disjoint_task_assignment_problem,
)

# Re-export for backward compatibility
__all__ = ['load_parameters', 'create_disjoint_task_assignment_problem']

if __name__ == "__main__":
    print("[INFO] gympn_problem_disjoint now delegates to gympn_problem_from_json", flush=True)
    print("[INFO] Both produce sequential case-flow with one action per activity", flush=True)
    parameters = load_parameters("simulation_parameters.json")
    problem = create_disjoint_task_assignment_problem(parameters)
    print("[SUCCESS] Problem created", flush=True)

