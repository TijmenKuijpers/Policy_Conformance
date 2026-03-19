# Extracts the parameters for simulation with gympn and saves them to a json file.

import json
import pm4py
import pandas as pd


if __name__ == "__main__":
    log = pm4py.read_xes("datasets/BPI_Challenge_2012.xes")

    #check how many unique activities there are
    activities = log["concept:name"].unique()
    print(f"Unique activities: {activities}")

    #check how many unique resources there are
    resources = log["org:resource"].unique()
    print(f"Unique resources: {resources}")

    #create a dictionary that specifies which activities can be performed by which resources
    activity_resource_mapping = {}
    for activity in activities:
        activity_resource_mapping[activity] = {}
        resources_for_activity = log[log["concept:name"] == activity]["org:resource"].unique().tolist()
        for resource in resources_for_activity:
            avg_duration = log[(log["concept:name"] == activity) & (log["org:resource"] == resource)]["time:timestamp"].diff().mean().total_seconds()
            activity_resource_mapping[activity][resource] = avg_duration

    #extract average inter-arrival time of cases
    log["case:duration"] = log.groupby("case:concept:name")["time:timestamp"].diff().dt.total_seconds()
    avg_inter_arrival_time = log.groupby("case:concept:name")["case:duration"].first().mean()

    #extract the transition probability between activities (within each case)
    # IMPORTANT: the last activity in each case transitions to "__DONE__" (case finished)
    log = log.sort_values(["case:concept:name", "time:timestamp"])
    log["next_activity"] = log.groupby("case:concept:name")["concept:name"].shift(-1)
    # Replace NaN (last event of each case) with "__DONE__" instead of dropping
    log["next_activity"] = log["next_activity"].fillna("__DONE__")
    transitions = log  # keep all rows including end-of-case
    all_targets = list(activities) + ["__DONE__"]
    transition_counts = transitions.groupby(["concept:name", "next_activity"]).size().unstack(fill_value=0)
    # Reindex so every activity appears as row, and all targets (incl __DONE__) as columns
    transition_counts = transition_counts.reindex(index=activities, columns=all_targets, fill_value=0)
    transition_probabilities = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)

    #extract the start activity probabilities (which activity starts a case)
    start_activities = log.sort_values(["case:concept:name", "time:timestamp"]).groupby("case:concept:name")["concept:name"].first()
    start_activity_counts = start_activities.value_counts()
    start_activity_probabilities = (start_activity_counts / start_activity_counts.sum()).to_dict()

    #save the parameters to a json file
    parameters = {
        "activities": activities.tolist(),
        "resources": resources.tolist(),
        "activity_resource_mapping": activity_resource_mapping,
        "avg_inter_arrival_time": avg_inter_arrival_time,
        "transition_probabilities": transition_probabilities.to_dict(orient='index'),
        "start_activity_probabilities": start_activity_probabilities,
    }


    with open("simulation_parameters.json", "w") as f:
        json.dump(parameters, f, indent=4)