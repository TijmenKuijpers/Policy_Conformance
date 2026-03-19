#!/usr/bin/env python3
#!/usr/bin/env python3
"""
regenerate_bpi_plots.py

Read the policy table CSV(s) produced by the BPI conformance script and
recreate heatmaps and metric charts without recomputing conformance.

This script shows a numeric y-axis scale (row indices) on the left heatmap
and removes any labels/ticks on the x-axis as requested.
"""
from __future__ import annotations
import argparse
import os
from typing import Dict, List, Optional

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def find_file_by_pattern(directory: str, pattern: str) -> Optional[str]:
    for fname in os.listdir(directory):
        if pattern in fname:
            return os.path.join(directory, fname)
    return None


def read_policy_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    # choose observation column
    obs_col = None
    for cand in ["Observation", "observation", "Observation_key", "state", "State"]:
        if cand in df.columns:
            obs_col = cand
            break
    if obs_col is None:
        obs_col = df.columns[0]

    df = df.copy()
    df[obs_col] = df[obs_col].astype(str)

    # Convert remaining columns to numeric probabilities if possible
    action_cols = [c for c in df.columns if c != obs_col]
    for c in action_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

    # store metadata about which column holds the observation and which likely holds visit counts
    df.attrs = getattr(df, 'attrs', {})
    df.attrs['obs_col'] = obs_col
    # find visit-count column (common names) or fallback to the second column
    visit_candidates = ["Visit Freq", "VisitFreq", "Visit_Freq", "Visit", "VisitFreq", "Visit Frequency", "Visit_Count", "count"]
    visit_col = None
    for cand in visit_candidates:
        if cand in df.columns:
            visit_col = cand
            break
    if visit_col is None:
        # fallback to second column if present
        if len(df.columns) >= 2:
            visit_col = df.columns[1]
    df.attrs['visit_col'] = visit_col
    return df


def build_union_matrix(dfs: Dict[str, pd.DataFrame]):
    obs_set = set()
    actions: List[str] = []
    for df in dfs.values():
        obs_col = [c for c in df.columns if c.lower().startswith('ob')][0]
        obs_set.update(df[obs_col].astype(str).tolist())
        for c in df.columns:
            if c != obs_col and c not in actions:
                actions.append(c)

    all_obs = sorted(obs_set, key=lambda x: x)

    mats: Dict[str, np.ndarray] = {}
    for key, df in dfs.items():
        obs_col = [c for c in df.columns if c.lower().startswith('ob')][0]
        mat = np.zeros((len(all_obs), len(actions)), dtype=float)
        obs_to_row = {o: i for i, o in enumerate(all_obs)}
        for _, row in df.iterrows():
            o = str(row[obs_col])
            if o not in obs_to_row:
                continue
            r = obs_to_row[o]
            for j, a in enumerate(actions):
                if a in df.columns:
                    try:
                        mat[r, j] = float(row[a])
                    except Exception:
                        mat[r, j] = 0.0
        mats[key] = mat

    return all_obs, actions, mats


def _short_label(s: str, width: int) -> str:
    s = str(s)
    if len(s) <= width:
        return s
    half = max(5, width // 2 - 1)
    return s[:half] + '…' + s[-half:]


def _numeric_label(s: str) -> Optional[str]:
    nums = re.findall(r"-?\d+", str(s))
    if nums:
        # format with spaces after commas to match choice_experiment style
        return "(" + ", ".join(nums) + ")"
    return None


def plot_heatmaps(all_obs: List[str], actions: List[str], mats: Dict[str, np.ndarray], outpath: str,
                  max_width: float = 16.0, max_height: float = 40.0,
                  label_width: int = 60, y_numeric: Optional[List[str]] = None):
    n_obs = len(all_obs)
    n_actions = len(actions)

    width = min(max_width, max(6.0, n_actions * 0.5 + 2))
    height = min(max_height, max(4.0, n_obs * 0.22 + 1))

    # Do not share y-axis across subplots; sharing can cause the left labels to be suppressed
    fig, axes = plt.subplots(1, len(mats), figsize=(width, height), sharey=False)
    if len(mats) == 1:
        axes = [axes]

    cmap = sns.color_palette("Blues", as_cmap=True)
    keys = list(mats.keys())

    # Build textual observation labels similar to `choice_experiment.py`:
    # prefer numeric tuple extraction (e.g. "(1,0,0,...)"), otherwise truncate raw string
    ylabels = []
    for o in all_obs:
        nl = _numeric_label(o)
        txt = nl if nl is not None else _short_label(o, label_width)
        ylabels.append(txt)

    # adjust left margin based on label length
    try:
        max_label_len = max(len(l) for l in ylabels) if ylabels else 10
    except Exception:
        max_label_len = 10
    # ensure enough left margin so labels drawn outside the axes are visible
    left_margin = min(0.8, max(0.18, 0.12 + max_label_len * 0.007))
    fig.subplots_adjust(left=left_margin)

    for idx, (ax, key) in enumerate(zip(axes, keys)):
        mat = mats[key]
        annot = (mat.size <= 2000)
        show_y = (idx == 0)

        # draw heatmap; do NOT pass yticklabels to seaborn (we will draw them manually)
        sns.heatmap(mat, ax=ax, cmap=cmap, vmin=0, vmax=1, annot=annot,
                    fmt=".2f" if annot else None, linewidths=0.3, linecolor='white',
                    xticklabels=False, yticklabels=False, cbar=key.endswith('gym'))

        ax.set_title(key, fontsize=10)

        # hide x-axis entirely
        ax.set_xlabel('')
        ax.set_xticks([])

        if show_y:
            try:
                # Ensure ticks align with heatmap rows (centered) but hide matplotlib ticklabels
                nrows = mat.shape[0]
                ticks = np.arange(nrows) + 0.5
                ax.set_yticks(ticks)
                # Set the observation tuple labels directly as yticklabels like choice_experiment
                if len(ylabels) == nrows:
                    ax.set_yticklabels(ylabels, fontsize=9, rotation=0, color='black')
                else:
                    ax.set_yticklabels([str(i) for i in range(nrows)], fontsize=9, rotation=0, color='black')
                ax.yaxis.set_ticks_position('left')
                ax.yaxis.set_label_position('left')
                ax.invert_yaxis()
                ax.tick_params(axis='y', which='major', labelsize=9, pad=6, rotation=0)
                try:
                    ax.set_ylabel(r"Observation", fontsize=11)
                except Exception:
                    pass
            except Exception:
                ax.tick_params(axis='y', labelsize=7)
        else:
            ax.set_yticks([])

                                        # Hide tick labels; we'll draw labels explicitly as text to avoid backend clipping
                                        ax.set_yticklabels([''] * nrows)

def plot_metrics(conformance_csv: str, out_prefix: str, max_width: float = 16.0,
                 col_width: float = 0.12, bar_width: float = 0.8):
    df = pd.read_csv(conformance_csv)
                                        try:
                                            ax.set_ylabel(r"Observation", fontsize=11)
                                        except Exception:
                                            pass
                                        # Draw labels manually to the left of the axis so they are always visible
                                        for i, lbl in enumerate(ylabels):
                                            y_frac = 1.0 - (i + 0.5) / float(max(1, nrows))
                                            ax.text(-0.01, y_frac, lbl, transform=ax.transAxes,
                                                    fontsize=9, va='center', ha='right', color='black', clip_on=False)
            if c in df.columns:
                return c
        return None

                            # Save with tight bbox and ample padding so manually drawn labels aren't clipped
                            fig.savefig(outpath, dpi=150, bbox_inches='tight', pad_inches=0.5)

    obs = df[obs_col].astype(str).tolist()
    x = np.arange(len(obs))
    padding = 2.0
    width = min(max_width, max(4.0, len(obs) * col_width + padding))

    if nratio_col is not None:
        vals = df[nratio_col].astype(float).fillna(0.0).tolist()
        fig, ax = plt.subplots(figsize=(width, 3.5))
        ax.bar(x, vals, width=bar_width, color='#4C72B0')
        ax.set_title('Visit ratio per observation')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(f"{out_prefix}_visit_ratio.png", dpi=150, bbox_inches='tight', pad_inches=0.06)
        plt.close(fig)
        print(f"Saved: {out_prefix}_visit_ratio.png")

    if emac_col is not None:
        vals = df[emac_col].astype(float).fillna(0.0).tolist()
        fig, ax = plt.subplots(figsize=(width, 3.5))
        ax.bar(x, vals, width=bar_width, color='#55A868')
        ax.set_ylim(0, 1)
        ax.set_title('EMAC per observation')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(f"{out_prefix}_emac.png", dpi=150, bbox_inches='tight', pad_inches=0.06)
        plt.close(fig)
        print(f"Saved: {out_prefix}_emac.png")

    if perg_col is not None:
        vals = df[perg_col].astype(float).fillna(0.0).tolist()
        fig, ax = plt.subplots(figsize=(width, 3.5))
        ax.bar(x, vals, width=bar_width, color='#DD8452')
        ax.axhline(0, color='black', linewidth=0.6)
        ax.set_title('PERG per observation')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(f"{out_prefix}_perg.png", dpi=150, bbox_inches='tight', pad_inches=0.06)
        plt.close(fig)
        print(f"Saved: {out_prefix}_perg.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='.', help='Directory containing the CSVs (default: current dir)')
    p.add_argument('--heuristic', help='Path to policy_table_heuristic_bpi.csv (overrides --dir search)')
    p.add_argument('--gym', help='Path to policy_table_gym_bpi.csv (overrides --dir search)')
    p.add_argument('--conformance', help='Path to conformance_results_bpi.csv (overrides --dir search)')
    p.add_argument('--out-prefix', default='policy_heatmaps_bpi_reduced', help='Output prefix for generated PNGs')
    p.add_argument('--max-width', type=float, default=16.0, help='Maximum figure width')
    p.add_argument('--max-height', type=float, default=40.0, help='Maximum figure height')
    p.add_argument('--col-width', type=float, default=0.12, help='Approx inches per observation column for metric plots (smaller => narrower columns)')
    p.add_argument('--bar-width', type=float, default=0.8, help='Bar width in axis units for metric plots (0-1; larger fills column)')
    p.add_argument('--label-width', type=int, default=60, help='Maximum characters per observation label before truncation')
    p.add_argument('--y-source', default='gym', help="Which policy CSV to use for the y-axis numeric scale: 'gym', 'heuristic', or a filename")
    p.add_argument('--open', action='store_true', help='Open generated images after creation (Windows only)')
    args = p.parse_args()

    directory = args.dir
    heuristic_path = args.heuristic or find_file_by_pattern(directory, 'policy_table_heuristic_bpi')
    gym_path = args.gym or find_file_by_pattern(directory, 'policy_table_gym_bpi')
    if heuristic_path is None:
        heuristic_path = find_file_by_pattern(directory, 'policy_table_heuristic')
    if gym_path is None:
        gym_path = find_file_by_pattern(directory, 'policy_table_gym')

    # fallback: any policy_table_*_bpi.csv
    if heuristic_path is None and gym_path is None:
        candidates = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith('policy_table_') and f.endswith('_bpi.csv')]
        if candidates:
            heuristic_path = candidates[0]
            if len(candidates) > 1:
                gym_path = candidates[1]

    conformance_path = args.conformance or find_file_by_pattern(directory, 'conformance_results_bpi')

    dfs: Dict[str, pd.DataFrame] = {}
    if heuristic_path and os.path.exists(heuristic_path):
        dfs['heuristic'] = read_policy_table(heuristic_path)
    if gym_path and os.path.exists(gym_path):
        dfs['gym'] = read_policy_table(gym_path)

    if not dfs:
        print('No policy_table CSVs found in', directory)
        return

    all_obs, actions, mats = build_union_matrix(dfs)

    # Determine y-axis numeric labels from the chosen source (visit counts)
    y_numeric = None
    y_source = args.y_source
    src_df = None
    if y_source in dfs:
        src_df = dfs[y_source]
    else:
        # if user provided a filename, try to read it
        if y_source and os.path.exists(y_source):
            try:
                src_df = read_policy_table(y_source)
            except Exception:
                src_df = None
    if src_df is not None:
        obs_col = src_df.attrs.get('obs_col', None) or next((c for c in src_df.columns if c.lower().startswith('ob')), src_df.columns[0])
        visit_col = src_df.attrs.get('visit_col', None)
        if visit_col is None or visit_col not in src_df.columns:
            # fallback: second column
            visit_col = src_df.columns[1] if len(src_df.columns) > 1 else None
        if visit_col is not None:
            # build mapping obs -> visit count
            obs_vals = src_df[[obs_col, visit_col]].set_index(obs_col)[visit_col].to_dict()
            # align to all_obs order
            y_numeric = [str(int(obs_vals.get(obs, 0))) for obs in all_obs]
            print(f"[DEBUG] Using y-axis numeric values from '{y_source}' column '{visit_col}'. First values: {y_numeric[:10]}")

    heat_out = args.out_prefix + '.png'
    plot_heatmaps(all_obs, actions, mats, heat_out, max_width=args.max_width, max_height=args.max_height, label_width=args.label_width, y_numeric=y_numeric)

    if conformance_path and os.path.exists(conformance_path):
        plot_metrics(conformance_path, out_prefix='metrics_bpi_reduced', max_width=args.max_width, col_width=args.col_width, bar_width=args.bar_width)
    else:
        print('Conformance CSV not found; skipping metric plots.')

    # If running in an IDE on Windows, optionally open the generated PNG files
    if args.open:
        to_open = []
        if os.path.exists(heat_out):
            to_open.append(heat_out)
        for name in [f"metrics_bpi_reduced_visit_ratio.png", f"metrics_bpi_reduced_emac.png", f"metrics_bpi_reduced_perg.png"]:
            if os.path.exists(name):
                to_open.append(name)
        for fpath in to_open:
            try:
                os.startfile(fpath)
            except Exception:
                try:
                    # fallback for other platforms
                    import webbrowser
                    webbrowser.open(fpath)
                except Exception:
                    print(f"Could not open {fpath}")


if __name__ == '__main__':
    main()
