"""Generate a trajectory comparison plot for every trial saved by simple_optim.py,
plus a single ensemble plot overlaying all simulated trials against the recording,
plus a parameter scatter matrix colored by trajectory mismatch loss."""

from sppaper.common.plot import setup_matplotlib_params

setup_matplotlib_params()

import optuna
import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns

from sppaper.common.resources import get_outputs_dir
from sppaper.kinematics.visualize import plot_trajectory, plot_trajectory_ensemble

OPTIM_DIR = get_outputs_dir() / "neuromechfly_replay/hparam_optim/"
STUDY_NAME = "trajmse_walkingsnippet21"
WALKING_PERIOD_TIMERANGE = (0.5, 2.5)
OUTPUT_DIR = OPTIM_DIR / "trajplots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_ROT = (
    180  # just to put traj in a neater orientation for aesthetics, no data affected
)

FAILED_COLOR = "#a23e48"
SCATTER_PARAMS = [
    "params_actuator_gain",
    "params_adhforce_fmlegs",
    "params_adhforce_hlegs_scale",
    "params_joint_damping",
    "params_sliding_friction",
]
SCATTER_LOSS_VMIN, SCATTER_LOSS_VMAX = 0, 120

storage_url = f"sqlite:///{OPTIM_DIR / 'optuna_study.db'}"
study = optuna.load_study(study_name=STUDY_NAME, storage=storage_url)

trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print(f"Plotting {len(trials)} completed trials...")

# --- Per-trial trajectory plots ---
valid_trial_dirs = []
for trial in trials:
    trial_dir = OPTIM_DIR / f"trial{trial.number:03d}"
    if not (trial_dir / "sim_data.pkl").exists():
        print(f"  Skipping trial {trial.number:03d} — sim_data.pkl not found")
        continue

    fig, _ = plot_trajectory(sim_dir=trial_dir, t_range=WALKING_PERIOD_TIMERANGE, base_rot_deg=BASE_ROT)
    fig.suptitle(
        f"Trial {trial.number:03d}  |  loss={trial.value:.4f}\n"
        + "  ".join(f"{k}={v:.3f}" for k, v in trial.params.items()),
        fontsize=6,
    )
    fig.savefig(OUTPUT_DIR / f"trial{trial.number:03d}.pdf")
    plt.close(fig)
    print(f"  Saved trial {trial.number:03d}  loss={trial.value:.4f}")
    valid_trial_dirs.append(trial_dir)

# --- Ensemble plot ---
print(f"\nGenerating ensemble plot for {len(valid_trial_dirs)} trials...")
fig, _ = plot_trajectory_ensemble(
    sim_dirs=valid_trial_dirs,
    t_range=WALKING_PERIOD_TIMERANGE,
    base_rot_deg=BASE_ROT,
    traj_xlim=(-3, 17),
    traj_ylim=(-3, 17),
    linspeed_ylim=(0, 25),
    turnrate_ylim=(-2.5, 2.5),
)
fig.suptitle(
    f"Ensemble: {len(valid_trial_dirs)} simulated trials vs. recording",
    fontsize=7,
)
fig.savefig(OUTPUT_DIR / "ensemble.pdf")
plt.close(fig)

# --- Parameter scatter matrix ---
print("Generating parameter scatter matrix...")

MM_TO_IN = 1 / 25.4
norm = mcolors.Normalize(vmin=SCATTER_LOSS_VMIN, vmax=SCATTER_LOSS_VMAX)
cmap = cm.viridis_r

df = study.trials_dataframe()
within = df["value"] <= SCATTER_LOSS_VMAX
exceeded = ~within


def fmt_label(col):
    return col.replace("params_", "").replace("_", " ")


n = len(SCATTER_PARAMS)
fig, axes = plt.subplots(n, n, figsize=(130 * MM_TO_IN, 130 * MM_TO_IN))

for i, row_param in enumerate(SCATTER_PARAMS):
    for j, col_param in enumerate(SCATTER_PARAMS):
        ax = axes[i, j]

        if j >= i:
            ax.set_visible(False)
            continue

        colors = cmap(norm(df.loc[within, "value"]))
        ax.scatter(df.loc[within, col_param], df.loc[within, row_param], c=colors, s=2)
        ax.scatter(
            df.loc[exceeded, col_param],
            df.loc[exceeded, row_param],
            c=FAILED_COLOR,
            s=8,
            marker="x",
            linewidths=0.5,
        )

        sns.despine(ax=ax)

        if i == n - 1:
            ax.set_xlabel(fmt_label(col_param))
        else:
            ax.set_xticklabels([])

        if j == 0:
            ax.set_ylabel(fmt_label(row_param))
        else:
            ax.set_yticklabels([])

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(
    sm,
    ax=axes,
    orientation="vertical",
    fraction=0.02,
    pad=0.02,
    label="trajectory mismatch",
)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "param_scatter_matrix.pdf")
plt.close(fig)

print(f"\nDone. Plots saved to {OUTPUT_DIR}")
