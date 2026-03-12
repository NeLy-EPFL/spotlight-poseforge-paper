"""Generate a trajectory comparison plot for every trial saved by simple_optim.py."""

import optuna
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sppaper.common.resources import get_outputs_dir
from sppaper.kinematics.visualize import plot_trajectory

OPTIM_DIR = get_outputs_dir() / "neuromechfly_replay/hparam_optim/"
STUDY_NAME = "trajmse_walkingsnippet21"
WALKING_PERIOD_TIMERANGE = (0.5, 2.5)
OUTPUT_DIR = OPTIM_DIR / "trajplots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_ROT = 180  # just to traj in a neater orientation for aesthetics, no data affected

storage_url = f"sqlite:///{OPTIM_DIR / 'optuna_study.db'}"
study = optuna.load_study(study_name=STUDY_NAME, storage=storage_url)

trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print(f"Plotting {len(trials)} completed trials...")

for trial in trials:
    trial_dir = OPTIM_DIR / f"trial{trial.number:03d}"
    if not (trial_dir / "sim_data.pkl").exists():
        print(f"  Skipping trial {trial.number:03d} — sim_data.pkl not found")
        continue

    fig, _ = plot_trajectory(
        sim_dir=trial_dir, t_range=WALKING_PERIOD_TIMERANGE, base_rot_deg=BASE_ROT
    )
    fig.suptitle(
        f"Trial {trial.number:03d}  |  loss={trial.value:.4f}\n"
        + "  ".join(f"{k}={v:.3f}" for k, v in trial.params.items()),
        fontsize=6,
    )
    fig.savefig(OUTPUT_DIR / f"trial{trial.number:03d}.png")
    plt.close(fig)
    print(f"  Saved trial {trial.number:03d}  loss={trial.value:.4f}")

print(f"\nDone. Plots saved to {OUTPUT_DIR}")
