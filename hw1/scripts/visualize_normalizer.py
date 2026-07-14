"""Visualize the Push-T dataset before and after Normalizer is applied.

Run with:
    uv run scripts/visualize_normalizer.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hw1_imitation.data import Normalizer, download_pusht, load_pusht_zarr

DATA_DIR = Path("data")
OUTPUT_DIR = Path("scripts/output")

STATE_LABELS = ["agent_x", "agent_y", "block_x", "block_y", "block_angle"]
ACTION_LABELS = ["target_x", "target_y"]


def plot_dim_histograms(
    raw: np.ndarray, normed: np.ndarray, labels: list[str], title: str, path: Path
) -> None:
    num_dims = raw.shape[1]
    fig, axes = plt.subplots(2, num_dims, figsize=(3.2 * num_dims, 6))

    for dim in range(num_dims):
        axes[0, dim].hist(raw[:, dim], bins=80, color="tab:blue")
        axes[0, dim].set_title(labels[dim])
        if dim == 0:
            axes[0, dim].set_ylabel("before (raw)")

        axes[1, dim].hist(normed[:, dim], bins=80, color="tab:orange")
        if dim == 0:
            axes[1, dim].set_ylabel("after (normalized)")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_action_scatter(raw: np.ndarray, normed: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    axes[0].scatter(raw[:, 0], raw[:, 1], s=1, alpha=0.2, color="tab:blue")
    axes[0].set_title("before (raw action space)")
    axes[0].set_xlabel(ACTION_LABELS[0])
    axes[0].set_ylabel(ACTION_LABELS[1])
    axes[0].set_aspect("equal")

    axes[1].scatter(normed[:, 0], normed[:, 1], s=1, alpha=0.2, color="tab:orange")
    axes[1].set_title("after (normalized action space)")
    axes[1].set_xlabel(ACTION_LABELS[0])
    axes[1].set_ylabel(ACTION_LABELS[1])
    axes[1].set_aspect("equal")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    zarr_path = download_pusht(DATA_DIR)
    states, actions, _ = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    normed_states = normalizer.normalize_state(states)
    normed_actions = normalizer.normalize_action(actions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_dim_histograms(
        states,
        normed_states,
        STATE_LABELS,
        "State dimensions before/after normalization",
        OUTPUT_DIR / "state_normalization.png",
    )
    plot_dim_histograms(
        actions,
        normed_actions,
        ACTION_LABELS,
        "Action dimensions before/after normalization",
        OUTPUT_DIR / "action_normalization.png",
    )
    plot_action_scatter(
        actions, normed_actions, OUTPUT_DIR / "action_scatter.png"
    )

    print("\nPer-dimension stats:")
    print(f"  state_mean:  {normalizer.state_mean}")
    print(f"  state_std:   {normalizer.state_std}")
    print(f"  action_mean: {normalizer.action_mean}")
    print(f"  action_std:  {normalizer.action_std}")


if __name__ == "__main__":
    main()
