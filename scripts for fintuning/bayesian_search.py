"""Bayesian Optimization Search and Loss Tracking for LoRA fine-tuning.

Includes publication-ready loss curve tracking, CSV export, and convergence
visualization for research papers.
"""
import csv
import gc
import json
import os
import shutil
from pathlib import Path

# Configure headless matplotlib for server/script environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class ParameterSpec:
    def __init__(self, name, spec):
        self.name = name
        if isinstance(spec, (list, tuple)) and len(spec) == 3 and isinstance(spec[2], str):
            self.kind = spec[2].lower()
            self.low = float(spec[0])
            self.high = float(spec[1])
            self.choices = None
        elif isinstance(spec, (list, tuple)) and len(spec) == 2 and all(isinstance(x, float) for x in spec):
            self.kind = "log" if (spec[0] > 0 and spec[1] / spec[0] >= 10) else "linear"
            self.low = float(spec[0])
            self.high = float(spec[1])
            self.choices = None
        elif isinstance(spec, (list, tuple)):
            self.kind = "categorical"
            self.choices = list(spec)
            self.low = 0
            self.high = len(self.choices) - 1
        else:
            raise ValueError(f"Unsupported parameter specification for '{name}': {spec}")

    def sample_random(self, rng):
        if self.kind == "categorical":
            val = rng.choice(self.choices)
            if hasattr(val, "item"):
                val = val.item()
            return val
        elif self.kind == "log":
            val = 10 ** rng.uniform(np.log10(self.low), np.log10(self.high))
            return float(val)
        else:
            return float(rng.uniform(self.low, self.high))

    def to_normalized(self, val):
        """Map value to [0, 1] for GP."""
        if self.kind == "categorical":
            idx = self.choices.index(val) if val in self.choices else 0
            return idx / max(1, len(self.choices) - 1)
        elif self.kind == "log":
            return (np.log10(val) - np.log10(self.low)) / (np.log10(self.high) - np.log10(self.low) + 1e-12)
        else:
            return (val - self.low) / (self.high - self.low + 1e-12)

    def from_normalized(self, norm_val):
        """Map [0, 1] back to parameter value."""
        norm_val = np.clip(norm_val, 0.0, 1.0)
        if self.kind == "categorical":
            idx = int(round(norm_val * (len(self.choices) - 1)))
            val = self.choices[idx]
            if hasattr(val, "item"):
                val = val.item()
            return val
        elif self.kind == "log":
            log_val = np.log10(self.low) + norm_val * (np.log10(self.high) - np.log10(self.low))
            return float(10 ** log_val)
        else:
            return float(self.low + norm_val * (self.high - self.low))


def expected_improvement(candidates, X_obs, y_obs, gp):
    """Compute Expected Improvement for minimization."""
    mu, sigma = gp.predict(candidates, return_std=True)
    best_y = np.min(y_obs)
    improvement = best_y - mu
    sigma = np.maximum(sigma, 1e-8)
    Z = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei


def save_loss_history_csv(log_history, csv_path):
    """Save training log history to a clean CSV for paper tables and LaTeX pgfplots."""
    rows = []
    for entry in log_history:
        row = {
            "epoch": entry.get("epoch", ""),
            "step": entry.get("step", ""),
            "train_loss": entry.get("loss", ""),
            "eval_loss": entry.get("eval_loss", ""),
            "learning_rate": entry.get("learning_rate", ""),
        }
        if row["train_loss"] != "" or row["eval_loss"] != "":
            rows.append(row)

    if not rows:
        return

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "step", "train_loss", "eval_loss", "learning_rate"])
        writer.writeheader()
        writer.writerows(rows)


def plot_loss_curve(log_history, output_prefix, title="Training & Validation Loss", params=None):
    """Generate publication-ready loss curve plots in PNG and vector PDF."""
    if not log_history:
        return

    train_pts = [(x.get("epoch", x.get("step")), x["loss"]) for x in log_history if "loss" in x and ("epoch" in x or "step" in x)]
    eval_pts = [(x.get("epoch", x.get("step")), x["eval_loss"]) for x in log_history if "eval_loss" in x and ("epoch" in x or "step" in x)]

    if not train_pts and not eval_pts:
        return

    x_label = "Epoch" if any("epoch" in x for x in log_history) else "Step"

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)

    if train_pts:
        tx, ty = zip(*train_pts)
        ax.plot(tx, ty, label="Training Loss", color="#1f77b4", linewidth=2.0, marker=".", markersize=5, alpha=0.9)

    if eval_pts:
        ex, ey = zip(*eval_pts)
        ax.plot(ex, ey, label="Validation Loss", color="#ff7f0e", linewidth=2.4, marker="o", markersize=6)
        min_idx = int(np.argmin(ey))
        best_x, best_y = ex[min_idx], ey[min_idx]
        ax.scatter([best_x], [best_y], color="#d62728", s=110, zorder=6, label=f"Best Val: {best_y:.4f} (@{x_label} {best_x})")
        ax.axhline(best_y, color="#d62728", linestyle=":", alpha=0.5)

    ax.set_xlabel(x_label, fontsize=12, fontweight="semibold")
    ax.set_ylabel("Loss (Cross-Entropy)", fontsize=12, fontweight="semibold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)

    # Subtitle with hyperparameters if provided
    if params:
        subtitle = ", ".join(f"{k}={v:.2e}" if isinstance(v, float) and v < 0.01 else f"{k}={v}" for k, v in params.items())
        ax.set_title(f"{title}\n({subtitle})", fontsize=11, fontweight="bold", pad=10)
    else:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    fig.tight_layout()

    out_p = Path(output_prefix)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_prefix}.png", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)

    # Save corresponding CSV
    save_loss_history_csv(log_history, f"{output_prefix}_history.csv")


def plot_optimization_convergence(trial_numbers, trial_losses, evaluated_params, output_prefix):
    """Plot Bayesian Optimization convergence curve across trials for the paper."""
    if not trial_losses:
        return

    cum_best = np.minimum.accumulate(trial_losses)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    ax.plot(trial_numbers, trial_losses, "o--", color="#1f77b4", linewidth=1.8, markersize=7, label="Trial Val Loss", alpha=0.8)
    ax.step(trial_numbers, cum_best, where="mid", color="#d62728", linewidth=2.5, label="Cumulative Best Val Loss")

    # Annotate global best point
    best_idx = int(np.argmin(trial_losses))
    best_trial = trial_numbers[best_idx]
    best_loss = trial_losses[best_idx]
    ax.scatter([best_trial], [best_loss], color="#2ca02c", s=130, zorder=6, label=f"Optimal Trial {best_trial} ({best_loss:.4f})")

    ax.set_xlabel("Bayesian Search Trial", fontsize=12, fontweight="semibold")
    ax.set_ylabel("Validation Loss", fontsize=12, fontweight="semibold")
    ax.set_title("Bayesian Optimization Hyperparameter Convergence", fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(trial_numbers)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
    fig.tight_layout()

    out_p = Path(output_prefix)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_prefix}.png", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)

    # Save summary CSV for paper table
    summary_path = f"{output_prefix}_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["trial", "val_loss", "cum_best_loss"]
        param_keys = list(evaluated_params[0].keys()) if evaluated_params else []
        fieldnames.extend(param_keys)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (t_num, loss, c_best, p) in enumerate(zip(trial_numbers, trial_losses, cum_best, evaluated_params)):
            row = {"trial": t_num, "val_loss": loss, "cum_best_loss": c_best}
            row.update(p)
            writer.writerow(row)


def bayesian_search_cv(
    train_fn,
    search_space,
    n_trials=5,
    initial_params=None,
    output_dir="best_model",
    tokenizer=None,
    seed=42,
):
    """Run Bayesian Search over hyperparameters minimizing validation loss.

    Args:
        train_fn: Callable(params, trial_dir) -> (val_loss, model) or (val_loss, model, log_history)
        search_space: dict of param_name -> range or choices list
        n_trials: int, total number of Bayesian search trials
        initial_params: dict of starting hyperparameters (optional)
        output_dir: str/Path, where the best model, plots, and CSVs are saved
        tokenizer: Optional tokenizer to save to output_dir
        seed: Random seed

    Returns:
        best_params (dict), best_val_loss (float)
    """
    rng = np.random.RandomState(seed)
    specs = {name: ParameterSpec(name, spec) for name, spec in search_space.items()}
    param_names = list(specs.keys())

    X_obs = []
    y_obs = []
    evaluated_params = []
    best_loss = float("inf")
    best_params = None
    best_log_history = None

    trials_base_dir = Path(f"{output_dir}_trials")
    trials_base_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print(f" Starting Bayesian Search CV ({n_trials} trials)")
    print(f" Search Space: {list(search_space.keys())}")
    print(f" Final Output & Plots Dir: {output_dir}")
    print("=" * 65)

    for trial_idx in range(n_trials):
        # 1. Propose hyperparameters
        if trial_idx == 0 and initial_params is not None:
            candidate_params = dict(initial_params)
        elif len(X_obs) < 2:
            candidate_params = {name: spec.sample_random(rng) for name, spec in specs.items()}
        else:
            gp = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                alpha=1e-5,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=seed + trial_idx,
            )
            X_arr = np.array(X_obs)
            y_arr = np.array(y_obs)
            gp.fit(X_arr, y_arr)

            n_candidates = 500
            cand_norm = rng.uniform(0.0, 1.0, size=(n_candidates, len(param_names)))
            ei = expected_improvement(cand_norm, X_arr, y_arr, gp)
            best_cand_norm = cand_norm[np.argmax(ei)]

            candidate_params = {
                name: specs[name].from_normalized(best_cand_norm[i])
                for i, name in enumerate(param_names)
            }

        x_norm = [specs[name].to_normalized(candidate_params[name]) for name in param_names]

        # 2. Train and evaluate
        trial_dir = trials_base_dir / f"trial_{trial_idx + 1}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Trial {trial_idx + 1}/{n_trials}] Evaluating params: {candidate_params}")

        log_history = None
        try:
            train_result = train_fn(candidate_params, str(trial_dir))
            if isinstance(train_result, tuple):
                if len(train_result) == 3:
                    val_loss, trained_model, log_history = train_result
                else:
                    val_loss, trained_model = train_result[:2]
            else:
                val_loss, trained_model = train_result, None
        except Exception as e:
            print(f"Trial {trial_idx + 1} failed with error: {e}")
            val_loss = float("inf")
            trained_model = None

        X_obs.append(x_norm)
        y_obs.append(val_loss)
        evaluated_params.append(candidate_params)

        print(f"[Trial {trial_idx + 1}/{n_trials}] Val Loss = {val_loss:.4f} (Best so far: {min(y_obs):.4f})")

        # Plot and save this trial's loss curve
        if log_history:
            trial_prefix = str(plots_dir / f"trial_{trial_idx + 1}_loss")
            plot_loss_curve(log_history, trial_prefix, title=f"Trial {trial_idx + 1} Loss Curve", params=candidate_params)
            print(f" Saved trial loss plot: {trial_prefix}.png / .pdf")

        # 3. Check if new best
        if val_loss < best_loss and trained_model is not None:
            best_loss = val_loss
            best_params = candidate_params
            best_log_history = log_history
            print(f"★ New best model found! Saving to {output_dir}...")
            trained_model.save_pretrained(output_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)

            with open(Path(output_dir) / "best_params.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"best_params": best_params, "best_val_loss": best_loss},
                    f,
                    indent=2,
                    default=lambda o: int(o) if isinstance(o, np.integer) else (float(o) if isinstance(o, np.floating) else str(o)),
                )

            # Save best loss curve directly in output_dir for easy paper insertion
            if best_log_history:
                best_prefix = str(Path(output_dir) / "loss_curve")
                plot_loss_curve(best_log_history, best_prefix, title="Best Model Loss Curve", params=best_params)

        # 4. Cleanup trial memory and disk space
        if trained_model is not None:
            del trained_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            shutil.rmtree(trial_dir, ignore_errors=True)
        except Exception:
            pass

    # Cleanup temporary trials directory
    shutil.rmtree(trials_base_dir, ignore_errors=True)

    # Plot final Bayesian Optimization convergence curve and save summary CSV
    conv_prefix = str(Path(output_dir) / "bayesian_optimization_convergence")
    trial_nums = list(range(1, len(y_obs) + 1))
    plot_optimization_convergence(trial_nums, y_obs, evaluated_params, conv_prefix)

    print("\n" + "=" * 65)
    print(f" Bayesian Search Completed!")
    print(f" Best Validation Loss: {best_loss:.4f}")
    print(f" Best Hyperparameters: {best_params}")
    print(f" Best Model & Artifacts saved at: {output_dir}")
    print(f"   ├── loss_curve.png / .pdf (Publication plot)")
    print(f"   ├── loss_curve_history.csv (Raw loss table)")
    print(f"   ├── bayesian_optimization_convergence.png / .pdf")
    print(f"   ├── bayesian_optimization_convergence_summary.csv")
    print(f"   └── plots/ (Per-trial curves)")
    print("=" * 65 + "\n")

    return best_params, best_loss
