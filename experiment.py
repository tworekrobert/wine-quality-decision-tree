"""
Wine Quality: decision tree experiment.

Data mining part of the group project for "Hurtownie i eksploracja danych"
(Wroclaw University of Science and Technology).

What the script does, step by step:
  1. Loads the UCI Wine Quality data (red + white wines).
  2. Removes repeated rows (exact copies of the same wine).
  3. Turns the 0-10 quality score into two classes: good (>= 7) and not good.
  4. Keeps 20% of the wines aside as a final test set.
  5. Runs a grid search over two decision tree hyperparameters.
     Every combination is scored with 10-fold cross-validation.
  6. Checks the best tree on the test set it has never seen.
  7. Saves tables to results/ and charts to figures/.

Run:  python experiment.py
"""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")  # draw charts to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

RANDOM_STATE = 42      # fixed seed, so every run gives the same numbers
GOOD_THRESHOLD = 7     # quality >= 7 means "good wine"
TEST_SIZE = 0.20       # 20% of wines kept aside for the final check
CV_FOLDS = 10          # 10-fold cross-validation

# The grid: every combination of these values is tried (6 x 4 = 24 trees).
PARAM_GRID = {
    "max_depth": [2, 3, 5, 8, 12, None],      # None = no depth limit
    "min_samples_leaf": [1, 5, 10, 20],
}

# Chart colours
WINE = "#7f1d1d"
WINE_LIGHT = "#e7b8b8"
GREY = "#6b7280"


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Read both CSV files (they use ';' as separator), join them, drop copies."""
    red = pd.read_csv(DATA_DIR / "winequality-red.csv", sep=";")
    white = pd.read_csv(DATA_DIR / "winequality-white.csv", sep=";")
    red["wine_type"] = "red"
    white["wine_type"] = "white"
    wines = pd.concat([red, white], ignore_index=True)

    # About 18% of the rows are exact copies of another row. A copy that lands
    # in both the training and the test part lets the tree "remember" the
    # answer, so the score looks better than it really is. Keep one of each.
    wines = wines.drop_duplicates().reset_index(drop=True)

    # The target: 1 = good wine, 0 = not good.
    wines["good"] = (wines["quality"] >= GOOD_THRESHOLD).astype(int)
    return wines


# ---------------------------------------------------------------------------
# 2. Grid search with cross-validation
# ---------------------------------------------------------------------------

def run_grid_search(X_train, y_train) -> GridSearchCV:
    # Stratified = every fold keeps the same share of good wines (about 20%).
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid=PARAM_GRID,
        cv=folds,
        scoring={
            "f1": "f1",
            "accuracy": "accuracy",
            "recall": "recall",
            # zero_division=0: a tree that never says "good" gets precision 0
            "precision": make_scorer(precision_score, zero_division=0),
        },
        refit="f1",               # the winner is the tree with the best F1
        return_train_score=True,  # needed to show overfitting
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def results_table(search: GridSearchCV) -> pd.DataFrame:
    """Turn scikit-learn's raw results into a readable table."""
    raw = pd.DataFrame(search.cv_results_)
    table = pd.DataFrame({
        "max_depth": raw["param_max_depth"].apply(lambda d: "none" if d is None else int(d)),
        "min_samples_leaf": raw["param_min_samples_leaf"].astype(int),
        "f1_validation": raw["mean_test_f1"],
        "f1_validation_std": raw["std_test_f1"],
        "f1_train": raw["mean_train_f1"],
        "accuracy_validation": raw["mean_test_accuracy"],
        "recall_validation": raw["mean_test_recall"],
        "precision_validation": raw["mean_test_precision"],
        "rank": raw["rank_test_f1"],
    })
    return table.sort_values("rank").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Charts
# ---------------------------------------------------------------------------

def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#374151")


def plot_quality_distribution(wines: pd.DataFrame):
    counts = wines["quality"].value_counts().sort_index()
    colours = [WINE if q >= GOOD_THRESHOLD else WINE_LIGHT for q in counts.index]
    share_good = wines["good"].mean()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts.index, counts.values, color=colours)
    ax.axvline(GOOD_THRESHOLD - 0.5, color=GREY, linestyle="--", linewidth=1)
    ax.text(GOOD_THRESHOLD - 0.4, counts.max() * 0.9,
            f"good wine (>= {GOOD_THRESHOLD})\n{share_good:.1%} of all wines",
            color=WINE, fontsize=10)
    ax.set_xlabel("Quality score given by experts")
    ax.set_ylabel("Number of wines")
    ax.set_title("Most wines are average, only about one in five is good", loc="left")
    ax.set_xticks(counts.index)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "quality_distribution.png", dpi=150)
    plt.close(fig)


def plot_grid_heatmap(table: pd.DataFrame):
    depth_order = [str(d) if d is not None else "none" for d in PARAM_GRID["max_depth"]]
    leaf_order = PARAM_GRID["min_samples_leaf"]
    t = table.copy()
    t["max_depth"] = t["max_depth"].astype(str)
    grid = t.pivot(index="max_depth", columns="min_samples_leaf", values="f1_validation")
    grid = grid.loc[depth_order, leaf_order]

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(grid.values, cmap="Reds", aspect="auto")
    best = np.unravel_index(np.nanargmax(grid.values), grid.shape)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid.values[i, j]
            dark = value > np.nanmean(grid.values)
            ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                    color="white" if dark else "#111827",
                    fontweight="bold" if (i, j) == best else "normal")
    ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1,
                               fill=False, edgecolor="#111827", linewidth=2))
    ax.set_xticks(range(len(leaf_order)), leaf_order)
    ax.set_yticks(range(len(depth_order)), depth_order)
    ax.set_xlabel("min_samples_leaf")
    ax.set_ylabel("max_depth")
    ax.set_title("F1 in 10-fold cross-validation for every combination", loc="left")
    fig.colorbar(image, ax=ax, label="F1 (validation)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "grid_search_heatmap.png", dpi=150)
    plt.close(fig)


def plot_overfitting(table: pd.DataFrame):
    """Train vs validation F1 for each depth, with min_samples_leaf = 1."""
    t = table[table["min_samples_leaf"] == 1].copy()
    order = [str(d) if d is not None else "none" for d in PARAM_GRID["max_depth"]]
    t["max_depth"] = t["max_depth"].astype(str)
    t = t.set_index("max_depth").loc[order]

    x = range(len(order))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, t["f1_train"], marker="o", color=GREY, label="training data")
    ax.plot(x, t["f1_validation"], marker="o", color=WINE, linewidth=2.5,
            label="validation data (not seen in training)")
    ax.fill_between(x,
                    t["f1_validation"] - t["f1_validation_std"],
                    t["f1_validation"] + t["f1_validation_std"],
                    color=WINE, alpha=0.12, linewidth=0)
    ax.set_xticks(list(x), order)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("max_depth (min_samples_leaf = 1)")
    ax.set_ylabel("F1 for good wines")
    ax.set_title("Deeper trees learn the training data by heart", loc="left")
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "overfitting_curve.png", dpi=150)
    plt.close(fig)


def plot_tree_top(model, feature_names):
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(model, max_depth=3, feature_names=feature_names,
              class_names=["not good", "good"], filled=True, rounded=True,
              impurity=False, proportion=True, fontsize=9, ax=ax)
    ax.set_title("Best decision tree: first three levels", loc="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "best_tree_top.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(model, feature_names):
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    colours = [WINE if v == importance.max() else WINE_LIGHT for v in importance.values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance.index, importance.values, color=colours)
    for i, v in enumerate(importance.values):
        ax.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=9, color="#374151")
    ax.set_xlabel("Share of the tree's decisions (feature importance)")
    ax.set_title(f"{importance.idxmax().capitalize()} matters most", loc="left")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def plot_confusion(cm):
    labels = ["not good", "good"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cm, cmap="Reds")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "#111827")
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted by the tree")
    ax.set_ylabel("Actual (experts)")
    ax.set_title("Confusion matrix on the test set", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    wines = load_data()
    feature_names = [c for c in wines.columns if c not in ("quality", "good", "wine_type")]
    X = wines[feature_names]
    y = wines["good"]
    print(f"Unique wines: {len(wines)} ({(wines.wine_type == 'red').sum()} red, "
          f"{(wines.wine_type == 'white').sum()} white), good: {y.mean():.1%}")

    # Final test set, kept aside until the very end.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    search = run_grid_search(X_train, y_train)
    table = results_table(search)
    table.to_csv(RESULTS_DIR / "grid_search_results.csv", index=False, float_format="%.4f")

    best = search.best_estimator_
    y_pred = best.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    test_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    # A "model" that always says "not good" - shows why accuracy alone misleads.
    baseline_accuracy = 1 - y_test.mean()

    summary = {
        "wines_total": int(len(wines)),
        "wines_red": int((wines.wine_type == "red").sum()),
        "wines_white": int((wines.wine_type == "white").sum()),
        "share_good": round(float(y.mean()), 4),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "combinations_tested": int(len(table)),
        "cv_folds": CV_FOLDS,
        "best_params": {k: (v if v is not None else "none") for k, v in search.best_params_.items()},
        "best_f1_cross_validation": round(float(search.best_score_), 4),
        "test": {k: round(float(v), 4) for k, v in test_metrics.items()},
        "test_confusion_matrix": {
            "true_not_good_pred_not_good": int(cm[0, 0]),
            "true_not_good_pred_good": int(cm[0, 1]),
            "true_good_pred_not_good": int(cm[1, 0]),
            "true_good_pred_good": int(cm[1, 1]),
        },
        "baseline_always_not_good_accuracy": round(float(baseline_accuracy), 4),
        "best_tree_depth": int(best.get_depth()),
        "best_tree_leaves": int(best.get_n_leaves()),
        "feature_importance": {
            name: round(float(v), 4)
            for name, v in sorted(zip(feature_names, best.feature_importances_),
                                  key=lambda p: -p[1])
        },
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (RESULTS_DIR / "best_tree_rules.txt").write_text(
        export_text(best, feature_names=feature_names, max_depth=4))

    plot_quality_distribution(wines)
    plot_grid_heatmap(table)
    plot_overfitting(table)
    plot_tree_top(best, feature_names)
    plot_feature_importance(best, feature_names)
    plot_confusion(cm)

    print(f"Best settings: {summary['best_params']}  "
          f"(F1 in cross-validation: {search.best_score_:.3f})")
    print("Test set: " + ", ".join(f"{k} {v:.3f}" for k, v in test_metrics.items()))
    print(f"Always saying 'not good' would give accuracy {baseline_accuracy:.3f}")
    print("Saved tables to results/ and charts to figures/")


if __name__ == "__main__":
    main()
