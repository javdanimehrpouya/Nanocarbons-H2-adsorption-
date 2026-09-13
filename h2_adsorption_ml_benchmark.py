"""
Machine-learning benchmark for H2 adsorption-energy prediction on modified
carbon nanomaterials.

Models
------
- Random Forest
- XGBoost
- CatBoost
- LightGBM
- Elastic Net
- Lasso

Workflow
--------
1. Load separate training and independent holdout-test datasets.
2. Train and evaluate six regression models.
3. Perform 5-fold cross-validation on the training dataset only.
4. Save train/test/CV metrics.
5. Generate actual-vs-predicted figures.
6. Select the best model according to mean CV R2.
7. Run SHAP analysis for the best-performing model.
8. Refit the selected model and optionally predict new structures.

Important
---------
The independent holdout test set is never included in cross-validation.
If augmented training data are used, they should be generated only from the
training portion of the original dataset.
"""

from __future__ import annotations

import argparse
import copy
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------

TARGET_COLUMN = "E_ads_eV"
RANDOM_STATE = 42
CV_FOLDS = 5

PLOT_SETTINGS = {
    "figsize": (6.7, 6.3),
    "composite_figsize": (16, 10),
    "dpi": 700,
    "train_color": "cyan",
    "test_color": "blue",
    "line_color": "#222222",
    "train_marker": "o",
    "test_marker": "s",
    "scatter_size_train": 55,
    "scatter_size_test": 75,
    "scatter_alpha_train": 0.68,
    "scatter_alpha_test": 0.88,
    "line_width": 1.7,
    "line_style": "--",
    "title_size": 15,
    "label_size": 13,
    "tick_size": 10,
    "text_size": 10,
    "legend_size": 10,
    "composite_title_size": 14,
    "composite_label_size": 12,
    "composite_tick_size": 9,
    "composite_text_size": 9,
    "composite_legend_size": 9,
    "grid": False,
    "spine_width": 1.2,
    "save_formats": ("png", "pdf", "svg"),
}


# ---------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------

def load_table(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel table."""
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)

    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)

    raise ValueError("Input data must be a CSV or Excel file.")


def resolve_features(
    train_df: pd.DataFrame,
    target_col: str,
    selected_features: list[str] | None = None,
) -> list[str]:
    """Return numeric model features, or validate a supplied feature list."""
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' was not found.")

    if selected_features is None:
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        excluded = {target_col, "is_synthetic"}
        return [col for col in numeric_cols if col not in excluded]

    missing = [col for col in selected_features if col not in train_df.columns]
    if missing:
        raise ValueError(f"Missing selected features in training file: {missing}")

    return selected_features.copy()


def prepare_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
):
    """Prepare clean training and independent test matrices."""
    required_cols = feature_cols + [target_col]

    missing_test = [col for col in required_cols if col not in test_df.columns]
    if missing_test:
        raise ValueError(f"Missing columns in test file: {missing_test}")

    train_clean = train_df[required_cols].dropna().copy()
    test_clean = test_df[required_cols].dropna().copy()

    X_train = train_clean[feature_cols].copy()
    y_train = train_clean[target_col].copy()

    X_test = test_clean[feature_cols].copy()
    y_test = test_clean[target_col].copy()

    return train_clean, test_clean, X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    """Return R2, MAE, and RMSE."""
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
    }


RMSE_SCORER = make_scorer(
    lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    greater_is_better=False,
)

SCORING = {
    "R2": "r2",
    "MAE": "neg_mean_absolute_error",
    "RMSE": RMSE_SCORER,
}


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

def build_models(random_state: int = RANDOM_STATE) -> dict[str, object]:
    """Construct the regression models used in the benchmark."""
    models: dict[str, object] = {}

    models["Random Forest"] = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=700,
            learning_rate=0.02,
            max_depth=6,
            subsample=0.55,
            colsample_bytree=0.55,
            reg_alpha=0.01,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    except ImportError:
        print("XGBoost is not installed; skipping XGBoost.")

    try:
        from catboost import CatBoostRegressor

        models["CatBoost"] = CatBoostRegressor(
            iterations=700,
            learning_rate=0.01,
            depth=4,
            loss_function="RMSE",
            random_seed=random_state,
            verbose=0,
        )
    except ImportError:
        print("CatBoost is not installed; skipping CatBoost.")

    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=600,
            learning_rate=0.01,
            max_depth=-1,
            num_leaves=15,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.01,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        print("LightGBM is not installed; skipping LightGBM.")

    models["Lasso"] = Pipeline(
        [
            ("scaler", RobustScaler()),
            (
                "lasso",
                LassoCV(
                    alphas=np.logspace(-5, 1, 80),
                    cv=5,
                    max_iter=20000,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    models["Elastic Net"] = Pipeline(
        [
            ("scaler", RobustScaler()),
            (
                "elastic_net",
                ElasticNetCV(
                    l1_ratio=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0],
                    alphas=np.logspace(-5, 2, 120),
                    cv=5,
                    max_iter=50000,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return models


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def set_closed_frame(ax, spine_width: float = 1.2) -> None:
    """Show all four plot spines."""
    for spine in ("top", "bottom", "left", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(spine_width)


def get_common_limits(y_train, y_train_pred, y_test, y_test_pred):
    """Return common plotting limits for actual and predicted values."""
    actual = np.concatenate([np.asarray(y_train), np.asarray(y_test)])
    predicted = np.concatenate(
        [np.asarray(y_train_pred), np.asarray(y_test_pred)]
    )

    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())

    span = max_val - min_val
    padding = 0.06 * span if span > 0 else 0.1

    return min_val - padding, max_val + padding


def draw_actual_vs_predicted_on_axis(
    ax,
    y_train,
    y_train_pred,
    y_test,
    y_test_pred,
    model_name: str,
    test_metrics: dict[str, float],
    style: dict,
    composite: bool = False,
) -> None:
    """Draw one actual-vs-predicted panel."""
    if composite:
        title_size = style["composite_title_size"]
        label_size = style["composite_label_size"]
        tick_size = style["composite_tick_size"]
        text_size = style["composite_text_size"]
        legend_size = style["composite_legend_size"]
    else:
        title_size = style["title_size"]
        label_size = style["label_size"]
        tick_size = style["tick_size"]
        text_size = style["text_size"]
        legend_size = style["legend_size"]

    ax.scatter(
        y_train,
        y_train_pred,
        s=style["scatter_size_train"],
        alpha=style["scatter_alpha_train"],
        color=style["train_color"],
        marker=style["train_marker"],
        label="Train",
    )

    ax.scatter(
        y_test,
        y_test_pred,
        s=style["scatter_size_test"],
        alpha=style["scatter_alpha_test"],
        color=style["test_color"],
        marker=style["test_marker"],
        label="Test",
    )

    min_lim, max_lim = get_common_limits(
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
    )

    ax.plot(
        [min_lim, max_lim],
        [min_lim, max_lim],
        color=style["line_color"],
        linewidth=style["line_width"],
        linestyle=style["line_style"],
        label="Ideal",
    )

    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)
    ax.set_xlabel(r"Actual $E_{\mathrm{ads}}$ (eV)", fontsize=label_size)
    ax.set_ylabel(r"Predicted $E_{\mathrm{ads}}$ (eV)", fontsize=label_size)
    ax.set_title(model_name, fontsize=title_size, pad=10)

    metric_text = (
        f"Test $R^2$ = {test_metrics['R2']:.3f}\n"
        f"Test MAE = {test_metrics['MAE']:.3f} eV\n"
        f"Test RMSE = {test_metrics['RMSE']:.3f} eV"
    )

    ax.text(
        0.05,
        0.95,
        metric_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=text_size,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 0.55,
            "alpha": 0.82,
        },
    )

    ax.legend(fontsize=legend_size, frameon=False, loc="lower right")
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.grid(alpha=0.25) if style["grid"] else ax.grid(False)
    set_closed_frame(ax, style["spine_width"])


def save_figure(fig, output_base: Path, style: dict) -> None:
    """Save a figure in all configured formats."""
    for fmt in style["save_formats"]:
        fig.savefig(
            output_base.with_suffix(f".{fmt}"),
            dpi=style["dpi"],
            bbox_inches="tight",
        )


def plot_single_actual_vs_predicted(
    y_train,
    y_train_pred,
    y_test,
    y_test_pred,
    model_name: str,
    test_metrics: dict[str, float],
    output_dir: Path,
    style: dict,
) -> None:
    """Create and save an actual-vs-predicted plot for one model."""
    fig, ax = plt.subplots(figsize=style["figsize"])

    draw_actual_vs_predicted_on_axis(
        ax=ax,
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_test=y_test,
        y_test_pred=y_test_pred,
        model_name=model_name,
        test_metrics=test_metrics,
        style=style,
        composite=False,
    )

    fig.tight_layout()

    safe_name = model_name.replace(" ", "_").replace("/", "_")
    save_figure(
        fig,
        output_dir / f"actual_vs_predicted_{safe_name}",
        style,
    )
    plt.close(fig)


def plot_composite_actual_vs_predicted(
    predictions: dict,
    model_order: list[str],
    y_train,
    y_test,
    output_dir: Path,
    style: dict,
) -> None:
    """Create and save the six-model composite figure."""
    fig, axes = plt.subplots(2, 3, figsize=style["composite_figsize"])
    axes = axes.flatten()

    for ax, model_name in zip(axes, model_order):
        pred = predictions[model_name]

        draw_actual_vs_predicted_on_axis(
            ax=ax,
            y_train=y_train,
            y_train_pred=pred["y_train_pred"],
            y_test=y_test,
            y_test_pred=pred["y_test_pred"],
            model_name=model_name,
            test_metrics=pred["test_metrics"],
            style=style,
            composite=True,
        )

    for idx in range(len(model_order), len(axes)):
        fig.delaxes(axes[idx])

    fig.tight_layout()

    save_figure(
        fig,
        output_dir / "actual_vs_predicted_all_models_2x3",
        style,
    )
    plt.close(fig)


# ---------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------

def run_benchmark(
    models: dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
):
    """Train, evaluate, and cross-validate all models."""
    cv = KFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    results = []
    trained_models = {}
    predictions = {}

    for model_name, model in models.items():
        print(f"\nTraining: {model_name}")

        try:
            # Cross-validation uses training data only.
            cv_scores = cross_validate(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring=SCORING,
                n_jobs=-1,
                return_train_score=False,
            )

            cv_r2_mean = float(cv_scores["test_R2"].mean())
            cv_r2_std = float(cv_scores["test_R2"].std())

            cv_mae_mean = float(-cv_scores["test_MAE"].mean())
            cv_mae_std = float(cv_scores["test_MAE"].std())

            cv_rmse_mean = float(-cv_scores["test_RMSE"].mean())
            cv_rmse_std = float(cv_scores["test_RMSE"].std())

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_metrics = calculate_metrics(y_train, y_train_pred)
            test_metrics = calculate_metrics(y_test, y_test_pred)

            results.append(
                {
                    "Model": model_name,
                    "Train_R2": train_metrics["R2"],
                    "Train_MAE": train_metrics["MAE"],
                    "Train_RMSE": train_metrics["RMSE"],
                    "Test_R2": test_metrics["R2"],
                    "Test_MAE": test_metrics["MAE"],
                    "Test_RMSE": test_metrics["RMSE"],
                    "CV_R2_mean": cv_r2_mean,
                    "CV_R2_std": cv_r2_std,
                    "CV_MAE_mean": cv_mae_mean,
                    "CV_MAE_std": cv_mae_std,
                    "CV_RMSE_mean": cv_rmse_mean,
                    "CV_RMSE_std": cv_rmse_std,
                }
            )

            trained_models[model_name] = model
            predictions[model_name] = {
                "y_train_pred": y_train_pred,
                "y_test_pred": y_test_pred,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
            }

            print("Train metrics:", train_metrics)
            print("Test metrics:", test_metrics)
            print(f"CV R2: {cv_r2_mean:.3f} ± {cv_r2_std:.3f}")
            print(f"CV MAE: {cv_mae_mean:.3f} ± {cv_mae_std:.3f}")
            print(f"CV RMSE: {cv_rmse_mean:.3f} ± {cv_rmse_std:.3f}")

            plot_single_actual_vs_predicted(
                y_train=y_train,
                y_train_pred=y_train_pred,
                y_test=y_test,
                y_test_pred=y_test_pred,
                model_name=model_name,
                test_metrics=test_metrics,
                output_dir=output_dir,
                style=PLOT_SETTINGS,
            )

        except Exception as exc:
            print(f"{model_name} failed: {exc}")

    if not results:
        raise RuntimeError("No model completed successfully.")

    results_df = (
        pd.DataFrame(results)
        .sort_values("CV_R2_mean", ascending=False)
        .reset_index(drop=True)
    )

    results_df.to_excel(
        output_dir / "model_performance_train_test_cv.xlsx",
        index=False,
    )

    preferred_order = [
        "Random Forest",
        "XGBoost",
        "CatBoost",
        "LightGBM",
        "Elastic Net",
        "Lasso",
    ]

    model_order = [
        model_name
        for model_name in preferred_order
        if model_name in predictions
    ]

    plot_composite_actual_vs_predicted(
        predictions=predictions,
        model_order=model_order,
        y_train=y_train,
        y_test=y_test,
        output_dir=output_dir,
        style=PLOT_SETTINGS,
    )

    best_model_name = str(results_df.iloc[0]["Model"])
    best_model = trained_models[best_model_name]

    print("\nBest model based on mean CV R2:", best_model_name)

    return (
        results_df,
        trained_models,
        predictions,
        best_model_name,
        best_model,
    )


# ---------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------

def run_shap_analysis(
    best_model_name: str,
    best_model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
    random_state: int = RANDOM_STATE,
) -> None:
    """Run SHAP interpretation for the selected model."""
    try:
        import shap
    except ImportError:
        print("SHAP is not installed; skipping SHAP analysis.")
        return

    shap_dir = output_dir / "shap_best_model"
    shap_dir.mkdir(parents=True, exist_ok=True)

    max_background = min(100, len(X_train))
    max_explain = min(150, len(X_test))

    X_background = X_train.sample(
        n=max_background,
        random_state=random_state,
    )

    X_explain = X_test.sample(
        n=max_explain,
        random_state=random_state,
    )

    safe_name = best_model_name.replace(" ", "_").replace("/", "_")

    tree_models = {
        "Random Forest",
        "XGBoost",
        "CatBoost",
        "LightGBM",
    }

    if best_model_name in tree_models:
        explainer = shap.Explainer(best_model, X_background)
        shap_values = explainer(X_explain)
    else:
        def predict_function(input_data):
            input_df = pd.DataFrame(input_data, columns=feature_cols)
            return best_model.predict(input_df)

        explainer = shap.Explainer(
            predict_function,
            X_background,
        )
        shap_values = explainer(X_explain)

    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    fig = plt.gcf()
    plt.title(f"SHAP beeswarm - {best_model_name}", fontsize=14)
    plt.tight_layout()
    save_figure(fig, shap_dir / f"shap_beeswarm_{safe_name}", PLOT_SETTINGS)
    plt.close(fig)

    shap.plots.bar(shap_values, max_display=20, show=False)
    fig = plt.gcf()
    plt.title(f"SHAP feature importance - {best_model_name}", fontsize=14)
    plt.tight_layout()
    save_figure(fig, shap_dir / f"shap_bar_{safe_name}", PLOT_SETTINGS)
    plt.close(fig)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    shap_importance_df = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "mean_abs_SHAP": mean_abs_shap,
            }
        )
        .sort_values("mean_abs_SHAP", ascending=False)
        .reset_index(drop=True)
    )

    shap_importance_df.to_excel(
        shap_dir / "shap_feature_importance.xlsx",
        index=False,
    )


# ---------------------------------------------------------------------
# Prediction of new structures
# ---------------------------------------------------------------------

def check_applicability_domain(
    new_X: pd.DataFrame,
    feature_min: pd.Series,
    feature_max: pd.Series,
) -> pd.DataFrame:
    """Check whether descriptors fall within training min-max ranges."""
    rows = []

    for idx in new_X.index:
        for col in new_X.columns:
            value = new_X.loc[idx, col]
            min_val = feature_min[col]
            max_val = feature_max[col]

            if value < min_val or value > max_val:
                rows.append(
                    {
                        "sample_index": idx,
                        "feature": col,
                        "new_value": value,
                        "training_min": min_val,
                        "training_max": max_val,
                    }
                )

    return pd.DataFrame(rows)


def classify_adsorption(energy: float) -> str:
    """Return a simple adsorption-strength label."""
    if energy < -0.6:
        return "strong adsorption"
    if energy <= -0.2:
        return "moderate adsorption"
    return "weak adsorption"


def refit_and_save_best_model(
    best_model_name: str,
    models: dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: list[str],
    output_dir: Path,
):
    """
    Refit the selected model on the complete training dataset and save it.

    The independent test set is intentionally excluded from refitting so that
    it remains an external evaluation set.
    """
    prediction_dir = output_dir / "new_structure_prediction"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    final_model = copy.deepcopy(models[best_model_name])
    final_model.fit(X_train, y_train)

    joblib.dump(
        final_model,
        prediction_dir / "final_best_model.joblib",
    )

    pd.DataFrame({"feature": feature_cols}).to_excel(
        prediction_dir / "feature_list.xlsx",
        index=False,
    )

    return final_model, prediction_dir


def predict_new_structures(
    final_model,
    new_structures_file: str | Path,
    feature_cols: list[str],
    X_train: pd.DataFrame,
    prediction_dir: Path,
) -> None:
    """Predict adsorption energies for structures supplied in CSV/Excel."""
    new_df = load_table(new_structures_file)

    missing_cols = [col for col in feature_cols if col not in new_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in new-structure file: {missing_cols}"
        )

    new_X = new_df[feature_cols].copy()
    predictions = final_model.predict(new_X)

    new_df["predicted_E_ads_eV"] = predictions
    new_df["predicted_adsorption_class"] = [
        classify_adsorption(value) for value in predictions
    ]

    feature_min = X_train.min()
    feature_max = X_train.max()

    applicability_summary = []
    all_warnings = []

    for idx in new_X.index:
        sample = new_X.loc[[idx]]

        warnings_df = check_applicability_domain(
            sample,
            feature_min,
            feature_max,
        )

        if warnings_df.empty:
            applicability_summary.append("OK")
        else:
            outside_features = warnings_df["feature"].tolist()
            applicability_summary.append(
                "Outside: " + ", ".join(outside_features)
            )
            all_warnings.append(warnings_df)

    new_df["applicability_domain"] = applicability_summary

    new_df.to_excel(
        prediction_dir / "new_structures_predictions.xlsx",
        index=False,
    )

    if all_warnings:
        pd.concat(all_warnings, ignore_index=True).to_excel(
            prediction_dir / "applicability_domain_warnings.xlsx",
            index=False,
        )


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_feature_list(value: str | None) -> list[str] | None:
    """Parse a comma-separated feature list."""
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ML models for H2 adsorption-energy prediction."
        )
    )

    parser.add_argument(
        "--train",
        required=True,
        help="Training dataset (.xlsx, .xls, or .csv).",
    )

    parser.add_argument(
        "--test",
        required=True,
        help="Independent holdout-test dataset (.xlsx, .xls, or .csv).",
    )

    parser.add_argument(
        "--output-dir",
        default="ml_benchmark_outputs",
        help="Directory for generated results.",
    )

    parser.add_argument(
        "--target",
        default=TARGET_COLUMN,
        help=f"Target column name. Default: {TARGET_COLUMN}",
    )

    parser.add_argument(
        "--features",
        default=None,
        help=(
            "Optional comma-separated feature names. If omitted, all numeric "
            "columns except the target and is_synthetic are used."
        ),
    )

    parser.add_argument(
        "--new-structures",
        default=None,
        help=(
            "Optional CSV/Excel file containing new structures for prediction."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help=f"Random seed. Default: {RANDOM_STATE}",
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=CV_FOLDS,
        help=f"Number of CV folds. Default: {CV_FOLDS}",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_table(args.train)
    test_df = load_table(args.test)

    selected_features = parse_feature_list(args.features)

    feature_cols = resolve_features(
        train_df=train_df,
        target_col=args.target,
        selected_features=selected_features,
    )

    (
        train_clean,
        test_clean,
        X_train,
        y_train,
        X_test,
        y_test,
    ) = prepare_data(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=args.target,
    )

    print("Training dataset shape:", train_clean.shape)
    print("Test dataset shape:", test_clean.shape)
    print("Number of features:", len(feature_cols))

    models = build_models(random_state=args.seed)

    (
        results_df,
        trained_models,
        predictions,
        best_model_name,
        best_model,
    ) = run_benchmark(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        output_dir=output_dir,
        cv_folds=args.cv_folds,
        random_state=args.seed,
    )

    print("\nModel performance:")
    print(results_df)

    run_shap_analysis(
        best_model_name=best_model_name,
        best_model=best_model,
        X_train=X_train,
        X_test=X_test,
        feature_cols=feature_cols,
        output_dir=output_dir,
        random_state=args.seed,
    )

    final_model, prediction_dir = refit_and_save_best_model(
        best_model_name=best_model_name,
        models=models,
        X_train=X_train,
        y_train=y_train,
        feature_cols=feature_cols,
        output_dir=output_dir,
    )

    if args.new_structures:
        predict_new_structures(
            final_model=final_model,
            new_structures_file=args.new_structures,
            feature_cols=feature_cols,
            X_train=X_train,
            prediction_dir=prediction_dir,
        )


if __name__ == "__main__":
    main()
