"""
Physics-constrained Gaussian data augmentation for H2 adsorption-energy data.

Important
---------
This script is intended to be applied to the TRAINING subset only.
The train/test split should be performed before augmentation so that
the independent test set remains untouched.

The augmentation procedure:
1. Randomly selects an existing DFT-derived training sample as a base sample.
2. Applies small Gaussian perturbations to eligible continuous descriptors.
3. Preserves integer-valued structural descriptors.
4. Keeps fractions within [0, 1].
5. Keeps surface curvature on values observed in the original training data.
6. Restricts sp-carbon fraction to allowed discrete values.
7. Preserves min/max descriptor ordering.
8. Clips continuous descriptors and the target to ranges observed in the
   original training data.

By default, both 4x and 8x augmented training datasets are generated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMN = "E_ads_eV"
RANDOM_STATE = 42
CONTINUOUS_NOISE_FACTOR = 0.04
TARGET_NOISE_STD = 0.04

INTEGER_COLUMNS = [
    "carbon_atoms_count",
    "decoration_atoms_count",
    "substitution_atoms_count",
    "dopant_elements_count",
    "d_electron_count",
    "min_periodic_num",
    "max_periodic_num",
    "min_valence_elec_count",
    "max_valence_elec_count",
]

FRACTION_COLUMNS = [
    "dopant_fraction",
    "dopant_concentration",
    "sp_carbon_fraction",
    "sp-carbon fraction",
    "sp-carbon_fraction",
    "sp carbon fraction",
]

SURFACE_CURVATURE_COLUMN = "surface_curvature"
SP_CARBON_FRACTION_COLUMN = "sp_carbon_fraction"
ALLOWED_SP_CARBON_VALUES = np.array([0.0, 0.5, 1.0])

MIN_MAX_PAIRS = [
    ("min_periodic_num", "max_periodic_num"),
    ("min_valence_elec_count", "max_valence_elec_count"),
    ("min_elec_negativ", "max_elec_negativ"),
    ("min_ionic_radius", "max_ionic_radius"),
    ("min_elec_affinity", "max_elec_affinity"),
]


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load an Excel or CSV dataset."""
    file_path = Path(file_path)

    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)

    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)

    raise ValueError("Input file must be .xlsx, .xls, or .csv")


def nearest_allowed_value(x: float, allowed_values: np.ndarray) -> float:
    """Return the allowed value closest to x."""
    allowed_values = np.asarray(allowed_values, dtype=float)
    return float(allowed_values[np.argmin(np.abs(allowed_values - x))])


def detect_column(
    columns: list[str],
    preferred_name: str,
    required_terms: tuple[str, ...],
) -> str | None:
    """Find a column by exact name or by required terms."""
    if preferred_name in columns:
        return preferred_name

    candidates = [
        col
        for col in columns
        if all(term in col.lower() for term in required_terms)
    ]
    return candidates[0] if candidates else None


def enforce_min_max_pairs(
    data: pd.DataFrame,
    pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    """Ensure each minimum descriptor is <= its corresponding maximum."""
    data = data.copy()

    for min_col, max_col in pairs:
        if min_col in data.columns and max_col in data.columns:
            min_values = data[[min_col, max_col]].min(axis=1)
            max_values = data[[min_col, max_col]].max(axis=1)
            data[min_col] = min_values
            data[max_col] = max_values

    return data


def apply_physical_constraints(
    synthetic_df: pd.DataFrame,
    original_df: pd.DataFrame,
    target_col: str,
    integer_cols: list[str],
    fraction_cols: list[str],
    surface_curvature_col: str | None,
    sp_carbon_fraction_col: str | None,
    allowed_sp_carbon_values: np.ndarray,
    min_max_pairs: list[tuple[str, str]],
    clip_continuous_to_original_range: bool = True,
    clip_target_to_original_range: bool = True,
) -> pd.DataFrame:
    """Apply descriptor-level physical constraints to synthetic samples."""
    synthetic_df = synthetic_df.copy()

    # Integer-valued descriptors
    for col in integer_cols:
        if col in synthetic_df.columns:
            synthetic_df[col] = synthetic_df[col].round().astype(int)
            synthetic_df[col] = synthetic_df[col].clip(lower=0)

    # Fraction descriptors
    for col in fraction_cols:
        if col in synthetic_df.columns:
            synthetic_df[col] = synthetic_df[col].clip(lower=0.0, upper=1.0)

    # Surface curvature must remain on values observed in the real data
    if (
        surface_curvature_col is not None
        and surface_curvature_col in synthetic_df.columns
    ):
        original_surface_values = np.sort(
            original_df[surface_curvature_col]
            .dropna()
            .unique()
            .astype(float)
        )

        if len(original_surface_values) > 0:
            synthetic_df[surface_curvature_col] = synthetic_df[
                surface_curvature_col
            ].apply(
                lambda x: nearest_allowed_value(x, original_surface_values)
            )

    # Restrict sp-carbon fraction to allowed discrete values
    if (
        sp_carbon_fraction_col is not None
        and sp_carbon_fraction_col in synthetic_df.columns
    ):
        synthetic_df[sp_carbon_fraction_col] = synthetic_df[
            sp_carbon_fraction_col
        ].apply(
            lambda x: nearest_allowed_value(x, allowed_sp_carbon_values)
        )

    # Preserve min/max ordering
    synthetic_df = enforce_min_max_pairs(synthetic_df, min_max_pairs)

    # Clip continuous descriptors to ranges observed in the original data
    if clip_continuous_to_original_range:
        numeric_cols = original_df.select_dtypes(include=[np.number]).columns

        protected_cols = set(integer_cols)
        protected_cols.update(fraction_cols)

        if surface_curvature_col is not None:
            protected_cols.add(surface_curvature_col)

        if sp_carbon_fraction_col is not None:
            protected_cols.add(sp_carbon_fraction_col)

        for col in numeric_cols:
            if col == target_col or col not in synthetic_df.columns:
                continue

            if col in protected_cols:
                continue

            min_val = original_df[col].min()
            max_val = original_df[col].max()

            synthetic_df[col] = synthetic_df[col].clip(
                lower=min_val,
                upper=max_val,
            )

    # Clip adsorption energy to the range observed in the original data
    if clip_target_to_original_range:
        y_min = original_df[target_col].min()
        y_max = original_df[target_col].max()

        synthetic_df[target_col] = synthetic_df[target_col].clip(
            lower=y_min,
            upper=y_max,
        )

    return synthetic_df


def gaussian_physics_constrained_augmentation(
    df: pd.DataFrame,
    target_col: str,
    n_synthetic: int,
    feature_cols: list[str],
    integer_cols: list[str],
    fraction_cols: list[str],
    surface_curvature_col: str | None = None,
    sp_carbon_fraction_col: str | None = None,
    allowed_sp_carbon_values: np.ndarray = ALLOWED_SP_CARBON_VALUES,
    min_max_pairs: list[tuple[str, str]] | None = None,
    continuous_noise_factor: float = CONTINUOUS_NOISE_FACTOR,
    target_noise_std: float = TARGET_NOISE_STD,
    clip_continuous_to_original_range: bool = True,
    clip_target_to_original_range: bool = True,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate physics-constrained synthetic samples.

    Continuous descriptor noise:
        sigma = continuous_noise_factor * descriptor standard deviation

    Target noise:
        sigma = target_noise_std (eV)
    """
    rng = np.random.default_rng(random_state)
    min_max_pairs = min_max_pairs or []

    work_cols = feature_cols + [target_col]
    work_df = df[work_cols].dropna().copy()

    if work_df.empty:
        raise ValueError("No complete numeric rows are available for augmentation.")

    feature_std = work_df[feature_cols].std(ddof=0).replace(0, 1e-12)
    synthetic_rows = []

    for _ in range(n_synthetic):
        base_idx = rng.integers(0, len(work_df))
        base_row = work_df.iloc[base_idx].copy()
        new_row = base_row.copy()

        for col in feature_cols:
            # Preserve integer-valued descriptors from the parent sample
            if col in integer_cols:
                continue

            # Preserve surface curvature from the parent sample
            if (
                surface_curvature_col is not None
                and col == surface_curvature_col
            ):
                new_row[col] = base_row[col]
                continue

            sigma = continuous_noise_factor * feature_std[col]
            new_row[col] = base_row[col] + rng.normal(0.0, sigma)

        new_row[target_col] = base_row[target_col] + rng.normal(
            0.0,
            target_noise_std,
        )

        synthetic_rows.append(new_row)

    synthetic_df = pd.DataFrame(synthetic_rows)

    synthetic_df = apply_physical_constraints(
        synthetic_df=synthetic_df,
        original_df=df,
        target_col=target_col,
        integer_cols=integer_cols,
        fraction_cols=fraction_cols,
        surface_curvature_col=surface_curvature_col,
        sp_carbon_fraction_col=sp_carbon_fraction_col,
        allowed_sp_carbon_values=allowed_sp_carbon_values,
        min_max_pairs=min_max_pairs,
        clip_continuous_to_original_range=clip_continuous_to_original_range,
        clip_target_to_original_range=clip_target_to_original_range,
    )

    real_df = df.copy()
    real_df["is_synthetic"] = 0
    synthetic_df["is_synthetic"] = 1

    final_cols = feature_cols + [target_col, "is_synthetic"]

    augmented_df = pd.concat(
        [
            real_df[final_cols],
            synthetic_df[final_cols],
        ],
        axis=0,
        ignore_index=True,
    )

    return synthetic_df, augmented_df


def validate_synthetic_data(
    synthetic_df: pd.DataFrame,
    original_df: pd.DataFrame,
    integer_cols: list[str],
    fraction_cols: list[str],
    surface_curvature_col: str | None,
    sp_carbon_fraction_col: str | None,
    allowed_sp_carbon_values: np.ndarray,
    min_max_pairs: list[tuple[str, str]],
) -> dict[str, bool]:
    """Run basic physical-constraint checks."""
    checks: dict[str, bool] = {}

    for col in integer_cols:
        if col in synthetic_df.columns:
            checks[f"{col}: integer"] = bool(
                np.allclose(
                    synthetic_df[col],
                    synthetic_df[col].round(),
                )
            )

    for col in fraction_cols:
        if col in synthetic_df.columns:
            checks[f"{col}: [0,1]"] = bool(
                synthetic_df[col].between(0, 1).all()
            )

    if (
        surface_curvature_col is not None
        and surface_curvature_col in synthetic_df.columns
    ):
        original_values = set(
            original_df[surface_curvature_col].dropna().unique()
        )
        synthetic_values = set(
            synthetic_df[surface_curvature_col].dropna().unique()
        )

        checks[f"{surface_curvature_col}: observed values only"] = (
            synthetic_values.issubset(original_values)
        )

    if (
        sp_carbon_fraction_col is not None
        and sp_carbon_fraction_col in synthetic_df.columns
    ):
        checks[f"{sp_carbon_fraction_col}: allowed values"] = bool(
            synthetic_df[sp_carbon_fraction_col]
            .isin(allowed_sp_carbon_values)
            .all()
        )

    for min_col, max_col in min_max_pairs:
        if min_col in synthetic_df.columns and max_col in synthetic_df.columns:
            checks[f"{min_col} <= {max_col}"] = bool(
                (synthetic_df[min_col] <= synthetic_df[max_col]).all()
            )

    return checks


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save dataframe based on file extension."""
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_excel(path, index=False)


def run_augmentation(
    df: pd.DataFrame,
    output_dir: Path,
    target_col: str,
    augmentation_factors: tuple[int, ...] = (4, 8),
    random_state: int = RANDOM_STATE,
) -> None:
    """Generate and save augmented datasets."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != target_col]

    integer_cols = [col for col in INTEGER_COLUMNS if col in df.columns]
    fraction_cols = [col for col in FRACTION_COLUMNS if col in df.columns]

    surface_curvature_col = detect_column(
        df.columns.tolist(),
        SURFACE_CURVATURE_COLUMN,
        ("surface", "curvature"),
    )

    sp_carbon_fraction_col = detect_column(
        df.columns.tolist(),
        SP_CARBON_FRACTION_COLUMN,
        ("sp", "carbon", "fraction"),
    )

    if (
        sp_carbon_fraction_col is not None
        and sp_carbon_fraction_col not in fraction_cols
    ):
        fraction_cols.append(sp_carbon_fraction_col)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Original training dataset shape: {df.shape}")
    print(f"Number of numeric features: {len(feature_cols)}")
    print(f"Continuous noise factor: {CONTINUOUS_NOISE_FACTOR}")
    print(f"Target noise standard deviation: {TARGET_NOISE_STD} eV")

    for factor in augmentation_factors:
        if factor < 1:
            raise ValueError("Augmentation factor must be >= 1.")

        # factor = total final size / original size
        n_synthetic = (factor - 1) * len(df)

        synthetic_df, augmented_df = gaussian_physics_constrained_augmentation(
            df=df,
            target_col=target_col,
            n_synthetic=n_synthetic,
            feature_cols=feature_cols,
            integer_cols=integer_cols,
            fraction_cols=fraction_cols,
            surface_curvature_col=surface_curvature_col,
            sp_carbon_fraction_col=sp_carbon_fraction_col,
            allowed_sp_carbon_values=ALLOWED_SP_CARBON_VALUES,
            min_max_pairs=MIN_MAX_PAIRS,
            continuous_noise_factor=CONTINUOUS_NOISE_FACTOR,
            target_noise_std=TARGET_NOISE_STD,
            clip_continuous_to_original_range=True,
            clip_target_to_original_range=True,
            random_state=random_state,
        )

        checks = validate_synthetic_data(
            synthetic_df=synthetic_df,
            original_df=df,
            integer_cols=integer_cols,
            fraction_cols=fraction_cols,
            surface_curvature_col=surface_curvature_col,
            sp_carbon_fraction_col=sp_carbon_fraction_col,
            allowed_sp_carbon_values=ALLOWED_SP_CARBON_VALUES,
            min_max_pairs=MIN_MAX_PAIRS,
        )

        print(f"\n{factor}x augmentation")
        print(f"Synthetic samples: {len(synthetic_df)}")
        print(f"Final augmented dataset shape: {augmented_df.shape}")
        print("Constraint checks:")
        for name, passed in checks.items():
            print(f"  {name}: {passed}")

        synthetic_path = output_dir / f"synthetic_{factor}x.xlsx"
        augmented_path = output_dir / f"augmented_training_{factor}x.xlsx"

        save_dataframe(synthetic_df, synthetic_path)
        save_dataframe(augmented_df, augmented_path)

        print(f"Saved: {synthetic_path}")
        print(f"Saved: {augmented_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physics-constrained Gaussian augmentation for H2 adsorption data."
    )

    parser.add_argument(
        "input_file",
        help="Path to the TRAINING dataset (.xlsx, .xls, or .csv).",
    )

    parser.add_argument(
        "--output-dir",
        default="gaussian_physics_constrained_augmentation",
        help="Directory for augmented datasets.",
    )

    parser.add_argument(
        "--target",
        default=TARGET_COLUMN,
        help=f"Target column name. Default: {TARGET_COLUMN}",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help=f"Random seed. Default: {RANDOM_STATE}",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_dataset(args.input_file)

    run_augmentation(
        df=df,
        output_dir=Path(args.output_dir),
        target_col=args.target,
        augmentation_factors=(4, 8),
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
