"""End-to-end: load -> fill masked TWS_t -> add neighbourhood features -> select
features -> fit -> evaluate -> predict on Test.csv.

Reproduces the current best validated pipeline. Run with: python -m src.train
"""
import pandas as pd

from src import config, data, evaluate, features, model


def get_feature_cols(df) -> list[str]:
    available = set(df.columns)
    cols = config.BASE_FEATURE_COLS + [c for c in config.OPTIONAL_FEATURE_COLS if c in available]
    cols += [c for c in config.NEIGHBOURHOOD_FEATURE_COLS if c in available]
    return cols


def main() -> None:
    train, test, sample_submission = data.load_raw_data()

    test_filled = features.fill_masked_tws(train, test)
    still_missing = test_filled[config.TWS_COL].isna().sum()
    print(f"Masked TWS_t rows: {test[config.TWS_MASKED_COL].sum()} "
          f"| still missing after backward-anchor fill: {still_missing}")

    cell_index = (
        pd.concat([train[[config.LAT_COL, config.LON_COL]], test[[config.LAT_COL, config.LON_COL]]])
        .drop_duplicates()
    )
    cell_id, adjacency, n_cells = features.build_spatial_adjacency(cell_index)
    train = features.add_neighbourhood_features(train, cell_id, adjacency, n_cells)
    test_filled = features.add_neighbourhood_features(test_filled, cell_id, adjacency, n_cells)

    feature_cols = get_feature_cols(train)
    print(f"Feature columns: {feature_cols}")

    # Plain chronological split: cheap, but pools forecast horizons far more evenly
    # than Test.csv actually does - kept only for reference (see the honest metric
    # below for the number that should drive feature-gate decisions).
    fit_df, val_df = evaluate.time_train_val_split(train)
    X_fit = features.select_base_features(fit_df, feature_cols)
    y_fit = fit_df[config.TARGET_COL].to_numpy()
    X_val = features.select_base_features(val_df, feature_cols)
    y_val = val_df[config.TARGET_COL].to_numpy()

    baseline_model = model.make_baseline_model()
    baseline_model.fit(X_fit, y_fit)
    y_val_pred = model.predict(baseline_model, X_val)
    y_val_persist = model.make_persistence_prediction(val_df)

    print("\n[Plain split - optimistic, kept for reference only]")
    print("Model:      ", evaluate.compute_metrics(y_val, y_val_pred))
    print("Persistence:", evaluate.compute_metrics(y_val, y_val_persist))

    # Honest split: restricted to Test.csv's actual forecast-horizon distribution.
    # Source: notebooks/05_leaderboard_gap_investigation.ipynb - the plain split's
    # RMSE (0.5994) badly overestimated real leaderboard performance (0.7822)
    # because it under-weights the long horizons (10-40 months) that make up 78% of
    # Test.csv. Use this number, not the one above, when deciding whether a new
    # feature is worth graduating.
    target_horizons = evaluate.compute_test_horizons(train, test)
    fit_df_h, val_df_h = evaluate.horizon_matched_split(train, target_horizons)
    X_fit_h = features.select_base_features(fit_df_h, feature_cols)
    y_fit_h = fit_df_h[config.TARGET_COL].to_numpy()
    X_val_h = features.select_base_features(val_df_h, feature_cols)
    y_val_h = val_df_h[config.TARGET_COL].to_numpy()

    honest_model = model.make_baseline_model()
    honest_model.fit(X_fit_h, y_fit_h)
    y_val_h_pred = model.predict(honest_model, X_val_h)

    print("\n[Horizon-matched split - honest proxy for the leaderboard]")
    print("Model:", evaluate.compute_metrics(y_val_h, y_val_h_pred))

    final_model = model.make_baseline_model()
    X_train_full = features.select_base_features(train, feature_cols)
    y_train_full = train[config.TARGET_COL].to_numpy()
    final_model.fit(X_train_full, y_train_full)

    X_test = features.select_base_features(test_filled, feature_cols)
    y_test_pred = model.predict(final_model, X_test)

    output_path = config.OUTPUTS_DIR / "submission.csv"
    data.save_submission(test[config.ID_COL], y_test_pred, output_path)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
