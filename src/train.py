"""End-to-end: load -> fill masked TWS_t -> add neighbourhood features -> select
features -> fit -> evaluate -> predict on Test.csv.

Reproduces the current best validated pipeline. Run with: python -m src.train
"""
from src import config, data, evaluate, features, model


def get_feature_cols(df) -> list[str]:
    available = set(df.columns)
    cols = config.BASE_FEATURE_COLS + [c for c in config.OPTIONAL_FEATURE_COLS if c in available]
    cols += [c for c in config.NEIGHBOURHOOD_FEATURE_COLS if c in available]
    cols += [c for c in config.CLIMATOLOGY_FEATURE_COLS if c in available]
    cols += [c for c in config.SPEI12_DYNAMICS_FEATURE_COLS if c in available]
    if config.ANCHOR_AGE_COL in available:
        cols.append(config.ANCHOR_AGE_COL)
    return cols


def main() -> None:
    raw_train, test, sample_submission = data.load_raw_data()

    train, test_filled = features.build_all_features(raw_train, test)
    still_missing = test_filled[config.TWS_COL].isna().sum()
    print(f"Masked TWS_t rows: {test[config.TWS_MASKED_COL].sum()} "
          f"| still missing after backward-anchor fill: {still_missing}")

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

    print("\n[Horizon-matched split - reference only, still validates on fully-observed "
          "TWS_t, unlike ~66.5% of real Test.csv rows]")
    print("Model:", evaluate.compute_metrics(y_val_h, y_val_h_pred))

    # Mask-aware honest split: additionally simulates Test.csv's real TWS_t masking
    # pattern on the validation fold before running the same feature pipeline used
    # for real Test.csv. Source: notebooks/07_mask_aware_validation.ipynb - closed
    # ~79% of the old proxy's gap to the real leaderboard score. This is now the
    # project's primary internal proxy metric; use it, not the split above, to gate
    # future features whose mechanism is backward-looking/seasonal/spatial. Anything
    # that reads or extrapolates the recent TWS_t trajectory still needs a real
    # Zindi submission before graduating - even this proxy did not reliably predict
    # the trend feature's real-world interaction with climatology (see that
    # notebook's Gate decision).
    masked_month_fraction, masked_row_fraction = evaluate.measure_masking_pattern(test)
    fit_df_m, val_df_m = evaluate.mask_aware_horizon_matched_split(
        raw_train, target_horizons, masked_month_fraction, masked_row_fraction,
    )
    X_fit_m = features.select_base_features(fit_df_m, feature_cols)
    y_fit_m = fit_df_m[config.TARGET_COL].to_numpy()
    X_val_m = features.select_base_features(val_df_m, feature_cols)
    y_val_m = val_df_m[config.TARGET_COL].to_numpy()

    mask_aware_model = model.make_baseline_model()
    mask_aware_model.fit(X_fit_m, y_fit_m)
    y_val_m_pred = model.predict(mask_aware_model, X_val_m)

    print("\n[Mask-aware horizon-matched split - reference only, superseded below: "
          "doesn't augment the FIT half, so months_since_anchor isn't learnable]")
    print("Model:", evaluate.compute_metrics(y_val_m, y_val_m_pred))

    # Mask-augmented honest split: additionally simulates masking on the FIT half
    # (not just validation), so months_since_anchor varies during training instead
    # of being a constant 0. Source: notebooks/08_anchor_age_feature.ipynb - beat
    # the mask-aware split above in 5/5 seeds, and its real-submission prediction
    # (a real leaderboard win, 0.7778 -> 0.7579) held up, unlike the trend feature's
    # proxy-inverting surprise - this is now the project's primary honest proxy.
    # Averaged over 5 masking realisations (not a single seed): a single draw's
    # specific masked-month selection is noisy enough on its own (0.69-0.73 range
    # observed in notebook 08) to occasionally look worse than the split above.
    augmented_feature_cols = None
    augmented_rmses = []
    for seed in range(5):
        fit_df_ma, val_df_ma = evaluate.mask_augmented_horizon_matched_split(
            raw_train, target_horizons, masked_month_fraction, masked_row_fraction,
            fit_seed=seed, val_seed=seed + 100,
        )
        augmented_feature_cols = get_feature_cols(fit_df_ma)
        X_fit_ma = features.select_base_features(fit_df_ma, augmented_feature_cols)
        y_fit_ma = fit_df_ma[config.TARGET_COL].to_numpy()
        X_val_ma = features.select_base_features(val_df_ma, augmented_feature_cols)
        y_val_ma = val_df_ma[config.TARGET_COL].to_numpy()

        mask_augmented_model = model.make_baseline_model()
        mask_augmented_model.fit(X_fit_ma, y_fit_ma)
        y_val_ma_pred = model.predict(mask_augmented_model, X_val_ma)
        augmented_rmses.append(evaluate.rmse(y_val_ma, y_val_ma_pred))

    print("\n[Mask-augmented horizon-matched split - primary honest proxy for the leaderboard, "
          "mean RMSE over 5 masking realisations]")
    print(f"Model: rmse={sum(augmented_rmses) / len(augmented_rmses):.4f} "
          f"(range {min(augmented_rmses):.4f}-{max(augmented_rmses):.4f})")
    print("Feature columns (final fit):", augmented_feature_cols)

    # Final production fit: train on masking-augmented Train.csv (validated above
    # and confirmed on the real leaderboard, 2026-09-04) so months_since_anchor is
    # learnable and the model has seen frozen-anchor examples during fitting, not
    # just at real inference time.
    train_augmented_raw, train_anchor_age = evaluate.augment_with_simulated_masking(
        raw_train, masked_month_fraction, masked_row_fraction, seed=config.RANDOM_STATE,
    )
    test_anchor_age = features.compute_anchor_age(raw_train, test)
    train_final, test_final = features.build_all_features(train_augmented_raw, test)
    train_final[config.ANCHOR_AGE_COL] = train_anchor_age.to_numpy()
    test_final[config.ANCHOR_AGE_COL] = test_anchor_age.to_numpy()

    final_feature_cols = get_feature_cols(train_final)
    final_model = model.make_baseline_model()
    X_train_full = features.select_base_features(train_final, final_feature_cols)
    y_train_full = train_final[config.TARGET_COL].to_numpy()
    final_model.fit(X_train_full, y_train_full)

    X_test = features.select_base_features(test_final, final_feature_cols)
    y_test_pred = model.predict(final_model, X_test)

    output_path = config.OUTPUTS_DIR / "submission.csv"
    data.save_submission(test[config.ID_COL], y_test_pred, output_path)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
