"""Diagnostics: adversarial validation + SHAP on bagged CatBoost.

Run modes:
    python diagnostics.py adversarial
    python diagnostics.py shap [--sample 30000]
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from features import CAT_COLS, build_features

warnings.filterwarnings('ignore')
SEED = 42
OUT = Path('diagnostics_out'); OUT.mkdir(exist_ok=True)


def adversarial():
    """Train LGB to classify train vs test. AUC > 0.55 means distribution shift."""
    X, X_te, y, feature_cols, _, _ = build_features()
    print(f'features: {len(feature_cols)}  train: {X.shape}  test: {X_te.shape}')

    combined = pd.concat([X.assign(_is_test=0), X_te.assign(_is_test=1)],
                         ignore_index=True)
    is_test = combined.pop('_is_test').values
    X_tr, X_va, y_tr, y_va = train_test_split(
        combined, is_test, test_size=0.2, random_state=SEED, stratify=is_test,
    )
    params = dict(
        objective='binary', metric='auc',
        learning_rate=0.05, num_leaves=255, min_child_samples=50,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        verbose=-1, seed=SEED, n_jobs=-1,
    )
    t0 = time.time()
    model = lgb.train(
        params,
        lgb.Dataset(X_tr, y_tr, categorical_feature=CAT_COLS),
        2000,
        valid_sets=[lgb.Dataset(X_va, y_va, categorical_feature=CAT_COLS)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    auc = roc_auc_score(y_va, model.predict(X_va, num_iteration=model.best_iteration))
    print(f'\nadversarial AUC: {auc:.5f}  ({time.time()-t0:.0f}s)')
    if auc < 0.55:
        print('  => no meaningful train/test shift')
    elif auc < 0.65:
        print('  => mild drift, check top features')
    else:
        print('  => strong drift, models may not generalise')

    imp = pd.DataFrame({
        'feature': feature_cols,
        'gain': model.feature_importance(importance_type='gain'),
    }).sort_values('gain', ascending=False).head(20)
    print('\ntop 20 features distinguishing train from test:')
    print(imp.to_string(index=False))


def shap_run(sample_n):
    import shap
    from catboost import CatBoostClassifier, Pool

    X, X_te, y, feature_cols, folds, _ = build_features()
    print(f'features: {len(feature_cols)}  train: {X.shape}')

    X_cb = X.copy()
    X_te_cb = X_te.copy()
    for c in CAT_COLS:
        X_cb[c]    = X_cb[c].astype(str).fillna('NA')
        X_te_cb[c] = X_te_cb[c].astype(str).fillna('NA')

    tr_idx, va_idx = folds[0]
    t0 = time.time()
    model = CatBoostClassifier(
        iterations=1200, depth=8, learning_rate=0.08,
        l2_leaf_reg=3.0, bagging_temperature=0.8, random_strength=1.0,
        eval_metric='AUC', cat_features=CAT_COLS, random_seed=SEED,
        early_stopping_rounds=60, verbose=0, task_type='CPU',
    )
    model.fit(X_cb.iloc[tr_idx], y[tr_idx],
              eval_set=(X_cb.iloc[va_idx], y[va_idx]), use_best_model=True)
    print(f'CB fold 0 trained in {time.time()-t0:.0f}s  '
          f'val AUC {roc_auc_score(y[va_idx], model.predict_proba(X_cb.iloc[va_idx])[:, 1]):.5f}')

    rng = np.random.default_rng(SEED)
    sample = rng.choice(va_idx, size=min(sample_n, len(va_idx)), replace=False)
    X_sample = X_cb.iloc[sample]

    t0 = time.time()
    shap_values = model.get_feature_importance(
        Pool(X_sample, cat_features=CAT_COLS),
        type='ShapValues',
    )[:, :-1]
    print(f'SHAP values computed in {time.time()-t0:.0f}s  shape {shap_values.shape}')

    mean_abs = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0),
    }).sort_values('mean_abs_shap', ascending=False)
    mean_abs.to_csv(OUT / 'shap_importance.csv', index=False)
    print('\ntop 25 features by mean |SHAP|:')
    print(mean_abs.head(25).to_string(index=False))

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols,
                      max_display=25, show=False)
    plt.tight_layout()
    plt.savefig(OUT / 'shap_beeswarm.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f'wrote {OUT}/shap_beeswarm.png')

    top3 = mean_abs.head(3)['feature'].tolist()
    for feat in top3:
        if feat in CAT_COLS:
            continue
        try:
            plt.figure(figsize=(7, 5))
            shap.dependence_plot(feat, shap_values, X_sample,
                                 feature_names=feature_cols, show=False)
            plt.tight_layout()
            plt.savefig(OUT / f'shap_dep_{feat}.png', dpi=120, bbox_inches='tight')
            plt.close()
            print(f'wrote {OUT}/shap_dep_{feat}.png')
        except Exception as e:
            print(f'skipped {feat}: {e}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['adversarial', 'shap'])
    parser.add_argument('--sample', type=int, default=30000)
    args = parser.parse_args()
    if args.mode == 'adversarial':
        adversarial()
    else:
        shap_run(args.sample)


if __name__ == '__main__':
    main()
