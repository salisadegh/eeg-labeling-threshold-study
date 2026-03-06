"""
Experiment 3: Effect of Removing Boundary Cases
Removes ambiguous samples (rating within ±0.5 of threshold=5)
and measures accuracy change.

Author: Dr Ali Sadegh-Zadeh
Affiliation: Staffordshire University
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


def run_svm_cv(X, y, n_splits=5, random_state=42):
    if len(np.unique(y)) < 2:
        return None
    min_cls = np.bincount(y).min()
    n_splits = min(n_splits, min_cls)
    if n_splits < 2:
        return None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs = []
    for tr, te in kf.split(Xs, y):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = SVC(kernel='rbf', C=1.0, random_state=random_state)
        clf.fit(Xs[tr], y[tr])
        accs.append(accuracy_score(y[te], clf.predict(Xs[te])))
    return np.mean(accs) if accs else None


def run_experiment(data_path='../data/deap_extracted.npz'):
    data     = np.load(data_path)
    features = data['features']
    labels   = data['labels']

    thresh = 5.0
    margin = 0.5

    acc_a_full, acc_v_full   = [], []
    acc_a_clean, acc_v_clean = [], []

    for subj in range(features.shape[0]):
        X     = features[subj]
        raw_a = labels[subj, :, 1]
        raw_v = labels[subj, :, 0]
        ya    = (raw_a >= thresh).astype(int)
        yv    = (raw_v >= thresh).astype(int)

        r_a = run_svm_cv(X, ya)
        r_v = run_svm_cv(X, yv)
        if r_a: acc_a_full.append(r_a)
        if r_v: acc_v_full.append(r_v)

        mask_a = np.abs(raw_a - thresh) > margin
        mask_v = np.abs(raw_v - thresh) > margin

        if mask_a.sum() >= 10:
            r_a2 = run_svm_cv(X[mask_a], ya[mask_a])
            if r_a2: acc_a_clean.append(r_a2)

        if mask_v.sum() >= 10:
            r_v2 = run_svm_cv(X[mask_v], yv[mask_v])
            if r_v2: acc_v_clean.append(r_v2)

    results = {
        'full_arousal_mean':  np.mean(acc_a_full)  * 100,
        'full_arousal_std':   np.std(acc_a_full)   * 100,
        'full_valence_mean':  np.mean(acc_v_full)  * 100,
        'full_valence_std':   np.std(acc_v_full)   * 100,
        'clean_arousal_mean': np.mean(acc_a_clean) * 100,
        'clean_arousal_std':  np.std(acc_a_clean)  * 100,
        'clean_valence_mean': np.mean(acc_v_clean) * 100,
        'clean_valence_std':  np.std(acc_v_clean)  * 100,
        'diff_arousal': (np.mean(acc_a_clean) - np.mean(acc_a_full)) * 100,
        'diff_valence': (np.mean(acc_v_clean) - np.mean(acc_v_full)) * 100,
    }

    print("=" * 60)
    print("Experiment 3: Boundary Cases Effect")
    print("=" * 60)
    print(f"\nWith boundary cases (n={len(acc_a_full)}):")
    print(f"  Arousal: {results['full_arousal_mean']:.2f}% +/- {results['full_arousal_std']:.2f}")
    print(f"  Valence: {results['full_valence_mean']:.2f}% +/- {results['full_valence_std']:.2f}")
    print(f"\nWithout boundary cases (margin=+/-{margin}, n={len(acc_a_clean)}):")
    print(f"  Arousal: {results['clean_arousal_mean']:.2f}% +/- {results['clean_arousal_std']:.2f}")
    print(f"  Valence: {results['clean_valence_mean']:.2f}% +/- {results['clean_valence_std']:.2f}")
    print(f"\nDifference:")
    print(f"  Arousal: {results['diff_arousal']:+.2f}%")
    print(f"  Valence: {results['diff_valence']:+.2f}%")

    np.save('../results/exp3_results.npy', results)
    print("\n[Saved] results/exp3_results.npy")
    return results


if __name__ == '__main__':
    run_experiment()
