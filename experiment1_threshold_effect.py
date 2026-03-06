"""
Experiment 1: Effect of Labeling Threshold on EEG-Based Emotion Recognition Accuracy
Dataset: DEAP (Koelstra et al., 2012)
Author: Dr Ali Sadegh-Zadeh
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')


def run_svm_cv(X, y, n_splits=5, random_state=42):
    """Run SVM with stratified k-fold cross-validation."""
    if len(np.unique(y)) < 2:
        return None, None
    min_cls = np.bincount(y).min()
    n_splits = min(n_splits, min_cls)
    if n_splits < 2:
        return None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accs, bal_accs = [], []
    for tr, te in kf.split(X_scaled, y):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = SVC(kernel='rbf', C=1.0, random_state=random_state)
        clf.fit(X_scaled[tr], y[tr])
        pred = clf.predict(X_scaled[te])
        accs.append(accuracy_score(y[te], pred))
        bal_accs.append(balanced_accuracy_score(y[te], pred))

    if not accs:
        return None, None
    return np.mean(accs), np.mean(bal_accs)


def run_experiment(data_path='../data/deap_extracted.npz'):
    data = np.load(data_path)
    features = data['features']  # (32, 40, 160)
    labels   = data['labels']    # (32, 40, 2): col0=valence, col1=arousal

    thresholds = [4.0, 4.5, 5.0, 5.5, 6.0]
    results = {}

    print("=" * 70)
    print("Experiment 1: Labeling Threshold Effect on Classification Accuracy")
    print("=" * 70)
    print(f"{'Threshold':>10} {'n_valid':>8} {'Arousal Acc':>12} {'Arousal Bal':>12} {'Valence Acc':>12} {'Valence Bal':>12}")
    print("-" * 70)

    for thresh in thresholds:
        acc_a, acc_v, bal_a, bal_v = [], [], [], []
        n_valid = 0

        for subj in range(features.shape[0]):
            X  = features[subj]
            ya = (labels[subj, :, 1] >= thresh).astype(int)
            yv = (labels[subj, :, 0] >= thresh).astype(int)

            r_a, rb_a = run_svm_cv(X, ya)
            r_v, rb_v = run_svm_cv(X, yv)

            if r_a is not None and r_v is not None:
                acc_a.append(r_a); bal_a.append(rb_a)
                acc_v.append(r_v); bal_v.append(rb_v)
                n_valid += 1

        results[thresh] = {
            'arousal_acc_mean': np.mean(acc_a) * 100,
            'arousal_acc_std':  np.std(acc_a)  * 100,
            'arousal_bal_mean': np.mean(bal_a) * 100,
            'valence_acc_mean': np.mean(acc_v) * 100,
            'valence_acc_std':  np.std(acc_v)  * 100,
            'valence_bal_mean': np.mean(bal_v) * 100,
            'n_valid': n_valid,
        }
        r = results[thresh]
        print(f"{thresh:>10.1f} {n_valid:>8d} "
              f"{r['arousal_acc_mean']:>10.2f}%  {r['arousal_bal_mean']:>10.2f}%  "
              f"{r['valence_acc_mean']:>10.2f}%  {r['valence_bal_mean']:>10.2f}%")

    np.save('../results/exp1_results.npy', results)
    print("\n[Saved] results/exp1_results.npy")
    return results


if __name__ == '__main__':
    run_experiment()
