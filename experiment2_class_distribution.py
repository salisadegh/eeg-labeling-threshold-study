"""
Experiment 2: Class Distribution Analysis Across Thresholds
Shows how threshold choice creates class imbalance that inflates accuracy.

Author: Dr Ali Sadegh-Zadeh
Affiliation: Staffordshire University
"""

import numpy as np


def majority_baseline(y):
    """Accuracy of always-predict-majority classifier."""
    counts = np.bincount(y)
    return counts.max() / len(y) * 100


def run_experiment(data_path='../data/deap_extracted.npz'):
    data   = np.load(data_path)
    labels = data['labels']  # (32, 40, 2)

    thresholds = [4.0, 4.5, 5.0, 5.5, 6.0]

    print("=" * 75)
    print("Experiment 2: Class Distribution and Majority Baseline")
    print("=" * 75)
    print(f"{'Threshold':>10} {'Arousal High%':>14} {'Arousal Imbal%':>15} "
          f"{'Valence High%':>14} {'Valence Imbal%':>15} {'Majority Baseline':>18}")
    print("-" * 75)

    results = {}
    for thresh in thresholds:
        all_a = labels[:, :, 1].flatten()
        all_v = labels[:, :, 0].flatten()

        ya = (all_a >= thresh).astype(int)
        yv = (all_v >= thresh).astype(int)

        a_high_pct   = ya.mean() * 100
        v_high_pct   = yv.mean() * 100
        imbalance_a  = max(a_high_pct, 100 - a_high_pct)
        imbalance_v  = max(v_high_pct, 100 - v_high_pct)
        majority_a   = majority_baseline(ya)
        majority_v   = majority_baseline(yv)

        results[thresh] = {
            'arousal_high_pct':  a_high_pct,
            'valence_high_pct':  v_high_pct,
            'imbalance_arousal': imbalance_a,
            'imbalance_valence': imbalance_v,
            'majority_baseline_arousal': majority_a,
            'majority_baseline_valence': majority_v,
        }

        print(f"{thresh:>10.1f} {a_high_pct:>12.1f}%  {imbalance_a:>13.1f}%  "
              f"{v_high_pct:>12.1f}%  {imbalance_v:>13.1f}%  "
              f"A:{majority_a:.1f}% V:{majority_v:.1f}%")

    print("\nKey Finding:")
    print("  At threshold=5.0, majority baseline for Arousal =",
          f"{results[5.0]['majority_baseline_arousal']:.1f}%")
    print("  At threshold=5.0, majority baseline for Valence =",
          f"{results[5.0]['majority_baseline_valence']:.1f}%")

    np.save('../results/exp2_results.npy', results)
    print("\n[Saved] results/exp2_results.npy")
    return results


if __name__ == '__main__':
    run_experiment()
