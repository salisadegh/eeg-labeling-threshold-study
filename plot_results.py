"""
Generate all figures for the paper.

Author: Dr Ali Sadegh-Zadeh
Affiliation: Staffordshire University
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_all():
    thresholds  = [4.0, 4.5, 5.0, 5.5, 6.0]
    arousal_acc = [66.25, 71.41, 75.09, 78.27, 81.59]
    arousal_std = [12.39, 11.28, 12.75, 11.10, 10.59]
    valence_acc = [66.64, 73.67, 79.04, 83.57, 84.67]
    valence_std = [10.38,  9.23,  9.43,  9.22,  8.20]
    majority_a  = [63.0, 70.4, 75.4, 79.6, 82.9]
    majority_v  = [62.1, 71.2, 77.7, 84.7, 86.6]
    n_valid     = [32, 32, 31, 26, 22]
    a_high      = [37.0, 29.6, 24.6, 20.4, 17.1]
    v_high      = [37.9, 28.8, 22.3, 15.3, 13.4]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        'Impact of Labeling Threshold on EEG-Based Emotion Recognition\n'
        '(DEAP Dataset, N=32 subjects, SVM-RBF, 5-fold CV)',
        fontsize=12, fontweight='bold')

    ax = axes[0]
    x = np.array(thresholds)
    ax.errorbar(x, arousal_acc, yerr=arousal_std, fmt='o-', color='#2196F3',
                lw=2.5, ms=8, capsize=5, label='Arousal (SVM)', zorder=3)
    ax.errorbar(x, valence_acc, yerr=valence_std, fmt='s-', color='#E91E63',
                lw=2.5, ms=8, capsize=5, label='Valence (SVM)', zorder=3)
    ax.plot(x, majority_a, 'o--', color='#2196F3', lw=1.5, ms=5,
            alpha=0.5, label='Arousal majority baseline')
    ax.plot(x, majority_v, 's--', color='#E91E63', lw=1.5, ms=5,
            alpha=0.5, label='Valence majority baseline')
    ax.axvline(x=5.0, color='orange', ls='--', lw=2, alpha=0.8, label='Standard threshold (5.0)')
    for i, (th, a) in enumerate(zip(thresholds, arousal_acc)):
        ax.annotate(f'n={n_valid[i]}', (th, a - 7), ha='center', fontsize=7.5, color='gray')
    ax.set_xlabel('Labeling Threshold', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Accuracy vs. Threshold', fontsize=11)
    ax.legend(fontsize=7.5, loc='upper left')
    ax.set_ylim(48, 100)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(thresholds)

    ax2 = axes[1]
    width = 0.35
    x2 = np.arange(len(thresholds))
    ax2.bar(x2 - width/2, a_high, width, label='Arousal High%',
            color='#2196F3', alpha=0.8, edgecolor='black', lw=0.5)
    ax2.bar(x2 + width/2, v_high, width, label='Valence High%',
            color='#E91E63', alpha=0.8, edgecolor='black', lw=0.5)
    ax2.axhline(y=50, color='green', ls='--', alpha=0.7, lw=1.5, label='Balanced (50%)')
    ax2.set_xlabel('Labeling Threshold', fontsize=11)
    ax2.set_ylabel('High-Label Percentage (%)', fontsize=11)
    ax2.set_title('Class Distribution', fontsize=11)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([str(t) for t in thresholds])
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 60)
    ax2.grid(True, alpha=0.3, axis='y')

    ax3 = axes[2]
    cats = ['Full\n(T=5.0)', 'No Boundary\n(+/-0.5 removed)']
    av   = [75.08, 76.24]; ae = [12.75, 12.17]
    vv   = [78.91, 81.84]; ve = [9.33,  8.97]
    x3   = np.arange(len(cats))
    ax3.bar(x3-width/2, av, width, yerr=ae, label='Arousal', color='#2196F3',
            alpha=0.8, capsize=5, edgecolor='black', lw=0.5)
    ax3.bar(x3+width/2, vv, width, yerr=ve, label='Valence', color='#E91E63',
            alpha=0.8, capsize=5, edgecolor='black', lw=0.5)
    for i in range(2):
        ax3.text(i-width/2, av[i]+ae[i]+1, f'{av[i]:.1f}%', ha='center', fontsize=9, color='#1565C0')
        ax3.text(i+width/2, vv[i]+ve[i]+1, f'{vv[i]:.1f}%', ha='center', fontsize=9, color='#880E4F')
    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.set_title('Effect of Removing\nBoundary Cases', fontsize=11)
    ax3.set_xticks(x3); ax3.set_xticklabels(cats, fontsize=9)
    ax3.legend(fontsize=9)
    ax3.set_ylim(60, 100)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('../figures/figure1_main_results.png', dpi=300, bbox_inches='tight')
    print("[Saved] figures/figure1_main_results.png")
    plt.close()


if __name__ == '__main__':
    plot_all()
