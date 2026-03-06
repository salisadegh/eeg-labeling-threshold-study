# Labeling Threshold Effect on EEG-Based Emotion Recognition

> **Paper:** "How Labeling Threshold Affects Reported Accuracy in EEG-Based Emotion Recognition: A Systematic Empirical Analysis on the DEAP Dataset"
> **Authors:** Dr Ali Sadegh-Zadeh
> **Journal:** PLOS ONE (Under Review)

## Overview

This repository contains the complete source code, experimental results, and figures for our systematic analysis of how the binary labeling threshold choice affects classification accuracy in EEG-based emotion recognition studies using the DEAP dataset.

### Key Findings

| Threshold | Arousal Acc (%) | Valence Acc (%) | Majority Baseline Arousal (%) |
|-----------|-----------------|-----------------|-------------------------------|
| 4.0       | 66.25 +/- 12.39 | 66.64 +/- 10.38 | 63.0                          |
| 4.5       | 71.41 +/- 11.28 | 73.67 +/- 9.23  | 70.4                          |
| **5.0 (standard)** | **75.09 +/- 12.75** | **79.04 +/- 9.43** | **75.4** |
| 5.5       | 78.27 +/- 11.10 | 83.57 +/- 9.22  | 79.6                          |
| 6.0       | 81.59 +/- 10.59 | 84.67 +/- 8.20  | 82.9                          |

**Critical finding:** At the commonly used threshold of 5.0, the majority-class baseline already achieves ~75% for Arousal and ~78% for Valence. A substantial portion of accuracy reported in prior literature may reflect class imbalance rather than genuine emotion discrimination.

## Repository Structure

```
code/
    experiment1_threshold_effect.py    # Accuracy vs labeling threshold
    experiment2_class_distribution.py  # Class imbalance and majority baseline
    experiment3_boundary_cases.py      # Effect of removing boundary samples
    plot_results.py                    # Figure generation
results/
    exp1_results.npy
    exp3_results.npy
figures/
    figure1_main_results.png
data/
    .gitkeep  (DEAP data not included - see below)
```

## Dataset

DEAP dataset (Koelstra et al., 2012): 32 participants, 40 trials, 32 EEG channels at 128 Hz.
Request access at: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html

## Feature Extraction

PSD features from five frequency bands (Delta 1-4 Hz, Theta 4-8 Hz, Alpha 8-13 Hz, Beta 13-30 Hz, Gamma 30-45 Hz) across 32 channels = 160-dimensional feature vector per trial.

## Requirements

```
numpy >= 1.21
scipy >= 1.7
scikit-learn >= 1.0
matplotlib >= 3.4
```

Install: `pip install numpy scipy scikit-learn matplotlib`

## How to Run

```bash
python code/experiment1_threshold_effect.py
python code/experiment2_class_distribution.py
python code/experiment3_boundary_cases.py
python code/plot_results.py
```

## Citation

```bibtex
@article{sadeghzadeh2025threshold,
  title   = {How Labeling Threshold Affects Reported Accuracy in EEG-Based
             Emotion Recognition: A Systematic Empirical Analysis on the DEAP Dataset},
  author  = {Sadegh-Zadeh, Ali},
  journal = {PLOS ONE},
  year    = {2025}
}
```

## Related Work



## License

MIT License. DEAP dataset is subject to its own license terms.
