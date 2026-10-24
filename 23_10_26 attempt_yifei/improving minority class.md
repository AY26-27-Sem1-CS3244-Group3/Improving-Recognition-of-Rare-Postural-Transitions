# Improving Recognition of Rare Postural Transitions

## Background and objective

As expected in our CS3244 proposal, models recognise the majority activities well but perform less well on the rare postural transitions. The HAPT dataset has 12 classes; its six transitions account for only 518 of 10,929 labelled windows. This project therefore tests ways to improve minority-class recognition while monitoring common-activity performance.

We use transition macro-F1 and transition macro-recall as the primary metrics, as specified in the proposal; accuracy is secondary. Higher F1 means better recognition. We report F1 rather than calling 1 − F1 an error rate.

## Methods

All comparisons use participant-grouped validation on the 21 training participants and the same fixed model settings. The three-fold validation uses seed 42. The nine-participant test split is reported separately and has been examined previously in this project.

| Approach | Method |
| --- | --- |
| Three-window context | Concatenate the previous, current, and following 561-feature windows into a 1,683-feature vector. Windows are joined only within the same participant. This is an offline method because the following window is required. |
| Mild class weighting | Give each training example a weight proportional to n_c^−0.5, where n_c is its class count in that training fold; normalise weights to mean 1. This changes each example's contribution to training, not the model architecture. |
| Random oversampling | Add random copies of transition-class examples within the training fold only, until each transition class reaches 60 or 100 rows. Validation and test data are never resampled. We test this as an addition to three-window context plus mild weighting. |

Class weights are passed as sample weights to logistic regression, linear SVM, random forest, and XGBoost. The MLP uses weighted cross-entropy. No model hyperparameter was changed to apply weighting. Fixed settings are: logistic regression C=0.1; linear SVM C=0.01; random forest 200 trees; XGBoost 300 trees, depth 6, learning rate 0.138, subsample 0.616, column sample 0.847, gamma 0.00075, and minimum child weight 7; MLP with 256-128-64 hidden units, Adam at 0.001, and 20 epochs. The comparisons use raw features without PCA.

An earlier PCA comparison reduced 1,683 contextual features to 200 components. It lowered validation transition F1 for logistic regression (0.863 to 0.841), linear SVM (0.865 to 0.813), random forest (0.824 to 0.742), and XGBoost (0.840 to 0.740); the MLP result was nearly unchanged (0.862 to 0.850 at 20 epochs). We therefore use raw features below.

## Results

Each cell reports transition macro-F1. The baseline is one window with no weighting. Weighting-only and context-only isolate each change; the final column combines them.

### Participant-grouped validation

| Model | Baseline | Weighting only (Δ vs baseline) | Three windows only (Δ vs baseline) | Three windows + weights (Δ vs baseline) |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.743 | 0.739 (−0.005) | 0.854 (+0.111) | **0.863 (+0.120)** |
| Linear SVM | 0.737 | 0.745 (+0.008) | 0.850 (+0.114) | **0.865 (+0.128)** |
| Random forest | 0.641 | 0.679 (+0.038) | 0.813 (+0.172) | **0.824 (+0.183)** |
| XGBoost | 0.677 | 0.756 (+0.079) | 0.795 (+0.118) | **0.840 (+0.162)** |
| MLP, 20 epochs | 0.734 | 0.742 (+0.008) | 0.831 (+0.097) | **0.862 (+0.127)** |

Across the five model configurations, the mean transition macro-F1 was 0.707 for the baseline, 0.732 with weighting only, 0.829 with three windows only, and 0.851 with both. The context-only gain over baseline was about 0.122; weighting-only added about 0.026 on average and varied by model. Combining the two gave the best validation result for every model in these folds.

### Held-out test participants

| Model | Baseline | Weighting only (Δ vs baseline) | Three windows only (Δ vs baseline) | Three windows + weights (Δ vs baseline) |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.753 | 0.725 (−0.028) | **0.904 (+0.151)** | 0.897 (+0.143) |
| Linear SVM | 0.811 | 0.781 (−0.031) | **0.882 (+0.071)** | 0.878 (+0.067) |
| Random forest | 0.712 | 0.753 (+0.041) | 0.872 (+0.160) | **0.873 (+0.161)** |
| XGBoost | 0.789 | 0.785 (−0.004) | **0.885 (+0.097)** | 0.846 (+0.058) |
| MLP, 20 epochs | 0.729 | 0.814 (+0.085) | 0.863 (+0.134) | **0.863 (+0.134)** |

Across models, three-window context raised mean test transition macro-F1 by about 0.122 over baseline. Weighting-only changed the mean by about 0.013, with gains for some models and losses for others. The combination was strongest in validation, though weighting did not improve every model on this test split. These results support context as the main source of improvement.

### Oversampling added to the combined approach

The following comparison measures the incremental effect of oversampling on three-window context plus mild weighting. Target 60 means each transition class is brought up to 60 training rows per fold.

| Model | Validation F1 without → with ROS 60 | Test F1 without → with ROS 60 |
| --- | ---: | ---: |
| Logistic regression | 0.863 → 0.864 (+0.001) | 0.897 → 0.897 (+0.000) |
| Linear SVM | 0.865 → 0.867 (+0.002) | 0.878 → 0.878 (+0.000) |
| Random forest | 0.824 → 0.837 (+0.014) | 0.873 → 0.882 (+0.009) |
| XGBoost | 0.840 → 0.836 (−0.004) | 0.846 → 0.859 (+0.012) |
| MLP, 20 epochs | 0.862 → 0.839 (−0.023) | 0.863 → 0.882 (+0.019) |

For random forest, the gain repeated across three validation seeds: mean transition F1 rose from 0.835 to 0.845 across nine paired folds, with improvement in eight folds. On the test split, this came with lower common-class macro-F1 (0.928 to 0.921) and accuracy (0.925 to 0.919). Other models show mixed validation and test results, so oversampling is not a universal improvement. Target 100 did not perform consistently better.

With target 100, validation F1 was 0.867, 0.866, 0.823, 0.819, and 0.835 for logistic regression, linear SVM, random forest, XGBoost, and MLP, respectively, compared with 0.863, 0.865, 0.824, 0.840, and 0.862 without oversampling.

## Final model comparison

The primary comparison is transition recognition for the five models using three-window context plus mild weighting, without oversampling. Validation columns are the seed-42 three-fold means; test columns use the held-out participants.

| Model | Validation transition F1 | Test transition F1 | Test transition recall |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.863 | **0.897** | **0.893** |
| Linear SVM | **0.865** | 0.878 | 0.877 |
| Random forest | 0.824 | 0.873 | 0.866 |
| XGBoost | 0.840 | 0.846 | 0.852 |
| MLP, 20 epochs | 0.862 | 0.863 | 0.871 |

### Transition-class detail for logistic regression

This table compares the one-window unweighted baseline with the selected three-window, mildly weighted model. It shows test F1 for each transition class, so gains are not hidden by the macro average.

| Transition class | Baseline F1 | Three windows + weights F1 | Change |
| --- | ---: | ---: | ---: |
| Stand to sit | 0.800 | 0.978 | +0.178 |
| Sit to stand | 1.000 | 0.952 | −0.048 |
| Sit to lie | 0.635 | 0.813 | +0.178 |
| Lie to sit | 0.745 | 0.906 | +0.161 |
| Stand to lie | 0.686 | 0.833 | +0.147 |
| Lie to stand | 0.653 | 0.898 | +0.245 |

### Secondary overall indicators

For logistic regression, common-class macro-F1 increased from 0.951 to 0.957 and accuracy from 0.939 to 0.953. These metrics are supplementary; transition-class F1 and recall are the main outcomes for this project.

Logistic regression has the highest test transition F1 among the combined models. Linear SVM is slightly ahead on the initial seed-42 validation (0.865 versus 0.863), but the difference is small; across three validation seeds, logistic regression averaged 0.859 compared with 0.846 for linear SVM. The test split has already been viewed during this project, so these test results are descriptive rather than a pristine final evaluation.

## Conclusion

Three-window context provides the clearest and most consistent improvement in transition recognition. Mild class weighting gives smaller, model-dependent gains. Random oversampling to 60 is a promising addition for random forest, with a modest transition-F1 gain and a small trade-off in common-class performance; it should not be applied to every model by default. Among the three-window, weighted models, logistic regression is the best-supported choice for transition recognition.
