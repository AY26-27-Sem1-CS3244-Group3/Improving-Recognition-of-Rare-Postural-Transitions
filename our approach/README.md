# Final models and evaluation

- `Improving Rare Postural Transitions.ipynb`: shared model implementations, recorded experiments and final configurations.
- `companion/LDA external.ipynb`: detailed LDA experiments and the recorded predictions for all 3,162 test windows.
- `evaluation.ipynb`: Zi Tian's evaluation, continued to include all six final configurations and verified LDA error analysis.
- `../doc/evaluation_results.tex`: evaluation-section text to adapt/include in the team's existing LaTeX report. This does not edit the Overleaf report.

## Read or rerun the evaluation

Open `evaluation.ipynb` to read its saved tables and figures. To regenerate them,
install `numpy`, `pandas`, `matplotlib` and `ipython` in your Python notebook
environment, then run all cells from this folder or from the repository root.
Keep the supplied `wrangled data` folder and both source notebooks at their
existing paths. Jupyter is required to use the notebook interface.

The evaluation reads source notebooks as data. It does not execute their
training code, refit preprocessing, or repeat hyperparameter searches. Each code
cell produces one table or figure. Missing or inconsistent source records stop
execution rather than silently changing a reported result.

## What the evaluation adds

- Six-model baselines, matched context/ROS comparisons, context selection, PCA trade-offs and final scores.
- Correct distinction between the original five weighted models and equal-prior LDA; its weighting entries are unavailable, not zero.
- LDA metrics recomputed from all saved predictions and checked against the companion summary, main notebook's full confusion counts, per-class report and all six transition-involved error cases.
- LDA/MLP transition recall comparison, retaining the existing MLP and raw-LR analyses.

LDA's final transition F1 is 0.9779 versus MLP's 0.9716. This is a comparison
of different retained pipelines on an inspected test set, not a significance
result or an isolated classifier-family effect. LDA has five incorrect true
transition windows and one additional common-to-transition false alarm.

## Evidence still needed from model authors

The current source notebooks do not contain complete final MLP predictions or
its fitted artifact. Its correct/support counts support recall, but do not
identify a full confusion matrix or per-class precision. Existing matched
unweighted/weighted predictions would also be needed for the missing controlled
recall/common-F1 comparisons. These gaps are labelled in evaluation section 8;
no missing predictions or metrics have been invented.

## Validation of this update

All 40 evaluation code cells were executed sequentially with saved outputs and
the notebook passed `nbformat` schema validation. LDA precision, recall, F1,
accuracy and confusion counts were also checked independently against
scikit-learn, without fitting any model. The two model notebooks and datasets
were unchanged.

The LaTeX fragment was compiled in single-column and two-column layouts. Its
results table spans both columns in a two-column report. The fragment still
needs to be included in the team's final report and checked in that document.

Source snapshot: main commit `a34f74496f7d349d5c8481415d92f2f82932c3bb`.

- Main notebook SHA-256: `008c09bf9b50959236f32e927cf5fd4b91d6924bb7d179cffba16e51e89d1388`
- LDA companion SHA-256: `ce5918463c07239071ca491eda1387af4a5a151ddb0a6a241635a9f4d63635f7`

These hashes identify the evaluated snapshot; they are not a restriction on
future edits. The evaluation locates records by variable names and table headers.
