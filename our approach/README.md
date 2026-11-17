# Final models and evaluation

- `Improving Rare Postural Transitions.ipynb`: shared model implementations, recorded experiments and final configurations.
- `companion/LDA external.ipynb`: detailed LDA experiments and the recorded predictions for all 3,162 test windows.
- `companion/MLP external.ipynb`: detailed MLP experiments and recorded test predictions.
- `evaluation.ipynb`: evaluation of all six final configurations, with controlled comparisons, verified metrics and error analysis.
- `../doc/evaluation_results.tex`: evaluation-section text to adapt/include in the team's existing LaTeX report. This does not edit the Overleaf report.

## Read or rerun the evaluation

Open `evaluation.ipynb` to read its saved tables and figures. To regenerate them,
install `numpy`, `pandas`, `matplotlib` and `ipython` in your Python notebook
environment, then run all cells from this folder or from the repository root.
Keep the model notebooks, `wrangled data/`, `evaluation_data/` and
`explainability/` at their supplied paths. Evaluation can run after
clearing notebook outputs. The [LR diagnostic archive](evaluation_data/raw_seven_window_lr.json)
stores the raw seven-window test counts, scores and transition errors.

The notebook reads the recorded model results and the saved summaries in
`explainability/`. See [the explainability README](explainability/README.md)
to rerun grouped window occlusion, Integrated Gradients, feature ablation and
the LIME-style probability surrogates on the fixed MLP.

## What the evaluation adds

- Six-model baselines, matched context and oversampling comparisons, context selection, PCA trade-offs and final scores.
- LDA metrics checked against all saved predictions, the recorded confusion counts, the class report and the six errors involving transitions.
- LDA and MLP transition recall comparison and a diagnostic for the raw logistic-regression reference.
- Section 7.7 uses a fixed MLP checkpoint to analyse all 166 true transition windows.

LDA's final transition F1 is 0.9779 versus MLP's 0.9716. This is a comparison
of different retained pipelines on an inspected test set, not a significance
result or an isolated classifier-family effect. LDA has five incorrect true
transition windows and one additional common-to-transition false alarm.

The evaluation uses the official test split as a descriptive comparison.
Section 8 documents the remaining evidence gaps.
