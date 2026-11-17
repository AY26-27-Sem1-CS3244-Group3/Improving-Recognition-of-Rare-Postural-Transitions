# Local explainability of the selected MLP

Evaluation §7.7 reads the CSV summaries in this folder. To regenerate them with
the fixed checkpoint, run from the repository root:

```sh
python "our approach/explainability/analyze.py"
python "our approach/explainability/local_surrogates.py"
```

Both scripts use the supplied HAPT CSVs and a scaler fitted on the training
inputs. `analyze.py` checks the final labels against the MLP companion and
recomputes window occlusion, Integrated Gradients and the two feature-ablation
cases. It retains the reported summaries and computes feature attributions in
memory. `local_surrogates.py` uses the targets and anchor probabilities in
`surrogate_cases.csv`. It selects the sparse fits on validation donors and
evaluates them on audit donors. The larger row 51 analysis is exploratory.

Dependencies: NumPy, pandas, SciPy, PyTorch, scikit-learn and threadpoolctl.
The recorded run used Python 3.10, NumPy 1.26.4, PyTorch 2.10.0 and
scikit-learn 1.7.2. The MLP is loaded from `mlp_fixed_refit.pt`; both scripts
leave model weights intact.
