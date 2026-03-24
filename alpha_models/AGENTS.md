# alpha_models/

ML/DL models for alpha signal generation.

## Files

| File                | Role                                              |
|---------------------|---------------------------------------------------|
| `LSTM.py`           | Bi-LSTM with attention + multi-task loss           |
| `quantTransformer.py` | Transformer-based alpha model                   |
| `qlib_workflow.py`  | Qlib experiment runner (train + evaluate)          |
| `workflow_config_*.yaml` | Qlib workflow configs (Alpha158 / Alpha360) |

## Conventions

- Models are pure PyTorch `nn.Module`. No business logic or I/O inside model classes.
- Loss functions are separate classes (e.g. `MultiTaskLoss`).
- Qlib configs use YAML; model hyperparameters live there, not hardcoded.

## See Also

- `scheduler/model_tasks.py` — calls `train_model` and `predict`
- `data_pipeline/preprocesser.py` — upstream feature engineering
