# Roadmap: alpha_models/

## 1. Overview

ML/DL models and Qlib workflows for alpha signal generation.

## 3. File-Role Mapping


| File / Subdirectory      | Role / Description                                                              |
| ------------------------ | ------------------------------------------------------------------------------- |
| `LSTM.py`                | Bi-LSTM with attention + multi-task loss                                        |
| `quantTransformer.py`    | Transformer-based alpha model                                                   |
| `qlib_workflow.py`       | Thin workflow entrypoint (`QlibWorkflowRunner`)                                 |
| `workflow/`              | YAML-driven Qlib workflow runner                                                |
| `workflow_config_*.yaml` | Qlib workflow configs (Alpha158 / Alpha360)                                     |


## 5. Navigation


| If you want to...                                     | Go to...                      |
| ----------------------------------------------------- | ----------------------------- |
| Modify the LSTM model                                 | `LSTM.py`                     |
| Modify the Transformer model                          | `quantTransformer.py`         |
| Run Qlib training & evaluation workflow               | `qlib_workflow.py`            |
| Change orchestration runtime flow                     | `workflow/runner.py`          |
| Change model hyperparameters or dataset config        | `workflow_config_transformer_Alpha158.yaml` |
| See how training/prediction is invoked by scheduler   | `../scheduler/model_tasks.py` |


## 6. Conventions

- Models are pure PyTorch `nn.Module`. No business logic or I/O inside model classes.
- Loss functions are separate classes (e.g. `MultiTaskLoss`).
- Qlib configs use YAML; model hyperparameters live there, not hardcoded.

