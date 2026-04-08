"""Evaluate a trained Qlib model on the full test segment.

Usage:
  python -m scripts.eval_test
  python -m scripts.eval_test --config alpha_models/workflow_config_transformer_Alpha158.yaml

Model selection:
  - If env QLIB_EXPERIMENT_ID + QLIB_RECORDER_ID are set, use them.
  - Else fallback to `.data/run_history.json` -> `qlib_train`.
"""

from __future__ import annotations

import argparse

from model_function.qlib import evaluate_test_predictions
from runtime.model_state import NO_TRAINED_MODEL_ERROR, build_model_runtime_state


def main() -> None:
    """Print full-test-segment metrics for the currently selected trained model."""

    runtime_state = build_model_runtime_state(refresh_settings=True)
    parser = argparse.ArgumentParser(description="Evaluate trained Qlib model on full test segment")
    parser.add_argument(
        "--config",
        type=str,
        default=runtime_state.settings.qlib_workflow_config,
        help="Workflow YAML config path (single file).",
    )
    args = parser.parse_args()

    workflow_inputs = runtime_state.resolve_training_workflow_inputs(config_source=args.config)
    identity = runtime_state.resolve_recorder_identity(
        missing_error_message=NO_TRAINED_MODEL_ERROR,
    )
    metrics = evaluate_test_predictions(
        config_source=workflow_inputs.config_source,
        identity=identity,
        provider_uri=workflow_inputs.provider_uri,
        mlruns_uri=workflow_inputs.mlruns_uri,
    )
    print("\n== Full test segment metrics ==")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
