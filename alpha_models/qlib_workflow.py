from __future__ import annotations

from model_function.qlib import (
    generate_analysis_view,
    RecorderIdentity,
    run_training_workflow,
)
from runtime.model_state import ModelRuntimeState, build_model_runtime_state


def _run_post_train_view(
    runtime_state: ModelRuntimeState,
    *,
    experiment_id: str,
    recorder_id: str,
) -> None:
    """Generate training-analysis artifacts for a successful workflow run."""

    generate_analysis_view(
        identity=RecorderIdentity(experiment_id=experiment_id, recorder_id=recorder_id),
        provider_uri=runtime_state.settings.qlib_provider_uri,
        analysis_path=runtime_state.settings.analysis_path,
        mlruns_uri=runtime_state.settings.qlib_mlruns_uri,
    )


def run_training(*, runtime_state: ModelRuntimeState | None = None):
    """Run the shared Qlib training workflow and return the resolved training result."""

    resolved_runtime_state = runtime_state or build_model_runtime_state(refresh_settings=True)
    workflow_inputs = resolved_runtime_state.resolve_training_workflow_inputs()
    result = run_training_workflow(
        config_source=workflow_inputs.config_source,
        provider_uri=workflow_inputs.provider_uri,
        mlruns_uri=workflow_inputs.mlruns_uri,
        experiment_name=workflow_inputs.experiment_name,
    )
    # Always generate visualization right after successful training.
    _run_post_train_view(
        resolved_runtime_state,
        experiment_id=result.experiment_id,
        recorder_id=result.recorder_id,
    )
    return result


def main() -> None:
    """Run the shared Qlib training workflow via the runtime-owned state boundary."""

    run_training()


if __name__ == "__main__":
    main()

