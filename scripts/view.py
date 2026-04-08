"""Visualize Qlib experiment results: IC/IR analysis and portfolio returns.

Usage:
    python -m scripts.view
    python -m scripts.view --experiment_id <id> --recorder_id <id>
"""

from __future__ import annotations

import argparse

from model_function.qlib import RecorderIdentity, generate_analysis_view
from runtime.model_state import build_model_runtime_state


def generate_view(
    *,
    experiment_id: str | None = None,
    recorder_id: str | None = None,
    provider_uri: str | None = None,
    mlruns_uri: str | None = None,
) -> str:
    """Generate analysis artifacts through the shared model-function helper."""

    runtime_state = build_model_runtime_state(refresh_settings=True)
    identity = runtime_state.resolve_recorder_identity(
        experiment_id=experiment_id,
        recorder_id=recorder_id,
        allow_settings_fallback=True,
    )
    return generate_analysis_view(
        identity=RecorderIdentity(
            experiment_id=identity.experiment_id,
            recorder_id=identity.recorder_id,
        ),
        provider_uri=provider_uri or runtime_state.settings.qlib_provider_uri,
        analysis_path=runtime_state.settings.analysis_path,
        mlruns_uri=mlruns_uri or runtime_state.settings.qlib_mlruns_uri,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Qlib run artifacts")
    parser.add_argument("--experiment_id", type=str, default=None, help="Qlib experiment id")
    parser.add_argument("--recorder_id", type=str, default=None, help="Qlib recorder id")
    args = parser.parse_args()

    generate_view(
        experiment_id=args.experiment_id,
        recorder_id=args.recorder_id,
    )


if __name__ == "__main__":
    main()
