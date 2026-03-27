from __future__ import annotations

import os
from pathlib import Path
import qlib.contrib.model.pytorch_transformer_ts
from config.settings import settings

from alpha_models.workflow.runner import QlibWorkflowRunner

from utils.run_tracker import record_run

def _default_config_path() -> str:
    return str(Path("alpha_models/workflow_config_transformer_Alpha158.yaml"))


def main() -> None:
    config_source = settings.qlib_workflow_config or _default_config_path()

    provider_uri = settings.qlib_provider_uri
    mlruns_uri = settings.qlib_mlruns_uri
    experiment_name = settings.qlib_experiment_name

    runner = QlibWorkflowRunner()
    result = runner.run_from_yaml(
        config_source=config_source,
        provider_uri_override=provider_uri,
        mlruns_uri=mlruns_uri,
        experiment_name=experiment_name,
    )

    payload = {
        "config_source": result.config_source,
        "experiment_id": result.experiment_id,
        "recorder_id": result.recorder_id,
    }
    if result.metrics is not None:
        payload.update(
            {
                "IC": result.metrics.ic,
                "ICIR": result.metrics.icir,
                "Rank IC": result.metrics.rank_ic,
                "Rank ICIR": result.metrics.rank_icir,
            }
        )
    record_run("qlib_train", **payload)


if __name__ == "__main__":
    main()

