from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import yaml


@dataclass(frozen=True)
class LoadedWorkflowConfig:
    qlib_init: Dict[str, Any]
    market: str
    benchmark: str
    task: Dict[str, Any]


@dataclass(frozen=True)
class TrainMetrics:
    ic: float
    icir: float
    rank_ic: float
    rank_icir: float


@dataclass(frozen=True)
class TrainResult:
    experiment_id: str
    recorder_id: str
    metrics: Optional[TrainMetrics]
    config_source: str


def _deep_merge_dict(base: Dict[str, Any], inc: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in inc.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge_dict(out[k], v)
        else:
            out[k] = v
    return out


class QlibWorkflowRunner:
    @staticmethod
    def _expand_provider_uri(provider_uri: Optional[str]) -> Optional[str]:
        if not provider_uri:
            return None
        return str(Path(provider_uri).expanduser())

    @staticmethod
    def _load_yaml_file(path: Path) -> Dict[str, Any]:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    @staticmethod
    def _iter_config_files(source: str) -> Tuple[str, Iterable[Path]]:
        """
        - If source is a directory: merge all *.yaml / *.yml files (sorted by name).
        - If source contains commas: merge each path in order.
        - Else: load single file.
        """
        raw = source.strip()
        if "," in raw:
            paths = [Path(p.strip()).expanduser() for p in raw.split(",") if p.strip()]
            return raw, paths
        p = Path(raw).expanduser()
        if p.is_dir():
            files = sorted(list(p.glob("*.yaml")) + list(p.glob("*.yml")))
            return str(p), files
        return str(p), [p]

    @classmethod
    def load_yaml_config(cls, config_source: str) -> LoadedWorkflowConfig:
        source_label, files = cls._iter_config_files(config_source)
        merged: Dict[str, Any] = {}
        for f in files:
            merged = _deep_merge_dict(merged, cls._load_yaml_file(f))

        return LoadedWorkflowConfig(
            qlib_init=merged.get("qlib_init") or {},
            market=merged.get("market") or "all",
            benchmark=merged.get("benchmark") or "",
            task=merged.get("task") or {},
        )

    @staticmethod
    def _as_date_str(dt: Any) -> str:
        s = str(dt)
        return s[:10] if len(s) >= 10 else s

    @staticmethod
    def _safe_backtest_end_time(*, start_time: str, end_time: str, freq: str = "day") -> str:
        """
        Qlib backtest TradeCalendar uses calendar_index+1 internally for end boundary.
        If end_time hits the last available calendar date, it may raise IndexError.
        """
        from qlib.data import D

        cal = D.calendar(start_time=start_time, end_time=end_time, freq=freq)
        if cal is None or len(cal) == 0:
            return end_time
        if len(cal) < 2:
            return QlibWorkflowRunner._as_date_str(cal[-1])
        return QlibWorkflowRunner._as_date_str(cal[-2])

    @classmethod
    def _patch_port_ana_config_for_calendar(cls, cfg: Any) -> Any:
        if not isinstance(cfg, dict):
            return cfg
        backtest = cfg.get("backtest")
        if not isinstance(backtest, dict):
            return cfg
        start_time = backtest.get("start_time")
        end_time = backtest.get("end_time")
        if not start_time or not end_time:
            return cfg
        backtest["end_time"] = cls._safe_backtest_end_time(
            start_time=str(start_time),
            end_time=str(end_time),
            freq=str(backtest.get("freq") or "day"),
        )
        return cfg

    @staticmethod
    def _extract_sig_metrics(recorder) -> Optional[TrainMetrics]:
        import numpy as np

        try:
            ic = recorder.load_object("sig_analysis/ic.pkl")
            ric = recorder.load_object("sig_analysis/ric.pkl")
        except Exception:
            return None

        ic_mean = float(np.nanmean(ic))
        ic_std = float(np.nanstd(ic))
        ric_mean = float(np.nanmean(ric))
        ric_std = float(np.nanstd(ric))
        return TrainMetrics(
            ic=ic_mean,
            icir=(ic_mean / ic_std) if ic_std else float("nan"),
            rank_ic=ric_mean,
            rank_icir=(ric_mean / ric_std) if ric_std else float("nan"),
        )

    def run_from_yaml(
        self,
        *,
        config_source: str,
        provider_uri_override: Optional[str] = None,
        mlruns_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> TrainResult:
        import qlib
        from qlib.utils import init_instance_by_config
        from qlib.workflow import R

        loaded = self.load_yaml_config(config_source)
        qlib_init = dict(loaded.qlib_init or {})

        provider_uri = provider_uri_override or qlib_init.get("provider_uri")
        if provider_uri is not None:
            qlib_init["provider_uri"] = self._expand_provider_uri(str(provider_uri))

        qlib.init(**qlib_init)

        if mlruns_uri:
            R.set_uri(mlruns_uri)

        exp_name = experiment_name or "qlib_workflow"
        with R.start(experiment_name=exp_name):
            model_cfg = dict(loaded.task["model"])
            model_kwargs = dict(model_cfg.get("kwargs") or {})

            # YAML compatibility: some configs use `nhead` but Qlib uses `n_head`.
            if "nhead" in model_kwargs and "n_head" not in model_kwargs:
                model_kwargs["n_head"] = model_kwargs.pop("nhead")

            # Windows reliability: default to single-process workers unless explicitly overridden.
            if os.name == "nt" and model_cfg.get("class") == "TransformerModel" and "n_jobs" in model_kwargs:
                from config.settings import settings
                override = settings.qlib_torch_dataloader_workers
                model_kwargs["n_jobs"] = int(override) if override is not None else 0

            model_cfg["kwargs"] = model_kwargs

            model = init_instance_by_config(model_cfg)
            dataset = init_instance_by_config(loaded.task["dataset"])
            model.fit(dataset)
            R.save_objects(trained_model=model)

            rec = R.get_recorder()
            for rec_cfg in loaded.task.get("record", []):
                kwargs = dict(rec_cfg.get("kwargs") or {})
                kwargs = {
                    k: (model if v == "<MODEL>" else dataset if v == "<DATASET>" else v)
                    for k, v in kwargs.items()
                }
                if rec_cfg.get("class") == "PortAnaRecord" and isinstance(kwargs.get("config"), dict):
                    kwargs["config"] = self._patch_port_ana_config_for_calendar(kwargs["config"])
                kwargs["recorder"] = rec

                record = init_instance_by_config(
                    {"class": rec_cfg["class"], "module_path": rec_cfg["module_path"], "kwargs": kwargs}
                )
                if hasattr(record, "generate"):
                    record.generate()

            metrics = self._extract_sig_metrics(rec)
            return TrainResult(
                experiment_id=str(rec.experiment_id),
                recorder_id=str(rec.id),
                metrics=metrics,
                config_source=str(config_source),
            )

