from __future__ import annotations

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
    config_source: str = "inline"


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

        return cls.load_dict_config(merged, source_label=source_label)

    @classmethod
    def load_dict_config(cls, config: Dict[str, Any], *, source_label: str = "python_config") -> LoadedWorkflowConfig:
        return LoadedWorkflowConfig(
            qlib_init=config.get("qlib_init") or {},
            market=config.get("market") or "all",
            benchmark=config.get("benchmark") or "",
            task=config.get("task") or {},
            config_source=source_label,
        )

    @classmethod
    def compose_config(
        cls,
        *,
        config_source: Optional[str] = None,
        config: Optional[Dict[str, Any] | LoadedWorkflowConfig] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        source_label: str = "python_config",
    ) -> LoadedWorkflowConfig:
        """
        Compose runtime config from either YAML source or in-memory dict/config object.
        This makes workflow control possible directly from Python code.
        """
        if config_source and config is not None:
            raise ValueError("Specify only one of `config_source` or `config`.")
        if not config_source and config is None:
            raise ValueError("Either `config_source` or `config` must be provided.")

        if config_source:
            loaded = cls.load_yaml_config(config_source)
        elif isinstance(config, LoadedWorkflowConfig):
            loaded = config
        else:
            if not isinstance(config, dict):
                raise TypeError("`config` must be a dict or LoadedWorkflowConfig.")
            loaded = cls.load_dict_config(dict(config or {}), source_label=source_label)

        if not config_overrides:
            return loaded

        merged = _deep_merge_dict(
            {
                "qlib_init": dict(loaded.qlib_init or {}),
                "market": loaded.market,
                "benchmark": loaded.benchmark,
                "task": dict(loaded.task or {}),
            },
            config_overrides,
        )
        return cls.load_dict_config(merged, source_label=f"{loaded.config_source}+override")

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
        config_overrides: Optional[Dict[str, Any]] = None,
        task_overrides: Optional[Dict[str, Any]] = None,
        qlib_init_overrides: Optional[Dict[str, Any]] = None,
        provider_uri_override: Optional[str] = None,
        mlruns_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> TrainResult:
        loaded = self.compose_config(config_source=config_source, config_overrides=config_overrides)
        return self.run(
            loaded_config=loaded,
            task_overrides=task_overrides,
            qlib_init_overrides=qlib_init_overrides,
            provider_uri_override=provider_uri_override,
            mlruns_uri=mlruns_uri,
            experiment_name=experiment_name,
        )

    def run_from_config(
        self,
        *,
        config: Dict[str, Any] | LoadedWorkflowConfig,
        source_label: str = "python_config",
        config_overrides: Optional[Dict[str, Any]] = None,
        task_overrides: Optional[Dict[str, Any]] = None,
        qlib_init_overrides: Optional[Dict[str, Any]] = None,
        provider_uri_override: Optional[str] = None,
        mlruns_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> TrainResult:
        loaded = self.compose_config(
            config=config,
            source_label=source_label,
            config_overrides=config_overrides,
        )
        return self.run(
            loaded_config=loaded,
            task_overrides=task_overrides,
            qlib_init_overrides=qlib_init_overrides,
            provider_uri_override=provider_uri_override,
            mlruns_uri=mlruns_uri,
            experiment_name=experiment_name,
        )

    def run(
        self,
        *,
        loaded_config: LoadedWorkflowConfig,
        task_overrides: Optional[Dict[str, Any]] = None,
        qlib_init_overrides: Optional[Dict[str, Any]] = None,
        provider_uri_override: Optional[str] = None,
        mlruns_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> TrainResult:
        import qlib
        from qlib.utils import init_instance_by_config
        from qlib.workflow import R

        qlib_init = dict(loaded_config.qlib_init or {})
        if qlib_init_overrides:
            qlib_init = _deep_merge_dict(qlib_init, qlib_init_overrides)

        provider_uri = provider_uri_override or qlib_init.get("provider_uri")
        if provider_uri is not None:
            qlib_init["provider_uri"] = self._expand_provider_uri(str(provider_uri))

        qlib.init(**qlib_init)
        if mlruns_uri:
            R.set_uri(mlruns_uri)

        task_cfg = dict(loaded_config.task or {})
        if task_overrides:
            task_cfg = _deep_merge_dict(task_cfg, task_overrides)
        if "model" not in task_cfg or "dataset" not in task_cfg:
            raise ValueError("task config must contain both `model` and `dataset`.")

        exp_name = experiment_name or "qlib_workflow"
        with R.start(experiment_name=exp_name):
            model = init_instance_by_config(dict(task_cfg["model"]))
            dataset = init_instance_by_config(dict(task_cfg["dataset"]))
            model.fit(dataset)
            R.save_objects(trained_model=model)

            rec = R.get_recorder()
            for rec_cfg in task_cfg.get("record", []):
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
                config_source=loaded_config.config_source,
            )

