from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence

from loguru import logger as _console
import sys


class Logger(Protocol):
    """Minimal logging interface used by runners and training loops.

    Implementations may forward to console, Weights & Biases, or both.
    """

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Record run parameters/hyperparameters.

        Implementations should be idempotent and tolerate repeated calls.
        """

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        """Record scalar metrics for an optional global step (e.g., epoch)."""

    def watch(self, model: Any) -> None:
        """Optionally attach model monitoring (no-op for console)."""

    def finish(self) -> None:
        """Flush and close resources if owned by this logger instance."""


class ConsoleLogger:
    """Console sink using loguru for readable output.

    This logger is safe to use everywhere, including sweeps and CV.
    """
    def __init__(self):
        _console.remove()
        _console.add(
            sys.stdout,
            level="TRACE",
            format="<level>{level: <8}</level> | <level>{message}</level>"
        )

    def log_params(self, params: Mapping[str, Any]) -> None:  # type: ignore[override]
        if not params:
            return
        _console.info("Run parameters:")
        for key, value in params.items():
            _console.info(f"  {key} = {value}")

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:  # type: ignore[override]
        if not metrics:
            return
        header = f"Metrics (step={step})" if step is not None else "Metrics"
        _console.info(header + ":")
        for key, value in metrics.items():
            _console.info(f"  {key}: {value}")

    def watch(self, model: Any) -> None:  # type: ignore[override]
        # Console logger does not watch models; intentionally a no-op
        return

    def finish(self) -> None:  # type: ignore[override]
        # Nothing to close for console output
        return


class WandbLogger:
    """Weights & Biases logger that owns or reuses a run.

    If a sweep agent has already initialized a run, this logger reuses it and
    will not call wandb.finish(). Otherwise it initializes a new run and takes
    responsibility for finishing it.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        group: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        job_type: Optional[str] = None,
        mode: Optional[str] = None,  # "online", "offline", or None
        config: Optional[Mapping[str, Any]] = None,
        allow_val_change: bool = True,
    ) -> None:
        try:
            import wandb  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "wandb is not installed but WandbLogger was requested"
            ) from exc

        self._wandb = wandb
        self._owns_run = False

        if wandb.run is None:
            init_kwargs: dict[str, Any] = {}
            if project is not None:
                init_kwargs["project"] = project
            if entity is not None:
                init_kwargs["entity"] = entity
            if run_name is not None:
                init_kwargs["name"] = run_name
            if group is not None:
                init_kwargs["group"] = group
            if tags is not None:
                init_kwargs["tags"] = list(tags)
            if job_type is not None:
                init_kwargs["job_type"] = job_type
            if mode is not None:
                init_kwargs["mode"] = mode

            wandb.init(**init_kwargs)
            self._owns_run = True

        if config:
            try:
                wandb.config.update(dict(config), allow_val_change=allow_val_change)
            except Exception:
                # Be permissive: in some agents, config is frozen; fall back to logging params
                pass

    def log_params(self, params: Mapping[str, Any]) -> None:  # type: ignore[override]
        if not params:
            return
        try:
            self._wandb.config.update(dict(params), allow_val_change=True)
        except Exception:
            # As a fallback, store params under a dedicated namespace in the run summary
            summary_params = {f"params/{k}": v for k, v in params.items()}
            run = getattr(self._wandb, "run", None)
            if run is not None:
                summary = getattr(run, "summary", None)
                if summary is not None:
                    try:
                        summary.update(summary_params)
                    except Exception:
                        pass

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:  # type: ignore[override]
        if not metrics:
            return
        data = dict(metrics)
        if step is not None:
            self._wandb.log(data, step=step)
        else:
            self._wandb.log(data)

    def watch(self, model: Any) -> None:  # type: ignore[override]
        if model is None:
            return
        try:
            self._wandb.watch(model)
        except Exception:
            # Some models or environments may not support watching; ignore silently
            return

    def finish(self) -> None:  # type: ignore[override]
        if self._owns_run:
            try:
                self._wandb.finish()
            except Exception:
                return


class MultiplexLogger:
    """Broadcasts logging calls to multiple sinks (e.g., console and W&B)."""

    def __init__(self, sinks: Sequence[Logger]) -> None:
        if not sinks:
            raise ValueError("MultiplexLogger requires at least one sink")
        self._sinks: tuple[Logger, ...] = tuple(sinks)

    def log_params(self, params: Mapping[str, Any]) -> None:  # type: ignore[override]
        for sink in self._sinks:
            sink.log_params(params)

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:  # type: ignore[override]
        for sink in self._sinks:
            sink.log_metrics(metrics, step=step)

    def watch(self, model: Any) -> None:  # type: ignore[override]
        for sink in self._sinks:
            sink.watch(model)

    def finish(self) -> None:  # type: ignore[override]
        for sink in self._sinks:
            sink.finish()


def make_logger(
    *,
    console: bool = True,
    wandb_enabled: bool = False,
    wandb_init: Optional[Mapping[str, Any]] = None,
) -> Logger:
    """Create a logger according to preferences.

    - If only console is requested, returns ConsoleLogger.
    - If wandb is enabled, returns MultiplexLogger(console + WandbLogger) by default.
    - If console=False and wandb=True, returns only WandbLogger.
    """
    sinks: list[Logger] = []

    if console:
        sinks.append(ConsoleLogger())

    if wandb_enabled:
        wandb_kwargs = dict(wandb_init or {})
        sinks.append(WandbLogger(**wandb_kwargs))

    if not sinks:
        # Fallback to console to avoid silent runs
        return ConsoleLogger()

    if len(sinks) == 1:
        return sinks[0]

    return MultiplexLogger(sinks)


__all__ = [
    "Logger",
    "ConsoleLogger",
    "WandbLogger",
    "MultiplexLogger",
    "make_logger",
]


