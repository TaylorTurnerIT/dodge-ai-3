"""Bounded, read-mostly artifacts for CNN image Double-DQN runs.

The trainer is deliberately not part of this module.  A trainer can create a
run, append metrics, publish checkpoints, and update status without making the
dashboard part of its hot path.  The dashboard side only reads this contract
and turns malformed or incomplete terminal runs into visible invalid records.

The on-disk layout is::

    <history-root>/cnn-image-ddqn/<run-id>/
        manifest.json       # immutable run and engine provenance
        config.json         # immutable learner/environment configuration
        status.json         # atomically replaced lifecycle snapshot
        metrics.jsonl       # append-only metric records
        evaluation.json     # optional until a run reaches a terminal state
        report.json         # final report, required for terminal runs
        checkpoints/        # trainer-owned checkpoint files

``history_root`` may also point directly at the ``cnn-image-ddqn`` directory
or at one run directory.  All reader limits are explicit so a dashboard
request cannot read an unbounded metric log or checkpoint directory.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final

VARIANT_ID: Final = "cnn-image-ddqn"
ARTIFACT_SCHEMA_VERSION: Final = 1

MANIFEST_FILENAME: Final = "manifest.json"
CONFIG_FILENAME: Final = "config.json"
STATUS_FILENAME: Final = "status.json"
METRICS_FILENAME: Final = "metrics.jsonl"
EVALUATION_FILENAME: Final = "evaluation.json"
REPORT_FILENAME: Final = "report.json"
CHECKPOINTS_DIRNAME: Final = "checkpoints"

LIFECYCLE_STATES: Final = frozenset(
    {"queued", "running", "paused", "stale", "completed", "failed", "stopped"}
)
GATE_STATES: Final = frozenset({"pending", "pass", "warn", "fail", "invalid"})
ACTIVE_STATES: Final = frozenset({"queued", "running", "paused", "stale"})
TERMINAL_STATES: Final = frozenset({"completed", "failed", "stopped"})

DEFAULT_MAX_RUNS: Final = 256
DEFAULT_MAX_METRIC_LINES: Final = 2_000
DEFAULT_MAX_METRIC_BYTES: Final = 2_000_000
DEFAULT_MAX_CHECKPOINTS: Final = 128
DEFAULT_MAX_REPORT_BYTES: Final = 512_000
DEFAULT_MAX_JSON_BYTES: Final = 2_000_000


class ArtifactError(ValueError):
    """Raised when a writer would violate the artifact contract."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _as_path(value: os.PathLike[str] | str) -> Path:
    return Path(value).expanduser()


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not run_id:
        raise ArtifactError("run_id must be a non-empty string")
    if run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ArtifactError("run_id must be a single safe path component")
    if "\\" in run_id:
        raise ArtifactError("run_id must not contain backslashes")
    return run_id


def _json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
        )
    except (TypeError, ValueError) as error:
        raise ArtifactError(
            f"artifact value is not JSON serializable: {error}"
        ) from error
    return (text + "\n").encode("utf-8")


def _json_line_bytes(value: object) -> bytes:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ArtifactError(
            f"metric value is not JSON serializable: {error}"
        ) from error
    return (text + "\n").encode("utf-8")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Replace ``path`` only after a complete file has been flushed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name)


def _create_immutable_bytes(path: Path, data: bytes) -> None:
    """Create a file once, using an atomic hard-link publication.

    A hard-link from a fully written temporary file gives immutable files an
    exclusive create operation without exposing a partially written payload.
    If another writer won the race, the caller can compare the existing JSON.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            return
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name)
        temporary_name = None
    finally:
        if temporary_name is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name)


def _read_json(path: Path, *, max_bytes: int = DEFAULT_MAX_JSON_BYTES) -> object:
    if path.is_symlink():
        raise OSError("symlinked artifact files are not readable")
    if path.stat().st_size > max_bytes:
        raise OSError("artifact JSON exceeds the reader bound")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{name} must contain a JSON object")
    return dict(value)


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactError("artifact paths must be non-empty strings")
    if "\\" in value:
        raise ArtifactError("artifact paths must use forward slashes")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ArtifactError(f"unsafe artifact path: {value!r}")
    return path.as_posix()


def _variant_root(history_root: Path) -> Path:
    if history_root.name == VARIANT_ID:
        return history_root
    if (history_root / VARIANT_ID).is_dir():
        return history_root / VARIANT_ID
    return history_root / VARIANT_ID


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Canonical paths for one run."""

    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / MANIFEST_FILENAME

    @property
    def config(self) -> Path:
        return self.root / CONFIG_FILENAME

    @property
    def status(self) -> Path:
        return self.root / STATUS_FILENAME

    @property
    def metrics(self) -> Path:
        return self.root / METRICS_FILENAME

    @property
    def evaluation(self) -> Path:
        return self.root / EVALUATION_FILENAME

    @property
    def report(self) -> Path:
        return self.root / REPORT_FILENAME

    @property
    def checkpoints(self) -> Path:
        return self.root / CHECKPOINTS_DIRNAME

    def relative(self, value: str) -> Path:
        relative = _safe_relative_path(value)
        path = self.root / relative
        root = self.root.resolve()
        if not path.resolve(strict=False).is_relative_to(root):
            raise ArtifactError(f"artifact path escapes run root: {value!r}")
        return path


@dataclass(frozen=True, slots=True)
class ArtifactLink:
    """A bounded artifact reference suitable for a dashboard link."""

    label: str
    relative_path: str
    exists: bool


class RunArtifactWriter:
    """Write one run's immutable, atomic, and append-only artifacts."""

    def __init__(
        self,
        paths: RunPaths,
        *,
        manifest: Mapping[str, object],
        config: Mapping[str, object],
        created_at: str | None = None,
    ) -> None:
        self.paths = paths
        self.run_id = _validate_run_id(paths.root.name)
        manifest_payload = dict(manifest)
        manifest_payload.setdefault("schema_version", ARTIFACT_SCHEMA_VERSION)
        manifest_payload.setdefault("variant_id", VARIANT_ID)
        manifest_payload.setdefault("run_id", self.run_id)
        manifest_payload.setdefault("created_at", created_at or _utc_now())
        if manifest_payload["variant_id"] != VARIANT_ID:
            raise ArtifactError("manifest variant_id does not match this variant")
        if manifest_payload["run_id"] != self.run_id:
            raise ArtifactError("manifest run_id does not match its directory")
        self._manifest = manifest_payload
        self._config = dict(config)

        self.paths.root.mkdir(parents=True, exist_ok=True)
        self._ensure_immutable_json(self.paths.manifest, self._manifest, "manifest")
        self._ensure_immutable_json(self.paths.config, self._config, "config")
        self._ensure_empty_metrics()
        if not self.paths.status.exists():
            self.update_status("queued")

    @classmethod
    def create(
        cls,
        history_root: os.PathLike[str] | str,
        run_id: str,
        *,
        manifest: Mapping[str, object],
        config: Mapping[str, object],
        created_at: str | None = None,
    ) -> RunArtifactWriter:
        """Create or reopen ``history_root/cnn-image-ddqn/run_id``."""

        safe_run_id = _validate_run_id(run_id)
        root = _variant_root(_as_path(history_root)) / safe_run_id
        return cls(
            RunPaths(root),
            manifest=manifest,
            config=config,
            created_at=created_at,
        )

    def _ensure_immutable_json(
        self, path: Path, value: Mapping[str, object], name: str
    ) -> None:
        payload = _json_bytes(value)
        _create_immutable_bytes(path, payload)
        try:
            existing = _mapping(_read_json(path), name=name)
        except (OSError, json.JSONDecodeError, ArtifactError) as error:
            raise ArtifactError(f"existing {name} is unreadable") from error
        if existing != dict(value):
            raise ArtifactError(f"immutable {name} does not match the requested value")

    def _ensure_empty_metrics(self) -> None:
        self.paths.metrics.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.paths.metrics,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
        except FileExistsError:
            return
        else:
            os.close(descriptor)

    def _current_status(self) -> dict[str, Any]:
        if not self.paths.status.exists():
            return {}
        try:
            return _mapping(_read_json(self.paths.status), name="status")
        except (OSError, json.JSONDecodeError, ArtifactError):
            return {}

    def update_status(
        self,
        state: str,
        *,
        gate: str | None = None,
        step: int | None = None,
        episode: int | None = None,
        message: str | None = None,
        current_metrics: Mapping[str, object] | None = None,
        artifacts: Mapping[str, object] | None = None,
        updated_at: str | None = None,
    ) -> dict[str, Any]:
        """Atomically publish the latest lifecycle snapshot."""

        normalized_state = str(state).lower()
        if normalized_state not in LIFECYCLE_STATES:
            raise ArtifactError(f"unknown lifecycle state: {state!r}")
        previous = self._current_status()
        normalized_gate = gate if gate is not None else previous.get("gate", "pending")
        normalized_gate = str(normalized_gate).lower()
        if normalized_gate not in GATE_STATES:
            raise ArtifactError(f"unknown evaluation gate: {normalized_gate!r}")

        status: dict[str, Any] = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "variant_id": VARIANT_ID,
            "run_id": self.run_id,
            "state": normalized_state,
            "gate": normalized_gate,
            "updated_at": updated_at or _utc_now(),
        }
        for name, value in (
            ("step", step),
            ("episode", episode),
            ("message", message),
            ("current_metrics", dict(current_metrics) if current_metrics else None),
        ):
            if value is not None:
                status[name] = value
            elif name in previous:
                status[name] = previous[name]
        existing_artifacts = previous.get("artifacts")
        if isinstance(existing_artifacts, Mapping):
            status["artifacts"] = dict(existing_artifacts)
        if artifacts is not None:
            status["artifacts"] = dict(artifacts)
        _atomic_write_bytes(self.paths.status, _json_bytes(status))
        return status

    write_status = update_status

    def append_metrics(self, record: Mapping[str, object]) -> None:
        """Append exactly one JSON object to the metrics log."""

        if not isinstance(record, Mapping):
            raise ArtifactError("metric records must be JSON objects")
        line = _json_line_bytes(dict(record))
        descriptor = os.open(
            self.paths.metrics,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        try:
            view = memoryview(line)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _update_artifact_reference(self, key: str, value: object) -> None:
        previous = self._current_status()
        artifacts = previous.get("artifacts")
        merged = dict(artifacts) if isinstance(artifacts, Mapping) else {}
        merged[key] = value
        state = str(previous.get("state", "queued"))
        self.update_status(state, artifacts=merged)

    def record_checkpoint(self, relative_path: str) -> str:
        """Record a trainer-owned checkpoint path without mutating its bytes."""

        safe_path = _safe_relative_path(relative_path)
        self.paths.relative(safe_path)
        previous = self._current_status()
        artifacts = previous.get("artifacts")
        merged = dict(artifacts) if isinstance(artifacts, Mapping) else {}
        checkpoints = merged.get("checkpoints")
        paths = (
            list(checkpoints)
            if isinstance(checkpoints, Sequence)
            and not isinstance(checkpoints, (str, bytes))
            else []
        )
        if safe_path not in paths:
            paths.append(safe_path)
        merged["checkpoints"] = paths
        self.update_status(str(previous.get("state", "queued")), artifacts=merged)
        return safe_path

    def write_evaluation_summary(
        self,
        summary: Mapping[str, object],
        *,
        relative_path: str = EVALUATION_FILENAME,
    ) -> str:
        """Atomically publish an evaluation summary and register its path."""

        if not isinstance(summary, Mapping):
            raise ArtifactError("evaluation summary must be a JSON object")
        safe_path = _safe_relative_path(relative_path)
        _atomic_write_bytes(self.paths.relative(safe_path), _json_bytes(dict(summary)))
        self._update_artifact_reference("evaluation", safe_path)
        return safe_path

    def write_report(
        self,
        report: Mapping[str, object] | str,
        *,
        relative_path: str = REPORT_FILENAME,
    ) -> str:
        """Atomically publish a JSON or text final report and register its path."""

        safe_path = _safe_relative_path(relative_path)
        payload = (
            _json_bytes(dict(report))
            if isinstance(report, Mapping)
            else str(report).encode("utf-8")
        )
        _atomic_write_bytes(self.paths.relative(safe_path), payload)
        self._update_artifact_reference("report", safe_path)
        return safe_path

    def finalize(
        self,
        *,
        gate: str,
        evaluation: Mapping[str, object] | None = None,
        report: Mapping[str, object] | str | None = None,
        state: str = "completed",
    ) -> dict[str, Any]:
        """Publish optional evaluation/report artifacts, then terminal status."""

        if evaluation is not None:
            self.write_evaluation_summary(evaluation)
        if report is not None:
            self.write_report(report)
        return self.update_status(state, gate=gate)


def create_run_artifacts(
    history_root: os.PathLike[str] | str,
    run_id: str,
    *,
    manifest: Mapping[str, object],
    config: Mapping[str, object],
    created_at: str | None = None,
) -> RunArtifactWriter:
    """Functional entry point for creating or reopening one run."""

    return RunArtifactWriter.create(
        history_root,
        run_id,
        manifest=manifest,
        config=config,
        created_at=created_at,
    )


def _optional_json(path: Path, name: str) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        return _mapping(_read_json(path), name=name), None
    except (OSError, json.JSONDecodeError, ArtifactError) as error:
        return None, f"{name}_malformed:{type(error).__name__}"


def _read_metrics(
    path: Path,
    *,
    state: str,
    max_lines: int,
    max_bytes: int,
) -> tuple[tuple[dict[str, Any], ...], list[str]]:
    if not path.exists() or path.is_symlink():
        return (), ["metrics_missing"]
    reasons: list[str] = []
    try:
        with path.open("rb") as stream:
            file_size = stream.seek(0, os.SEEK_END)
            offset = max(0, file_size - max_bytes)
            stream.seek(offset)
            raw = stream.read(max_bytes)
    except OSError as error:
        return (), [f"metrics_unreadable:{type(error).__name__}"]
    if offset:
        separator = raw.find(b"\n")
        raw = raw[separator + 1 :] if separator >= 0 else b""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return (), reasons + ["metrics_not_utf8"]
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]

    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = _mapping(json.loads(line), name="metric record")
        except (json.JSONDecodeError, ArtifactError):
            is_partial_tail = index == len(lines) - 1 and not text.endswith("\n")
            if is_partial_tail and state in ACTIVE_STATES:
                reasons.append("metrics_partial_tail_ignored")
                break
            reasons.append(f"metrics_malformed_line:{index + 1}")
            continue
        records.append(value)
    return tuple(records), reasons


def _declared_paths(
    status: Mapping[str, object],
    *,
    run_paths: RunPaths,
    max_checkpoints: int,
) -> tuple[list[ArtifactLink], list[str]]:
    reasons: list[str] = []
    links: list[ArtifactLink] = []

    def add(label: str, relative_path: str) -> None:
        try:
            safe_path = _safe_relative_path(relative_path)
            path = run_paths.relative(safe_path)
        except ArtifactError:
            reasons.append(f"artifact_path_invalid:{label}")
            return
        links.append(
            ArtifactLink(label, safe_path, path.is_file() and not path.is_symlink())
        )

    for label, path in (
        ("Manifest", MANIFEST_FILENAME),
        ("Config", CONFIG_FILENAME),
        ("Status", STATUS_FILENAME),
        ("Metrics", METRICS_FILENAME),
    ):
        add(label, path)

    artifacts = status.get("artifacts")
    artifacts_map = artifacts if isinstance(artifacts, Mapping) else {}
    evaluation = artifacts_map.get("evaluation")
    if evaluation is None and run_paths.evaluation.is_file():
        evaluation = EVALUATION_FILENAME
    if isinstance(evaluation, str):
        add("Evaluation", evaluation)

    report = artifacts_map.get("report")
    if report is None:
        if run_paths.report.is_file():
            report = REPORT_FILENAME
        elif (run_paths.root / "REPORT.md").is_file():
            report = "REPORT.md"
    if isinstance(report, str):
        add("Report", report)

    checkpoint_values = artifacts_map.get("checkpoints")
    if checkpoint_values is not None and not isinstance(checkpoint_values, Sequence):
        reasons.append("checkpoints_malformed")
        checkpoint_values = ()
    for index, checkpoint in enumerate(checkpoint_values or ()):
        if index >= max_checkpoints:
            reasons.append("checkpoints_exceed_bound")
            break
        if isinstance(checkpoint, str):
            add("Checkpoint", checkpoint)
        else:
            reasons.append("checkpoint_path_invalid")

    if not checkpoint_values and run_paths.checkpoints.is_dir():
        try:
            discovered = sorted(
                path
                for path in run_paths.checkpoints.rglob("*")
                if path.is_file() and not path.is_symlink()
            )[:max_checkpoints]
        except OSError:
            discovered = []
            reasons.append("checkpoints_unreadable")
        for path in discovered:
            relative = path.relative_to(run_paths.root).as_posix()
            add("Checkpoint", relative)
    return links, reasons


def _read_report(path: Path) -> tuple[object | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        if path.stat().st_size > DEFAULT_MAX_REPORT_BYTES:
            return None, "report_exceed_bound"
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        return None, f"report_unreadable:{type(error).__name__}"
    if path.suffix.lower() == ".json":
        try:
            return json.loads(text), None
        except json.JSONDecodeError:
            return None, "report_malformed"
    return text, None


@dataclass(frozen=True, slots=True)
class RunRecord:
    """A safe, bounded snapshot of one run for catalog and dashboard use."""

    path: Path
    run_id: str
    variant_id: str
    manifest: dict[str, Any]
    config: dict[str, Any]
    status: dict[str, Any]
    metrics: tuple[dict[str, Any], ...]
    evaluation: dict[str, Any] | None
    report: object | None
    artifacts: tuple[ArtifactLink, ...]
    valid: bool
    invalid_reasons: tuple[str, ...]

    @property
    def state(self) -> str:
        value = str(self.status.get("state", "invalid")).lower()
        return value if value in LIFECYCLE_STATES else "invalid"

    @property
    def gate(self) -> str:
        if not self.valid:
            return "invalid"
        value = str(self.status.get("gate", "pending")).lower()
        return value if value in GATE_STATES else "invalid"

    @property
    def latest_metrics(self) -> dict[str, Any]:
        latest: dict[str, Any] = {}
        for record in self.metrics:
            latest.update(record)
        current = self.status.get("current_metrics")
        if isinstance(current, Mapping):
            latest.update(current)
        return latest


def load_run(
    run_root: os.PathLike[str] | str,
    *,
    expected_variant: str = VARIANT_ID,
    max_metric_lines: int = DEFAULT_MAX_METRIC_LINES,
    max_metric_bytes: int = DEFAULT_MAX_METRIC_BYTES,
    max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS,
) -> RunRecord:
    """Load one run without raising for malformed or incomplete artifacts."""

    root = _as_path(run_root)
    run_id = root.name or "unknown"
    reasons: list[str] = []

    manifest: dict[str, Any] = {}
    config: dict[str, Any] = {}
    status: dict[str, Any] = {}
    for path, name, target in (
        (root / MANIFEST_FILENAME, "manifest", manifest),
        (root / CONFIG_FILENAME, "config", config),
        (root / STATUS_FILENAME, "status", status),
    ):
        if not path.is_file():
            reasons.append(f"{name}_missing")
            continue
        try:
            target.update(_mapping(_read_json(path), name=name))
        except (OSError, json.JSONDecodeError, ArtifactError) as error:
            reasons.append(f"{name}_malformed:{type(error).__name__}")

    manifest_variant = manifest.get("variant_id")
    if manifest_variant != expected_variant:
        reasons.append("manifest_variant_mismatch")
    if manifest.get("run_id") != run_id:
        reasons.append("manifest_run_id_mismatch")
    if status.get("variant_id") not in {None, expected_variant}:
        reasons.append("status_variant_mismatch")
    if status.get("run_id") not in {None, run_id}:
        reasons.append("status_run_id_mismatch")

    state = str(status.get("state", "invalid")).lower()
    if state not in LIFECYCLE_STATES:
        reasons.append("state_invalid")
    gate = str(status.get("gate", "pending")).lower()
    if gate not in GATE_STATES:
        reasons.append("gate_invalid")
    if gate == "pass" and state not in {"completed"}:
        reasons.append("pass_gate_before_completion")

    metrics, metric_reasons = _read_metrics(
        root / METRICS_FILENAME,
        state=state,
        max_lines=max_metric_lines,
        max_bytes=max_metric_bytes,
    )
    reasons.extend(metric_reasons)

    evaluation: dict[str, Any] | None = None
    evaluation_path = status.get("artifacts", {})
    declared_evaluation = (
        evaluation_path.get("evaluation")
        if isinstance(evaluation_path, Mapping)
        else None
    )
    if isinstance(declared_evaluation, str):
        try:
            evaluation_file = RunPaths(root).relative(declared_evaluation)
        except ArtifactError:
            evaluation_file = root / "__invalid_evaluation_path__"
            reasons.append("evaluation_path_invalid")
    else:
        evaluation_file = root / EVALUATION_FILENAME
    evaluation, evaluation_error = _optional_json(evaluation_file, "evaluation")
    if evaluation_error:
        reasons.append(evaluation_error)

    links, artifact_reasons = _declared_paths(
        status,
        run_paths=RunPaths(root),
        max_checkpoints=max_checkpoints,
    )
    reasons.extend(artifact_reasons)

    report_file: Path | None = None
    declared_report = (
        evaluation_path.get("report") if isinstance(evaluation_path, Mapping) else None
    )
    if isinstance(declared_report, str):
        try:
            report_file = RunPaths(root).relative(declared_report)
        except ArtifactError:
            reasons.append("report_path_invalid")
    elif (root / REPORT_FILENAME).is_file():
        report_file = root / REPORT_FILENAME
    elif (root / "REPORT.md").is_file():
        report_file = root / "REPORT.md"
    report, report_error = _read_report(report_file) if report_file else (None, None)
    if report_error:
        reasons.append(report_error)

    if state in TERMINAL_STATES:
        if report_file is None or report is None:
            reasons.append("final_report_missing")
        if state == "completed" and evaluation is None:
            reasons.append("evaluation_missing")

    # Preserve the first occurrence of each reason so malformed files do not
    # produce an unreadable wall of duplicate dashboard diagnostics.
    unique_reasons = tuple(dict.fromkeys(reasons))
    return RunRecord(
        path=root,
        run_id=run_id,
        variant_id=str(manifest.get("variant_id", expected_variant)),
        manifest=manifest,
        config=config,
        status=status,
        metrics=metrics,
        evaluation=evaluation,
        report=report,
        artifacts=tuple(links),
        valid=not unique_reasons,
        invalid_reasons=unique_reasons,
    )


def _candidate_run_dirs(history_root: Path) -> list[Path]:
    if (history_root / MANIFEST_FILENAME).is_file() or (
        history_root / STATUS_FILENAME
    ).is_file():
        return [history_root]
    if history_root.name == VARIANT_ID:
        variant_root = history_root
    elif (history_root / VARIANT_ID).is_dir():
        variant_root = history_root / VARIANT_ID
    else:
        # This fallback makes a temporary test root containing run directories
        # useful without weakening the normal variant namespace layout.
        variant_root = history_root
    if not variant_root.is_dir():
        return []
    try:
        return sorted(
            (
                path
                for path in variant_root.iterdir()
                if path.is_dir()
                and not path.is_symlink()
                and not path.name.startswith(".")
            ),
            key=lambda path: path.name,
        )
    except OSError:
        return []


def discover_runs(
    history_root: os.PathLike[str] | str,
    *,
    max_runs: int = DEFAULT_MAX_RUNS,
    max_metric_lines: int = DEFAULT_MAX_METRIC_LINES,
    max_metric_bytes: int = DEFAULT_MAX_METRIC_BYTES,
    max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS,
) -> tuple[RunRecord, ...]:
    """Discover a bounded, sorted catalog; malformed entries never escape."""

    if max_runs < 1:
        raise ValueError("max_runs must be positive")
    root = _as_path(history_root)
    records: list[RunRecord] = []
    for run_root in _candidate_run_dirs(root)[:max_runs]:
        try:
            records.append(
                load_run(
                    run_root,
                    max_metric_lines=max_metric_lines,
                    max_metric_bytes=max_metric_bytes,
                    max_checkpoints=max_checkpoints,
                )
            )
        except Exception as error:  # noqa: BLE001 - catalog must not crash on a run
            records.append(
                RunRecord(
                    path=run_root,
                    run_id=run_root.name,
                    variant_id=VARIANT_ID,
                    manifest={},
                    config={},
                    status={},
                    metrics=(),
                    evaluation=None,
                    report=None,
                    artifacts=(),
                    valid=False,
                    invalid_reasons=(f"loader_error:{type(error).__name__}",),
                )
            )
    return tuple(records)


catalog_runs = discover_runs


__all__ = [
    "ACTIVE_STATES",
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactError",
    "ArtifactLink",
    "CONFIG_FILENAME",
    "EVALUATION_FILENAME",
    "GATE_STATES",
    "LIFECYCLE_STATES",
    "MANIFEST_FILENAME",
    "METRICS_FILENAME",
    "REPORT_FILENAME",
    "RunArtifactWriter",
    "RunPaths",
    "RunRecord",
    "STATUS_FILENAME",
    "VARIANT_ID",
    "catalog_runs",
    "create_run_artifacts",
    "discover_runs",
    "load_run",
]
