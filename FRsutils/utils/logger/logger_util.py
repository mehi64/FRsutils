"""
@file ml_logger.py
@brief Enhanced MLLogger for machine learning experiments.

Provides:
- Colorized terminal logging
- Structured CSV/JSON logging to a separate file
- Run ID & experiment tagging
- Config, metrics, Git info, system info logging
- Exception handling & timing utilities
- Step decorator for automatic context logging

@example
logger = MLLogger(
    name="MyExperiment",
    log_to_console=True,
    log_to_file=True,
    file_path="experiment.log",
    structured_output="json",
    level=logging.INFO,
    experiment_name="my_project"
)

logger.set_run(run_id="run_20250602")
logger.info("Training started")
with logger.log_time("training step"):
    train_model()
logger.log_metric("accuracy", 0.93, step=1)
"""

import logging
import os
import sys
import json
import csv
from datetime import datetime
from contextlib import contextmanager
from functools import wraps
import time

try:
    import colorlog
except ImportError:
    raise ImportError("Please install colorlog: pip install colorlog")


class MLLogger:
    """
    @class MLLogger
    @brief Flexible logger with rich utilities for ML experiments.
    """

    def __init__(
        self,
        name=__name__,
        log_to_console=True,
        log_to_file=False,
        file_path="ml.log",
        structured_output=None,  # 'csv' or 'json'
        level=logging.DEBUG,
        run_id=None,
        experiment_name=None
    ):
        """
        @brief Initializes the logger.
        @param name Logger name.
        @param log_to_console Enable terminal output.
        @param log_to_file Enable logging to file.
        @param file_path Path to the human-readable log file.
        @param structured_output Optional structured format: 'csv' or 'json'.
        @param level Logging level.
        @param run_id Unique identifier for this experiment run.
        @param experiment_name Optional experiment group label.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        self.structured_output = structured_output
        self.file_path = file_path
        self.run_id = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or "default_experiment"

        if structured_output:
            base, _ = os.path.splitext(file_path)
            self.structured_path = base + f".{structured_output}"
        else:
            self.structured_path = None

        # Avoid duplicate handlers
        if not self.logger.handlers:
            formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s [%(levelname)s] %(filename)s:%(funcName)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={
                    'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'bold_red'
                }
            )

            if log_to_console:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

            if log_to_file:
                file_handler = logging.FileHandler(self.file_path)
                file_handler.setFormatter(logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(filename)s:%(funcName)s:%(lineno)d - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                ))
                self.logger.addHandler(file_handler)

    def _structured_log(self, level_name, message, record):
        """
        @brief Writes a structured log entry to a separate structured file.
        @param level_name Log level as string.
        @param message Log message.
        @param record A LogRecord instance containing caller context.
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "experiment": self.experiment_name,
            "run_id": self.run_id,
            "level": level_name,
            "filename": record.filename,
            "function": record.funcName,
            "line": record.lineno,
            "message": message
        }

        if self.structured_output == "json":
            with open(self.structured_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        elif self.structured_output == "csv":
            write_header = not os.path.exists(self.structured_path) or os.stat(self.structured_path).st_size == 0
            with open(self.structured_path, "a", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(log_entry)

    def _log(self, level, message):
        """
        @brief Core log dispatcher. Writes to standard logger and structured file (if enabled).
        @param level Logging level (e.g., logging.INFO).
        @param message Message to log.
        """
        if not self.structured_output:
            self.logger.log(level, message)

        if self.structured_output:
            frame = sys._getframe(2)
            fake_record = logging.LogRecord(
                name=self.logger.name, level=level,
                pathname=frame.f_code.co_filename,
                lineno=frame.f_lineno,
                msg=message, args=(), exc_info=None,
                func=frame.f_code.co_name
            )
            self._structured_log(logging.getLevelName(level), message, fake_record)

    # Basic log level methods
    def debug(self, msg): self._log(logging.DEBUG, msg)
    def info(self, msg): self._log(logging.INFO, msg)
    def warning(self, msg): self._log(logging.WARNING, msg)
    def error(self, msg): self._log(logging.ERROR, msg)
    def critical(self, msg): self._log(logging.CRITICAL, msg)

    def set_run(self, run_id, experiment_name=None):
        """Update experiment run ID and optionally name."""
        self.run_id = run_id
        if experiment_name:
            self.experiment_name = experiment_name

    def log_config(self, config):
        """Log configuration parameters (e.g., model/training settings)."""
        self.info("Experiment config: " + json.dumps(config, indent=2))

    def log_metric(self, name, value, step=None):
        """Log a scalar metric.
        @param name Metric name.
        @param value Metric value.
        @param step Optional step or epoch.
        """
        msg = f"Metric [{name}] = {value}" + (f" @ step {step}" if step else "")
        self.info(msg)

        if self.structured_output == "json":
            metric_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "experiment": self.experiment_name,
                "run_id": self.run_id,
                "type": "metric",
                "metric_name": name,
                "value": value,
                "step": step
            }
            with open(self.structured_path, "a") as f:
                f.write(json.dumps(metric_record) + "\n")

    def log_git_info(self):
        """Log the current Git commit hash and dirty status."""
        try:
            import subprocess
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
            dirty = subprocess.call(['git', 'diff', '--quiet']) != 0
            self.info(f"Git commit: {commit} | Dirty: {dirty}")
        except Exception as e:
            self.warning(f"Could not retrieve Git info: {e}")

    def log_system_info(self):
        """Log system specs such as platform, CPU, memory, CUDA availability."""
        try:
            import platform, psutil
            info = {
                "platform": platform.platform(),
                "cpu": platform.processor(),
                "memory_gb": round(psutil.virtual_memory().total / 1e9, 2),
            }
            try:
                import torch
                info["cuda_available"] = torch.cuda.is_available()
            except ImportError:
                info["cuda_available"] = False
            self.info("System Info: " + json.dumps(info))
        except Exception as e:
            self.warning(f"Could not get system info: {e}")

    def attach_exception_hook(self):
        """Attach global exception hook to capture uncaught errors."""
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            self.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
        sys.excepthook = handle_exception

    @contextmanager
    def log_time(self, step_name):
        """Context manager for timing a code block.
        @param step_name Description of the timed step.
        """
        start = time.time()
        self.info(f"Started: {step_name}")
        yield
        end = time.time()
        self.info(f"Finished: {step_name} in {end - start:.2f}s")

    def log_step(self, step_name):
        """Decorator to log the entry and exit of a function, with duration."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.info(f"[STEP] Starting: {step_name}")
                start = time.time()
                try:
                    return func(*args, **kwargs)
                finally:
                    duration = time.time() - start
                    self.info(f"[STEP] Finished: {step_name} in {duration:.2f}s")
            return wrapper
        return decorator
