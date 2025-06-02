"""
@file ml_logger.py
@brief Enhanced MLLogger for machine learning experiments.

Provides:
- Colorized terminal logging
- Structured CSV/JSON logging
- Run ID & experiment tagging
- Config, metrics, Git info, system info logging
- Exception handling & timing utilities

@example
# Create logger
logger = MLLogger(
    name="MyExperiment",
    log_to_console=True,
    log_to_file=True,
    file_path="experiment.log",
    structured_output="json",
    level=logging.INFO,
    experiment_name="my_project"
)

# Set experiment run ID
logger.set_run(run_id="run_20250602")

# Log different levels
logger.info("Training started")
logger.warning("Low disk space")
logger.error("Invalid config")

# Log configuration
config = {"lr": 0.01, "batch_size": 32}
logger.log_config(config)

# Log system and Git info
logger.log_system_info()
logger.log_git_info()

# Track time spent on a step
with logger.log_time("Model training"):
    train_model()

# Log metrics
logger.log_metric("accuracy", 0.92, step=1)
logger.log_metric("loss", 0.35, step=1)

# Capture unhandled exceptions
logger.attach_exception_hook()

"""

import logging
import os
import sys
import json
import csv
from datetime import datetime
from contextlib import contextmanager
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
        level=logging.INFO,
        run_id=None,
        experiment_name=None
    ):
        """
        @brief Initializes the logger.
        @param name Logger name.
        @param log_to_console Enable terminal output.
        @param log_to_file Enable logging to file.
        @param file_path Path to the log file.
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

        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)s] %(filename)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red'
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
        @brief Writes a structured log entry to file.
        @param level_name Log level (e.g. INFO).
        @param message Log message.
        @param record LogRecord instance.
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
            with open(self.file_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        elif self.structured_output == "csv":
            write_header = not os.path.exists(self.file_path) or os.stat(self.file_path).st_size == 0
            write_header = True
            with open(self.file_path, "a", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(log_entry)

    def _log(self, level, message):
        record = self.logger.findCaller(stack_info=False)
        getattr(self.logger, level.lower())(message)
        if self.structured_output:
            frame = sys._getframe(2)
            fake_record = logging.LogRecord(
                name=self.logger.name, level=level, pathname=frame.f_code.co_filename,
                lineno=frame.f_lineno, msg=message, args=(), exc_info=None,
                func=frame.f_code.co_name
            )
            self._structured_log(level, message, fake_record)

    def debug(self, msg): self._log("DEBUG", msg)
    def info(self, msg): self._log("INFO", msg)
    def warning(self, msg): self._log("WARNING", msg)
    def error(self, msg): self._log("ERROR", msg)
    def critical(self, msg): self._log("CRITICAL", msg)

    def set_run(self, run_id, experiment_name=None):
        """
        @brief Update experiment run info.
        @param run_id New run ID.
        @param experiment_name Optional new experiment name.
        """
        self.run_id = run_id
        if experiment_name:
            self.experiment_name = experiment_name

    def log_config(self, config):
        """
        @brief Log model or training configuration.
        @param config Dictionary of parameters.
        """
        self.info("Experiment config: " + json.dumps(config, indent=2))

    def log_git_info(self):
        """
        @brief Log current Git commit hash and dirty state.
        """
        try:
            import subprocess
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
            dirty = subprocess.call(['git', 'diff', '--quiet']) != 0
            self.info(f"Git commit: {commit} | Dirty: {dirty}")
        except Exception as e:
            self.warning(f"Could not retrieve Git info: {e}")

    def attach_exception_hook(self):
        """
        @brief Log uncaught exceptions globally.
        """
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            self.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
        sys.excepthook = handle_exception

    @contextmanager
    def log_time(self, step_name):
        """
        @brief Context manager to time a block of code.
        @param step_name Label of the timed step.
        """
        start = time.time()
        self.info(f"Started: {step_name}")
        yield
        end = time.time()
        self.info(f"Finished: {step_name} in {end - start:.2f}s")

    def log_metric(self, name, value, step=None):
        """
        @brief Log a scalar metric (e.g. accuracy).
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
            with open(self.file_path, "a") as f:
                f.write(json.dumps(metric_record) + "\n")

    def log_system_info(self):
        """
        @brief Log host system and hardware info.
        """
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
