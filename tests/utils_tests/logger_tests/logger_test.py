from FRsutils.utils.logger.logger_util import MLLogger
import logging

logger = MLLogger(
    log_to_console=True,
    log_to_file=False,
    structured_output="csv",  # or "json" or None
    file_path="log_output.csv",
    level=logging.DEBUG
)

def evaluate_model():
    logger.info("Evaluating model...")
    logger.warning("Validation data is imbalanced.")
    # try:
    #     1 / 0
    # except ZeroDivisionError:
        
    #     logger.error("Division by zero in evaluation.")
    #     logger.critical("Critical error in evaluation.")
    #     logger.log_git_info()
    #     logger.log_system_info()
    #     logger.attach_exception_hook()

evaluate_model()
# for i in range(100):
#     evaluate_model()
