import logging
import os

def setup_logger(log_file='app.log', log_level=logging.INFO):
    """
    Set up the logger.

    Args:
        log_file (str): The name of the log file, including path.
        log_level (int): The logging level (e.g., logging.DEBUG, logging.INFO).
    """
    # Create the directory for the log file if it doesn't exist
    log_dir = os.path.dirname(log_file)  # Get the directory path
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # Create the directory if it doesn't exist

    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Create handlers
    console_handler = logging.StreamHandler()  # Logs to console
    file_handler = logging.FileHandler(log_file)  # Logs to file

    # Set the logging level for handlers
    console_handler.setLevel(log_level)
    file_handler.setLevel(log_level)

    # Create formatters and add them to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Example usage
if __name__ == '__main__':
    logger = setup_logger('logs/checkpoints/experiment_0/training.log')  # Adjusted path for example
    logger.info('Logger setup complete. Starting the application...')