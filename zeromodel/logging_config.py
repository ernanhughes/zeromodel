# zeromodel/logging_config.py
"""
Centralized logging configuration for the ZeroModel package.
"""

import logging
import sys
from typing import Optional

# Create a logger for this configuration module itself
_config_logger = logging.getLogger(__name__)

def configure_logging(
    level: int = logging.DEBUG,
    log_file: Optional[str] = 'zeromodel.log',
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    format_string: Optional[str] = None
) -> None:
    """
    Configures logging for the ZeroModel package.

    Sets up a logger that outputs to both the console and a file. Prevents
    adding multiple handlers if called repeatedly.

    Args:
        level: The base logging level for the package logger.
        log_file: The name of the log file. If None, no file handler is added.
        console_level: The logging level for the console output.
        file_level: The logging level for the file output.
        format_string: The format string for log messages. If None, a default is used.
    """
    # Get the logger for the main package (assuming the package is named 'zeromodel')
    # Using __package__ or a fixed name like 'zeromodel' is common.
    # If this file is in a sub-package, adjust accordingly.
    package_logger = logging.getLogger('zeromodel') # Or use logging.getLogger(__package__.split('.')[0]) if nested
    package_logger.setLevel(level)

    # Check if handlers already exist to prevent duplicates
    if not package_logger.handlers:
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        formatter = logging.Formatter(format_string)

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout) # Use stdout
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        package_logger.addHandler(console_handler)
        _config_logger.debug("Console handler added.")

        # File Handler
        if log_file:
            try:
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(file_level)
                file_handler.setFormatter(formatter)
                package_logger.addHandler(file_handler)
                _config_logger.debug(f"File handler added for '{log_file}'.")
            except Exception as e:
                _config_logger.error(f"Failed to create file handler for '{log_file}': {e}")
                # Fallback: Add a warning to the console handler if it exists
                # (though the console handler should already be logging errors from other loggers in the package)
                # Or just rely on the console handler added above.

        _config_logger.info("Logging configured for the 'zeromodel' package.")
    else:
        _config_logger.debug("Logging already configured for the 'zeromodel' package. Skipping.")

# --- Optional: Configure logging automatically when this module is imported ---
# This can be convenient but might be unexpected. Often better to call configure_logging explicitly.
# Uncomment the lines below if you want this behavior.
if __name__ != "__main__": # Avoid running on direct execution of this file if not desired
    configure_logging() 
    _config_logger.debug("Logging auto-configured on import.")
