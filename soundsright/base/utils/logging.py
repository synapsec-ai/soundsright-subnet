import bittensor as bt

def subnet_logger(severity: str, message: str, log_level: str):
    """This method is a wrapper for the bt.logging function to add extra
    functionality around the native logging capabilities. This method is
    used together with the neuron_logger() method."""
    
    if (isinstance(severity, str) and not isinstance(severity, bool)) and (
        isinstance(message, str) and not isinstance(message, bool) and (isinstance(log_level, str) and not isinstance(log_level, bool))
    ):
        # Do mapping of custom log levels
        log_levels = {
            "INFO": 0,
            "INFOX": 1,
            "DEBUG": 2,
            "DEBUGX": 3,
            "TRACE": 4,
            "TRACEX": 5
        }

        bittensor_severities = {
            "SUCCESS": "SUCCESS",
            "WARNING": "WARNING",
            "ERROR": "ERROR",
            "INFO": "INFO",
            "INFOX": "INFO",
            "DEBUG": "DEBUG",
            "DEBUGX": "DEBUG",
            "TRACE": "TRACE",
            "TRACEX": "TRACE"
        }

        severity_emoji = {
            "SUCCESS": chr(0x2705),
            "ERROR": chr(0x274C),
            "WARNING": chr(0x1F6A8),
            "INFO": chr(0x1F4A1),
            "DEBUG": chr(0x1F527),
            "TRACE": chr(0x1F50D),
        }

        # Use utils.subnet_logger() to write the logs
        if severity.upper() in ("SUCCESS", "ERROR", "WARNING") or log_levels[log_level] >= log_levels[severity.upper()]:

            general_severity=bittensor_severities[severity.upper()]

            if general_severity.upper() == "SUCCESS":
                bt.logging.success(msg=message, prefix=severity_emoji["SUCCESS"])

            elif general_severity.upper() == "ERROR":
                bt.logging.error(msg=message, prefix=severity_emoji["ERROR"])

            elif general_severity.upper() == "WARNING":
                bt.logging.warning(msg=message, prefix=severity_emoji["WARNING"])

            elif general_severity.upper() == "INFO":
                bt.logging.info(msg=message, prefix=severity_emoji["INFO"])

            elif general_severity.upper() == "DEBUG":
                bt.logging.debug(msg=message, prefix=severity_emoji["DEBUG"])

            if general_severity.upper() == "TRACE":
                bt.logging.trace(msg=message, prefix=severity_emoji["TRACE"])