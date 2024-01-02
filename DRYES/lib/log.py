import logging
import os

def setup_logging(log_destination = None, name = 'DRYES'):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_destination is None:
        log_file = 'log.txt'
    elif log_destination.type == 'local':
        log_file = log_destination.path()
    else:
        # TODO: add support for remote logs
        log_file = None
        logger.warning("Remote logs are not supported yet.")

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok = True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def log(message):
    logger = logging.getLogger('DRYES')
    logger.info(message)
