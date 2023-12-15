import logging
import os

def setup_logging(log_file: str = 'log.txt'):
    logger = logging.getLogger('DRYES')
    logger.setLevel(logging.INFO)

    os.makedirs(os.path.dirname(log_file), exist_ok = True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def log(message):
    logger = logging.getLogger('DRYES')
    logger.info(message)
