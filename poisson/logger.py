import logging

log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
handlers = [logging.FileHandler('train.log', mode='a'), logging.StreamHandler()]
logging.basicConfig(format=log_format, level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S', handlers=handlers)
logger = logging.getLogger(__name__)
