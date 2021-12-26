import logging, os


_nameToLevel = {
#    'CRITICAL': logging.CRITICAL,
#    'FATAL': logging.FATAL,  # FATAL = CRITICAL
#    'ERROR': logging.ERROR,
#    'WARN': logging.WARNING,
   'INFO': logging.INFO,
   'DEBUG': logging.DEBUG,
}


def get_logger(name=__name__, log_file=None, log_level=logging.DEBUG, log_level_name=''):
    """ default log level DEBUG """
    logger = logging.getLogger(name)
    if name == 'app':
        fmt= '%(asctime)s %(filename)10s %(levelname)s L %(lineno)d: %(message)s'
    else:
        fmt= '%(asctime)s %(name)s %(levelname)s L %(lineno)d: %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(format=fmt, datefmt=datefmt)
    if log_file is not None:
        log_file_folder = os.path.split(log_file)[0]
        if log_file_folder:
            os.makedirs(log_file_folder, exist_ok=True)
        fh = logging.FileHandler(log_file, 'a', encoding='utf-8')
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(fh)
    if log_level_name in _nameToLevel:
        log_level = _nameToLevel[log_level_name]
    logger.setLevel(log_level)
    return logger

