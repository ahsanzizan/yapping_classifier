import logging


class ColorfulLogger:
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m',  # Magenta
        'ENDC': '\033[0m',      # End color
    }

    class ColorFormatter(logging.Formatter):
        def format(self, record):
            levelname = record.levelname
            if levelname in ColorfulLogger.COLORS:
                color = ColorfulLogger.COLORS[levelname]
                record.levelname = f"{color}{levelname}{ColorfulLogger.COLORS['ENDC']}"
                record.msg = f"{color}{record.msg}{ColorfulLogger.COLORS['ENDC']}"
            return super().format(record)

    @staticmethod
    def get_logger(name: str, level=logging.DEBUG):
        logger = logging.getLogger(name)
        if not logger.hasHandlers():
            ch = logging.StreamHandler()
            ch.setLevel(level)
            formatter = ColorfulLogger.ColorFormatter('%(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            logger.setLevel(level)
        return logger
