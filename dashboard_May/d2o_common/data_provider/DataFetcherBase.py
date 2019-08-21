import os
import logging
from abc import abstractmethod, ABCMeta


class DataFetcherBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, logger_name, folder_path='cached_data', **kwargs):
        self._root_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir, os.pardir))
        self._cached_dir = os.path.join(self._root_dir, folder_path)
        self._logger = logging.getLogger(logger_name)

        log_folder = os.path.join(self._root_dir, 'logs')
        log_file = os.path.join(log_folder, 'data_provider.log')
        if not os.path.exists(log_folder):
            os.mkdir(log_folder)

        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('[%(asctime)s - %(name)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel('INFO')

    @abstractmethod
    def fetch_data(self, url):
        pass
