"""
Provider for local data
"""
import os
import urllib
import json

from d2o_common.data_provider.DataFetcherBase import DataFetcherBase
from d2o_common.api.exception import APIMessageError


class StaticDataFetcher(DataFetcherBase):
    def __init__(self, **kwargs):
        super(StaticDataFetcher, self).__init__(logger_name='static_data_provider', **kwargs)

    def fetch_data(self, url):
        filename = urllib.quote(url, '')

        try:
            with open(os.path.join(self._cached_dir, filename), 'r') as f:
                try:
                    return json.load(f)
                except ValueError as e:
                    self._logger.info(url)
                    self._logger.error(e.message)
                    raise APIMessageError('Data is invalid')
        except IOError as e:
            self._logger.info(url)
            self._logger.error(e.message)
            raise APIMessageError('No such data')
