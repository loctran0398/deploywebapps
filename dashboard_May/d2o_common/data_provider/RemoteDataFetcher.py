"""
Provider for remote data
"""
import os
import json
import urllib
import requests

from d2o_common.data_provider.DataFetcherBase import DataFetcherBase
from d2o_common.api.exception import APIMessageError

class RemoteDataFetcher(DataFetcherBase):
    def __init__(self, **kwargs):
        super(RemoteDataFetcher, self).__init__(logger_name='remote_data_provider', **kwargs)

    def fetch_data(self, url):
        try:
            response = requests.get(url, timeout=(5, None))
            data = response.json()

#            self.__cache_data(url, data)
            return data
        except Exception as e:
            self._logger.info(url)
            self._logger.error(e.message)
            print(url)
            raise APIMessageError(e.message)

    def __cache_data(self, url, data):
        filename = urllib.quote(url, '')

        with open(os.path.join(self._cached_dir, filename), 'w') as f:
            json.dump(data, f)
