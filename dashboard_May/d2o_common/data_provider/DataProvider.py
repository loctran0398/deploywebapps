from d2o_common.data_provider.StaticDataFetcher import StaticDataFetcher
from d2o_common.data_provider.RemoteDataFetcher import RemoteDataFetcher
from d2o_common.data_provider.config import *


class DataProvider(object):
    def __init__(self, **kwargs):
        data_source = kwargs.get('data_source')
        try:
            self._host = kwargs['host']
        except KeyError:
            raise Exception('Please specify the host')

        if data_source == 'local':
            self._fetcher = StaticDataFetcher(**kwargs)
        else:
            self._fetcher = RemoteDataFetcher(**kwargs)

    def make_url(self, api):
        return '{host}{api}'.format(host=self._host, api=api)

    def get_all_clients(self):
        url = self.make_url(CLIENT_LINK)

        return self._fetcher.fetch_data(url)

    def get_hotel_ids_by_client_id(self, client_id):
        url = self.make_url(HOTEL_IN_DATABASE_LINK.format(client_id=client_id))

        return self._fetcher.fetch_data(url)

    def get_dept_ids_and_configuration(self, client_id, hotel_id):
        url = self.make_url(DEPT_IN_HOTEL_LINK.format(client_id=client_id, h_id_parent=hotel_id))

        return self._fetcher.fetch_data(url)

    def get_hotel_data_in_a_specific_time_period(self, client_id, hotel_id, from_date, to_date):
#        url = self.make_url(REVENUE_LINK.format(client_id=client_id, h_id_parent=hotel_id,
#                                                from_date=str(from_date), to_date=str(to_date)))
        url = self.make_url(NEW_REVENUE_LINK.format(client_id=client_id, h_id_parent=hotel_id,
                                                from_date=str(from_date), to_date=str(to_date)))

        return self._fetcher.fetch_data(url)

    def get_labor_season_by_client_and_department_id(self, client_id, dept_id):
        url = self.make_url(LABOR_SEASON_LINK.format(client_id=client_id, h_id=dept_id))

        return self._fetcher.fetch_data(url)

    def get_labor_season_auto_by_client_and_department_id(self, client_id, dept_id):
        url = self.make_url(LABOR_SEASON_LINK_AUTO.format(client_id=client_id, h_id=dept_id))

        return self._fetcher.fetch_data(url)

    def get_hotel_food_revenue_in_a_specific_time_period(self, client_id, hotel_id, from_date, to_date):
        url = self.make_url(REVENUE_LINK_PROP.format(client_id=client_id, h_id=hotel_id,
                                                     from_date=str(from_date), to_date=str(to_date)))

        return self._fetcher.fetch_data(url)

    def get_client_department_default_parameter(self, client_id, dept_id):
        url = self.make_url(DEPT_DEF_PARAMETER_LINK.format(client_id=client_id, h_id=dept_id))

        return self._fetcher.fetch_data(url)

    def get_department_labor_in_a_specific_time_period(self, client_id, dept_id, from_date, to_date):
        url = self.make_url(LABOR_LINK_ONE_DEPT.format(client_id=client_id, h_id=dept_id,
                                                       from_date=from_date, to_date=to_date))

        return self._fetcher.fetch_data(url)

    def get_labor_of_all_departments_in_a_specific_time_period(self, client_id, hotel_id, from_date, to_date):
        url = self.make_url(LABOR_LINK_ALL_DEPT.format(client_id=client_id, h_id=hotel_id,
                                                       from_date=from_date, to_date=to_date))

        return self._fetcher.fetch_data(url)

    def get_department_forecast_accuracy_data(self, client_id, dep_id, lead_time):
#        url = self.make_url(ACCURACY_LINK.format(client_id=client_id, h_id=dep_id, lead_time=lead_time))
        url = self.make_url(ACCURACY_LINK.format(client_id=client_id, h_id=dep_id))

        return self._fetcher.fetch_data(url)

    def get_department_auto_labor_data(self, client_id, dep_id):
        url = self.make_url(LABOR_AUTO_DATA_LINK.format(client_id=client_id, h_id=dep_id))

        return self._fetcher.fetch_data(url)