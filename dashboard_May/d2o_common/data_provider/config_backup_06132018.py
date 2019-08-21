# HOST = "http://172.16.0.51:8485"
FORMAT_DATE = "%Y-%m-%d"
CLIENT_LINK = "/Client/Clients"
HOTEL_IN_DATABASE_LINK = "/Hierarchy/Properties/{client_id}"
DEPT_IN_HOTEL_LINK = "/Hierarchy/Departments/" \
                     "{client_id}/?h_id_parent={h_id_parent}"
REVENUE_LINK = "/Revenue/Departments/Days/" \
               "{client_id}/?h_id_parent={h_id_parent}" \
               "&total=true&segment=true" \
               "&from={from_date}&to={to_date}"
NEW_REVENUE_LINK = "/LiveForecast/Drivers/{client_id}/"\
                "?h_id_parent={h_id_parent}"\
                "&from={from_date}&to={to_date}"
DEPT_DEF_PARAMETER_LINK = "/Season/Auto/Settings/" \
                          "{client_id}/?h_id={h_id}"
#DEPT_SEASONS_DEF = "/Season/Auto/{client_id}/" \
#                   "?h_id={h_id}" & old link
DEPT_SEASONS_DEF = "/Season/Auto/{client_id}/" \
                   "?h_id={h_id}&type=0"                   
#DEPT_SEASON_LINK = "/Season/Auto/{client_id}/" \
#                   "?h_id={h_id}" & old link
DEPT_SEASON_LINK = "/Season/Auto/{client_id}/" \
                   "?h_id={h_id}&type=0"                   
GET_SEASON_AUTO = "/Forecast/RevenueDriver/Auto/" \
                  "{client_id}/?h_id={h_id}"
ONE_DEPT_IN_HOTEL = "/Hierarchy/Department/{client_id}/" \
                    "?h_id={h_id}"
REVENUE_LINK_ONE_DEPT = "/Revenue/Department/Days/{client_id}/?h_id={h_id}" \
                        "&total=true&segment=true" \
                        "&from={from_date}&to={to_date}"
LABOR_LINK_ONE_DEPT = "/Labor/Department/Days/" \
                      "{client_id}/?h_id={h_id}&total=true&segment=true" \
                      "&from={from_date}&to={to_date}"
REVENUE_LINK_PROP = "/Revenue/Property/Days/{client_id}/?h_id={h_id}" \
                    "&from={from_date}&to={to_date}"
LABOR_LINK_ALL_DEPT = "/Labor/Departments/Days/{client_id}/?h_id_parent={h_id}" \
                      "&from={from_date}&to={to_date}"
ACCURACY_LINK = "/Forecast/Accuracy/LeadTime/{client_id}/" \
                "?h_id={h_id}&leadtime={lead_time}"
LABOR_SEASON_LINK = "/Season/{client_id}/?h_id={h_id}&labor=true" 
#LABOR_SEASON_LINK_AUTO = "/Season/Auto/{client_id}/?h_id={h_id}&labor=true"  & old link 
LABOR_SEASON_LINK_AUTO = "/Season/Auto/{client_id}/?h_id={h_id}&type=1"
DEFAULT_TYPE_NAME_MAP = [('rv', 'Revenue'), ('rn', 'Units'), ('gn', 'Guests'),
                         ('food', 'FoodRevenue')]
LABOR_UPLOAD_LINK = "/Labor/Auto/{client_id}"

LABOR_AUTO_DATA_LINK = "/Labor/Auto/{client_id}/?h_id={h_id}"

