# -*- coding: utf-8 -*-
"""
This script is to flatten the amenities JSON file ('Surroundings.json') and extract features of interest from it.

The script will then join the extracted features with the POS sales file ('sales_granular.csv') and export the joint dataframe as a CSV file for further data exploration.

"""

import pandas as pd
from pandas.io.json import json_normalize
import json
from datetime import datetime

"""
Functions to transform JSON dataset and create dataframes of features
"""

###Function for store (POS) counts in JSON###

def json_store_count(json_data):
    
    return len(json_data)

###Create field names for features calculated###

def json_features_header_constructor(json_data,metric_name=''):
    
    ###takes the first line of loaded json data
    first_line = json_normalize(json_data[0])
    
    ###extracts headers of the first branches
    categories = list(first_line)
    
    #identify the number of headers to create for
    header_count = len(categories)
    
    #create header array
    category_headers = [None]*header_count
    
    for i in range(0, header_count):
        
        #converst headers to strings for header parsing
        header = str(categories[i])
        
        if header.count('surroundings.') > 0 and metric_name == '':
            # strips the "surroundings." portion from the header name
            category_headers[i] = header[13:]
        elif header.count('surroundings.') > 0:
             # strips the "surroundings." portion from the header name and appends feature name
            category_headers[i] = header[13:] + '_' + metric_name
        else:
            category_headers[i] = header
    
    return category_headers

###Function which constructs pandas dataframe for feature of interest###

def json_features_dataframe_constructor(json_data, metric_name='counts', average=True):
    
    #create number of rows for dataframe
    rows = json_store_count(json_data)
    
    #create headers for dataframe
    if average == True:
        category_headers = json_features_header_constructor(json_data, 'avg_' + metric_name)
    else:
        category_headers = json_features_header_constructor(json_data, metric_name)
    
    #create empty pandas DataFrame
    df = pd.DataFrame(index = range(0, rows), columns = category_headers)
    
    #create an array for types of amenities
    amenities = json_features_header_constructor(json_data, metric_name='')
    
    
    #filling in pandas DataFrame with feature values
    for i in range(0, rows):
        
        #parses 1 store's worth of JSON data
        store_line_item = json_normalize(json_data[i])
    
        for j in range(0, len(category_headers)):
        
            if j == 0:
                #captures and records the store code
                df.iloc[i][j] = store_line_item.iloc[0][j]
            else:
                #converts the remainder JSON data for 1 amenity category to string for parsing
                json_str = str(store_line_item.iloc[0][j])
                #the number of locations for will be based on count of place_id for each amenity category
                amenity_count = json_str.count("'place_id'")
                
                if metric_name == "counts":
                    #records the number of locations under each amenity type
                    df.iloc[i][j] = amenity_count
                else:
                    #looks into each amenity location to extract features of interest
                    total = 0
                    for k in range(0,amenity_count):
                        
                        if str(json_data[i]["surroundings"][amenities[j]][k]).count(metric_name) > 0:

                            try: 
                                total += json_data[i]["surroundings"][amenities[j]][k][metric_name]
                            except KeyError:
                                total += json_data[i]["surroundings"][amenities[j]][k]["reviews"][0]["rating"] #an exception occured with ratings where an user rating was not accurately reflected for the location average. this error handling is to capture the specific user rating for the given dataset
                        else:
                            total = total
                
                    if average == True and amenity_count > 0:
                        df.iloc[i][j] = total/amenity_count
                    else:
                        df.iloc[i][j] = total
    
    return df

"""
Functions to clean/manipulate sales volume csv dataset
"""

###create string to datetime parser for headers in sales data###

def headers_str_datetime(pd_dataframe):
    
    #extract headers from dataframes
    headers = pd_dataframe.columns
    
    #create empty list for converted datetime headers
    date_headers = [None]*len(headers)
    
    #first header of sales volume dataset is 'store_code' hence should be maintained
    date_headers[0] = headers[0]
    
    #convert string datetime headers into datetime format
    for i in range(1, len(headers)):
        date_headers[i] = datetime.strptime(headers[i], '%m/%d/%y %H:%M')
    
    return date_headers

###create column of sum of the sales within a given period in m-d-yy format###

def cond_sum_column(pd_dataframe, start_date, end_date, date_format='%m/%d/%y', new_column_name = 'cond_sum', weekends_only=False):
    
    #convert date strings into datetime format
    start = datetime.strptime(start_date, date_format)
    end = datetime.strptime(end_date, date_format)
    
    #extract headers from pandas dataframe
    
    headers = pd_dataframe.columns
    num_headers = len(headers)
    
    #create a list of applicable column headers to be summed
    
    applicable_headers = []
    
    #scan through each of pandas dataframe header and append ones that fit the conditions
    for i in range(1, num_headers):
        if str(type(headers[i])) == "<class 'datetime.datetime'>": 
            
            if weekends_only == False: #checks whether to exclude only for saturdays + sundays
                if headers[i] <= end and headers[i] >= start:
                    applicable_headers.append(headers[i])
            else:
                if headers[i] <= end and headers[i] >= start and headers[i].weekday() >= 5:
                    applicable_headers.append(headers[i])
            
                    
    #create a new column of the sum in applicable columns by their headers in the pandas dataframe
    
    pd_dataframe[new_column_name] = pd_dataframe[applicable_headers].sum(axis=1)
    
    return pd_dataframe

###create function to calculate date of the first sale of the POS - used to estimate opening dates###

def POS_first_sales(pd_dataframe):
    
    #define number of stores to look
    POS_count = len(pd_dataframe)
    
    #create list to store first sales datetimes
    first_sales = []
    
    #scrape first sales date
    for i in range(0, POS_count):
        
        j = 1
        
        while (str(pd_dataframe.iloc[i,j]) == 'nan'):
            j += 1
            
        first_sales.append(pd_dataframe.columns[j])
        
    
    #append first sales date to dataframe
    pd_first_sales_date = pd.Series(first_sales)
    
    return pd_first_sales_date

###create function to calculate most recent sales date of the POS - used to estimate closing dates###

def POS_recent_sales(pd_dataframe):
    
    #define number of stores to look
    POS_count = len(pd_dataframe)
    
    #define number of columns to consider max
    column_count = len(pd_dataframe.columns)
    
    #create list to store first sales datetimes
    recent_sales = []
    
    #scrape most recent sales date
    
    for i in range(0, POS_count):
        
        j = column_count - 1
        
        while (str(pd_dataframe.iloc[i,j]) == 'nan' and j >=0):
            j = j-1
        
        recent_sales.append(pd_dataframe.columns[j])
    
    #append recent sales date to dataframe
    
    pd_recent_sales_date = pd.Series(recent_sales)
    
    return pd_recent_sales_date

###create an extrapolated sum data series of volume by estimating daily volume of operation period and multiply by number of days to project for###
    
def extrapolate_sum(vol_series, first_sale_series, recent_sale_series, start_date, end_date, date_format='%m/%d/%y'):
    
    start = datetime.strptime(start_date, date_format)
    end = datetime.strptime(end_date, date_format)
    
    datediff = end - start
    n_days_to_fill = datediff.days
    
    #identify the number of POS
    
    store_count = len(vol_series)
    extrapolated_volume = []

    for i in range(0, store_count):
        
        real_first_date = max(first_sale_series.iloc[i], start)
        real_last_date = min(recent_sale_series.iloc[i], end)
        
        realdiff = real_last_date - real_first_date
        real_n_days_open = realdiff.days
        
        if real_n_days_open == 0:
            real_n_days_open = 1
        
        annualized_vol = (vol_series.iloc[i]/real_n_days_open)*n_days_to_fill
        
        extrapolated_volume.append(annualized_vol)
        
    extrapolated = pd.Series(extrapolated_volume)
    
    return extrapolated

### create column which evaluates volume contribution of each store at a given column in a pandas dataframe###

def vol_contribution(pd_dataframe, column_name, new_column_name = "_contribution_by_store"):
    
    #calculate sum of column of interest
    total = pd_dataframe[column_name].sum()
    
    #create header for the new volume contribution volume
    header = column_name + new_column_name
    
    #create volume contribution column and append to dataframe input
    pd_dataframe[header] = pd_dataframe[column_name]/total
    
    #output dataframe with volume contribution appended
    return pd_dataframe
    

"""
stores with annualized volume greater than annualized volume per POS (total volume/POS count) will be deemed as "performing" store

the following function constructs the classifying target variable for the models
"""

def high_performing_identifier(pd_criterion_series):
    
    length = len(pd_criterion_series)
    
    average = pd_criterion_series.mean()
    
    target_variable =[]
    
    for i in range(0,length):
        
        if pd_criterion_series.iloc[i]/average >= 1:
            target_variable.append(1)
        else:
            target_variable.append(0)
    
    target = pd.Series(target_variable)
    
    return target
    


"""
Utilizing the functions defined previous to prepare dataset
"""


if __name__ == '__main__':
    
    ### define pathway to amenities data
    json_dir = '''Surroundings JSON file directory'''
    
    ### define pathway to POS sales data
    csv_dir = '''Sales csv file directory'''
    
    """
    Create features dataframe from surroundings.json dataset
    """
    
    ### load amenities dataset
    f = open(json_dir)
    amenities_json_data = json.load(f)
    f.close()
    
    ### find the amount of POSs we have information on surrouding amenities
    POS_count = json_store_count(amenities_json_data)
    
    ### identifies the types of amenities around
    categories = json_features_header_constructor(amenities_json_data,metric_name='')
    
    ### construct dataframe for surrouding amenity location counts for each category for each POS
    
    amenity_count = json_features_dataframe_constructor(amenities_json_data, metric_name='counts', average=False)
    
    ### construct dataframe for average number of reviews per amenity type
    
    average_review_count_by_amenity_type = json_features_dataframe_constructor(amenities_json_data, metric_name='user_ratings_total', average=False)
    
    ### construct dataframe for average ratings per amenity type
    
    average_rating_by_amenity_type = json_features_dataframe_constructor(amenities_json_data, metric_name='rating', average=True)
    
    ###drop duplicate recrods (duplicate stores with same info) of tables
    
    amenity_count = amenity_count.drop_duplicates(subset="store_code").reset_index(drop=True)
    average_review_count_by_amenity_type = average_review_count_by_amenity_type.drop_duplicates(subset="store_code").reset_index(drop=True)
    average_rating_by_amenity_type = average_rating_by_amenity_type.drop_duplicates(subset="store_code").reset_index(drop=True)
    
    
    ###Merge amenity features into new 1 pandas dataframe
    
    amenities = amenity_count.join(average_review_count_by_amenity_type.set_index('store_code'), how="inner", on='store_code').join(average_rating_by_amenity_type.set_index('store_code'),how="inner",on="store_code")
    
    
    """
    Create sales volume dataframe from sales_granular.csv dataset
    """
    
    ###load sales_granular.csv dataset
    sales_volume = pd.read_csv(csv_dir)
    
    ###convert string datetime headers into datetime format
    datetime_headers = headers_str_datetime(sales_volume)
    
    sales_volume.columns = datetime_headers
    
    ###drop duplicate stores
    
    sales_volume = sales_volume.drop_duplicates(subset='store_code')
    
    sales_volume = sales_volume.reset_index(drop=True)
    
    ###approximate store status with first sales date and most recent sales date
    
    first_sales_date = POS_first_sales(sales_volume)
    recent_sales_date = POS_recent_sales(sales_volume)
    
    
    ###create sum up volume for each store in period of interest (which will be most recent year)
    
    start = '6/25/16'
    end = '6/25/17'
    
    sales_volume = cond_sum_column(sales_volume, start, end, new_column_name = 'volume')
    
    ###create a column for volume contribution of the period of interest
    
    sales_volume = vol_contribution(sales_volume, 'volume', new_column_name = '_contribution_by_store')
    
    ###create extrapolated volume for stores that did not operate during the entire period of interests
    
    extrapolated_volume = extrapolate_sum(sales_volume['volume'], first_sales_date, recent_sales_date, start, end)
    
    ###create target variable
    
    high_performing_POS = high_performing_identifier(extrapolated_volume)
    
    ###append calculated fields into POS sales volume dataframe
    
    sales_volume['POS_first_sale'] = first_sales_date
    sales_volume['POS_recent_sale'] = recent_sales_date
    sales_volume['annualized_volume'] = extrapolated_volume
    sales_volume['performing_POS'] = high_performing_POS
    
    ###create classification variable on store volume
    
    """
    Merge amenity features dataframe and sales volume dataframe for training
    """
    
    POS_fields_of_interest = ["store_code",'performing_POS',"volume","volume_contribution_by_store",'annualized_volume','POS_first_sale','POS_recent_sale']
    
    train = sales_volume[POS_fields_of_interest].merge(amenities, how="left", on="store_code")
    
    """
    Export train dataset as a csv
    """
    
    train_export_dir = "C:/Users/Tony Cai/OneDrive/Side Projects/Prestige/PMJ Data Scientist Tokyo/train.csv"
    
    train.to_csv(train_export_dir, index=False)