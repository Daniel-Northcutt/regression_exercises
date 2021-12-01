import pandas as pd
import numpy as np
import env
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
#----------------------------------------------------------#
#get connection

def get_connection(db, user=env.user, host=env.host, password=env.password):
    connection_info = f'mysql+pymysql://{user}:{password}@{host}/{db}'
    return connection_info

#----------------------------------------------------------#
#get zillow from SQL

def get_zillow_data(df):
    sql_query = '''
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
FROM properties_2017
LEFT JOIN propertylandusetype USING (propertylandusetypeid)
WHERE propertylandusedesc IN ("Single Family Residential", "Inferred Single Family Residential");
    '''
    df = pd.read_sql(sql_query, get_db_url('zillow'))


#----------------------------------------------------------#
#def function + csv

def get_zillow_data():
    '''
    This function reads in iris data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('telco.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = get_zillow_data()
        
        # Cache data
        df.to_csv('zillow.csv')
        
    return df