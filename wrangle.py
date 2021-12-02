import pandas as pd
import numpy as np
import env
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
#----------------------------------------------------------#
#get connection

def get_db_url(url):
    url = f'mysql+pymysql://{user}:{password}@{host}/{url}'
    return url

#----------------------------------------------------------#
#get zillow from SQL

def new_zillow_data():
    '''
    This function reads the titanic data from the Codeup db into a df,
    write it to a csv file, and returns the df.
    '''
    # Create SQL query.
    sql_query = """
            
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
    FROM properties_2017
    LEFT JOIN propertylandusetype USING(propertylandusetypeid)
    WHERE propertylandusedesc IN ("Single Family Residential",                       
                                  "Inferred Single Family Residential")"""

    
    df = pd.read_sql(sql_query, get_db_url('zillow'))

    return df

#----------------------------------------------------------#
#def function + csv

def acquire_zillow():
    '''
    This function reads in titanic data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow_df.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('zillow_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = new_zillow_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow_df.csv')
        
    return df