#outlier handling
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from eda_plots import distribution_plots

eda_cleaneddir='eda_clean_plots'

def load_data(file_path):
    return pd.read_csv(file_path)

def handle_outliers(df, column):
    Q1=df[column].quantile(0.25)
    Q3=df[column].quantile(0.75)
    IQR=Q3-Q1
    lower_limit=Q1-1.5*IQR
    upper_limit=Q3+1.5*IQR
    
    median=df[column].median()
    df[column] = np.where(np.logical_or(df[column] > upper_limit, df[column] < lower_limit), median, df[column])
    return df
def save_data(df, file_path):
    df.to_csv(file_path, index=False)
    
def main():
    cleaned_data= 'cleaned_data.csv'
    df=load_data('Advertising.csv')
    numeric_columns=df.select_dtypes(include='number').columns

    for column in numeric_columns:
        df=handle_outliers(df,column)
    save_data(df, cleaned_data)
    print('Data processing cpmplete. Cleaned data saved to')
    distribution_plots(cleaned_data, eda_cleaneddir)
if __name__ == "__main__":
    main()
    
    
    
    
