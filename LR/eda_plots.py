import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

data=pd.read_csv('preprocessed.csv')

def distribution_plots(df,output_dir, hue=None):
# Function that takes 2 params:
#1. df: the input dataframe
#2. output_dir: where the data will be saved

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for column in data.columns:
        plt.figure(figsize=(12,8))
        if pd.api.types.is_numeric_dtype(df[column]):
            sns.histplot(data[column], kde=True)
            plt.title(f'Distribution Plot for {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, f'{column}_distplot.png'))
            plt.close()

            plt.figure(figsize=(12,8))
            sns.boxplot(data=df, x=column, showfliers=True)
            plt.title(f'Box Plot for {column}')
            plt.xlabel(column)
            plt.savefig(os.path.join(output_dir, f'{column}_boxplot.png'))
            plt.close()
    if 'Unnamed: 0' in df.columns:
        df=df.drop('Unnamed: 0', axis=1)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(output_dir, 'heatmap.png'))
    plt.close()

    plt.figure()
    sns.pairplot(df, hue='Outcome')
    plt.close()

    
    plt.figure(figsize=(12,8))
    sns.countplot(x=data['Outcome'])
    plt.title('Countplot of the Target Column (Outcome)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'Outcome_countplot.png'))
    plt.close()

output_directory = 'eda_plots'

if __name__=='__main__':
    distribution_plots(data, output_directory)

