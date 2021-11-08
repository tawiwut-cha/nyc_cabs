'''
Module for visualizing data

Can be run as a script to generate visualizations (To be run from main directory only)
'''
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def corr_heatmap(df, cols:list=None):
    if cols:
        corr_matrix = df[cols].corr()
    else:
        corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix))
    return sns.heatmap(corr_matrix, annot=True, mask=mask)

def outlier_viz(df, cols:list=None):
    if not cols:
        cols = list(df.select_dtypes('number').columns)
    print(df[cols].describe())

    # create a square of subplots
    if np.sqrt(len(cols)) % 1 == 0:
        square_size = int(np.sqrt(len(cols)))
    else:
        square_size = int(np.sqrt(len(cols)) + 1)
    assert square_size**2 >= len(cols), 'outlier_viz: wrong square size logic'
    
    # plot boxplot for each num feat
    fig, ax = plt.subplots(square_size, square_size, figsize=(square_size*5,square_size*5))
    if square_size == 1:
        sns.boxplot(data=df, x=cols[0], ax=ax)
        plt.suptitle('Outliers visualization')
        # plt.show()
    else:
        ax = ax.flatten()
        # current_plot = 0
        for i in range(len(cols)):
            sns.boxplot(data=df, x=cols[i], ax=ax[i])
        plt.suptitle('Outliers visualization')
        # plt.show()
    return fig

def plot_trips_vs_datetime(df):
    # of NYC taxi trips by selected datetime features (Jan-Jun 2016)
    fig = plt.figure(figsize=(16,10))
    plt.suptitle('# of NYC taxi trips by selected datetime features (Jan-Jun 2016)', size=15)

    plt.subplot(221)
    sns.countplot(data=df, x= df['pickup_datetime'].dt.month, hue='vendor_id')
    plt.xlabel('Month')
    plt.ylabel('Trips count')

    plt.subplot(222)
    sns.countplot(data=df, x= df['pickup_datetime'].dt.day_of_week, hue='vendor_id')
    plt.xlabel('Day of week')
    plt.ylabel('Trips count')

    plt.subplot(223)
    sns.countplot(data=df, x= df['pickup_datetime'].dt.day, hue='vendor_id')
    plt.xlabel('Day of month')
    plt.ylabel('Trips count')

    plt.subplot(224)
    sns.countplot(data=df, x= df['pickup_datetime'].dt.hour, hue='vendor_id')
    plt.xlabel('Hour of day')
    plt.ylabel('Trips count')

    return fig

def plot_duration_vs_datetime(df):
    # NYC taxi average trip duration by selected datetime features (Jan-Jun 2016)
    fig = plt.figure(figsize=(16,10))
    plt.suptitle('NYC taxi average trip duration by selected datetime features (Jan-Jun 2016)', size=15)

    plt.subplot(221)
    sns.barplot(data=df, x= df['pickup_datetime'].dt.month, y='trip_duration', hue='vendor_id')
    plt.xlabel('Month')
    plt.ylabel('Average duration (sec)')

    plt.subplot(222)
    sns.barplot(data=df, x= df['pickup_datetime'].dt.day_of_week, y='trip_duration', hue='vendor_id')
    plt.xlabel('Day of week')
    plt.ylabel('Average duration (sec)')

    plt.subplot(223)
    sns.barplot(data=df, x= df['pickup_datetime'].dt.day, y='trip_duration', hue='vendor_id')
    plt.xlabel('Day of month')
    plt.ylabel('Average duration (sec)')

    plt.subplot(224)
    sns.barplot(data=df, x= df['pickup_datetime'].dt.hour, y='trip_duration', hue='vendor_id')
    plt.xlabel('Hour of day')
    plt.ylabel('Average duration (sec)')

    return fig

def main():
    import os
    import sys
    sys.path.append(os.path.abspath(os.getcwd()))
    from src.data.read import read_interim_data

    print('Reading interim data....')
    df = read_interim_data()

    print('Creating trips vs datetime viz....')
    plot_trips_vs_datetime(df).savefig('reports/figures/trips_vs_datetime.jpg')

    print('Creating duration vs datetime viz....')
    plot_duration_vs_datetime(df).savefig('reports/figures/duration_vs_datetime.jpg')

    print('DONE')

if __name__ == '__main__':
    main()