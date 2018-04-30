import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class DataExplore:

    @staticmethod
    def check_nans(data):
        '''
        :param train: train data
        :param test: test data
        :return: table of NAs is the data set 
        '''
        if data.isnull().sum().max()==0: Nans=False
        else: Nans=True
        dtype_df = data.dtypes.reset_index()
        print ("--------data feature types--------")
        print dtype_df
        dtype_df.columns = ["Count", "Column Type"]
        print dtype_df.groupby("Column Type").aggregate('count').reset_index()
        if Nans:
            nas = pd.concat([data.isnull().sum()], axis=1,
                            keys=['Train Dataset'])
            print ("--------Nans in the data sets--------") +'\n' + "train data size " + str(data.shape[0])
            print(nas[nas.sum(axis=1) > 0])


    @staticmethod
    def difference_t_test(data):
        '''
        preform t-test on mean column values grouped by isCategory=0/1
        :param train: 
        :return: sorted features by p-value - most significant feature on top
        '''
        print ("--------features AVG by category/not category in train data--------")
        print data.groupby(['isCategory']).mean()
        category_false = data[data['isCategory']==0]
        category_true = data[data['isCategory']==1]
        headers = list(data.columns.values)
        df_significant = pd.DataFrame(columns=['feature','p_val'])
        row = 0
        for feature in headers:
            print '-----------' + str(feature)
            sample_false = category_false[[str(feature)]]
            sample_true = category_true[[str(feature)]]
            t_stat, p_val = stats.ttest_ind(sample_false.dropna(), sample_true.dropna(), equal_var=False)
            df_significant.loc[row] = [feature,p_val]
            row += 1
        df_significant_sort = df_significant.sort_values(['p_val'],ascending=1)
        print df_significant_sort


    @staticmethod
    def check_corr(data):
        plt.figure(figsize=(10, 4))
        sns.heatmap(data.corr())
        plt.show()

    @staticmethod
    def mean_by_group(data):
        print data.groupby(['isCategory']).mean()
