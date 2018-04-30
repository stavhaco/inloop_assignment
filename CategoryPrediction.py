from PredictionModel import PredictionModel
from DataPreperation import DataPreperation
from DataExplore import DataExplore
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import train_test_split
from StatisticModel import *


def main():
    '''
    main function to predict article category. 
    SVM and RF algorithms preformed on train data, RF prediction function used to create prediction on test data
    output: id,prediction column to csv file 
    '''
    df = pd.DataFrame.from_csv('TrainDataset.csv')
    DataExplore.check_nans(df)
    #ignoring text features
    df_continues = DataPreperation.choose_features(df)
    DataExplore.difference_t_test(df_continues)
    DataExplore.check_corr(df_continues)
    # removing feature according to correlation heat map
    df_continues = DataPreperation.choose_features(df_continues,cols=['articleId','isCategory','nGramCnt', 'scores.betweeness', 'scores.closeness', 'scores.degree', 'scores.fromBegScore',
                    'scores.relDocsRatio', 'scores.relVsIrrelDocs',
                    'scores.relVsIrrelTerms', 'scores.relevance'])
    DataExplore.check_corr(df_continues)
    df_continues = DataPreperation.handleNAs_cont(df_continues)
    df_binary = DataPreperation.choose_features(df, cols = ['isCategory','scores.isInTag-a','scores.isInTag-b','scores.isInTag-h1','scores.isInTag-h2','scores.isInTag-h3','scores.isInTag-h4','scores.isInTag-h5','scores.isInTag-h6','scores.isInTag-i','scores.isInTag-strong','scores.isInTitle'])
    df_binary = DataPreperation.handleNAs_bin(df_binary)
    DataExplore.mean_by_group(df_binary)
    #choosing feature by mean group examination
    df_binary = DataPreperation.choose_features(df_binary, cols = ['scores.isInTitle'])
    data = pd.concat([df_binary,df_continues],axis=1)
    train, test = train_test_split(data, test_size=0.3,random_state = 100)
    train_unbias = DataPreperation.under_sampeling2(train)
    X_train,y_train,X_test,y_test = DataPreperation.split_X_y(train_unbias,test)
    #X_train,X_test = DataPreperation.featureEngineer(X_train,X_test)

    #RF
    rand_forest = RandomForest()
    y_pred_rand_forest,random_forest_trained = rand_forest.prediction_model(X_train,y_train,X_test,y_test)
    rand_forest.accuracy(y_pred_rand_forest,y_test)

    #SVM
    svm = SVM()
    y_pred_SVM = svm.prediction_model(X_train,y_train,X_test,y_test)
    svm.accuracy(y_pred_SVM,y_test)

    #predicting test data by random forest predictor
    df_test = pd.DataFrame.from_csv('TestDataset.csv')
    DataPreperation.prepare_data_to_test(df_test,random_forest_trained)


if __name__ == '__main__':
    main()


