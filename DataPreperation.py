# coding=utf-8
import numpy as np
import pandas as pd

class DataPreperation:
    '''
    aggregates functions related to data changing
    '''

    @staticmethod
    def split_X_y(train,test):
        '''
        :param train: train data 
        :param test: test data
        :return: splited data to label and features
        '''
        X_train = train.drop(["isCategory"], axis=1)
        y_train = train["isCategory"]
        X_test =test.drop(["isCategory"], axis = 1)
        y_test = test["isCategory"]
        return X_train,y_train,X_test,y_test

    @staticmethod
    def handleNAs_bin(data):
        '''
        :param data: binary data 
        :return: data filled with 0's instead of NAs 
        '''
        html_cols = ['scores.isInTag-a','scores.isInTag-b','scores.isInTag-h1','scores.isInTag-h2','scores.isInTag-h3','scores.isInTag-h4','scores.isInTag-h5','scores.isInTag-h6','scores.isInTag-i','scores.isInTag-strong','scores.isInTitle']
        data.loc[:,html_cols] = data.loc[:,html_cols].fillna(0)
        return data

    @staticmethod
    def handleNAs_cont(data):
        '''
        :param data: continues data 
        :return: data field with column mean instead of NAs
        '''
        data = data.fillna(data.mean())
        return data


    @staticmethod
    def featureEngineer(X_train,X_test):
        '''
        adding rank feature within the same article corresponding to value of feature
        Example: under id 100 3 keywords with relevance values 0.3,0.2,0.5 – rank feature will be filled with values 2 1 3 respectively.   
        :param X_train: features in train data 
        :param X_test: features in test data
        :return: train and test data with new feature
        '''
        X_train["rank_relevance"] = X_train.groupby("articleId")["scores.relevance"].rank(ascending=False)
        X_test["rank_relevance"] = X_test.groupby("articleId")["scores.relevance"].rank(ascending=False)
        return X_train,X_test

    @staticmethod
    def under_sampeling2(data):
        '''
        equaling the size of classified category/not-category in the train data by randomly removing rows
        :param data: 
        :return: data with equal size of rows category/not-category
        '''
        category_false = data[data['isCategory'] == 0]
        category_true = data[data['isCategory']==1]
        num_category = sum(category_true['isCategory'])
        category_false_subset = category_false.sample(num_category)
        return category_false_subset.append(category_true)

    @staticmethod
    def choose_features(data, cols = ['articleId','isCategory','nGramCnt', 'scores.betweeness', 'scores.closeness', 'scores.degree', 'scores.fromBegScore',
                    'scores.load', 'scores.relDocsRatio', 'scores.relTermsRatio', 'scores.relVsIrrelDocs',
                    'scores.relVsIrrelTerms', 'scores.relevance', 'termOccurrence']):
        '''
        :param data:  
        :param cols: columns to be selected
        :return: data with selected columns
        '''
        return data[cols]

    @staticmethod
    def prepare_data_to_test(df_test,model):

        df_continues_test = DataPreperation.choose_features(df_test)
        df_continues_test = DataPreperation.choose_features(df_continues_test,
                                                            cols=['articleId', 'nGramCnt', 'scores.betweeness',
                                                                  'scores.closeness', 'scores.degree',
                                                                  'scores.fromBegScore',
                                                                  'scores.relDocsRatio', 'scores.relVsIrrelDocs',
                                                                  'scores.relVsIrrelTerms', 'scores.relevance',
                                                                  ])
        df_continues_test = DataPreperation.handleNAs_cont(df_continues_test)
        df_binary_test = DataPreperation.choose_features(df_test,
                                                         cols=['isCategory', 'scores.isInTag-a', 'scores.isInTag-b',
                                                               'scores.isInTag-h1', 'scores.isInTag-h2',
                                                               'scores.isInTag-h3', 'scores.isInTag-h4',
                                                               'scores.isInTag-h5', 'scores.isInTag-h6',
                                                               'scores.isInTag-i', 'scores.isInTag-strong',
                                                               'scores.isInTitle'])
        df_binary_test = DataPreperation.handleNAs_bin(df_binary_test)
        # DataExplore.mean_by_group(df_binary)
        df_binary_test = DataPreperation.choose_features(df_binary_test, cols=['scores.isInTitle'])
        data_test = pd.concat([df_binary_test, df_continues_test], axis=1)
        pred = model.predict(data_test)
        y_pred_test = pred
        y_pred_test[y_pred_test >= 0.5] = 1
        y_pred_test[y_pred_test < 0.5] = 0
        submit_idx = df_test.index.values.tolist()
        df_id = pd.DataFrame(submit_idx,columns=['id'])
        df_pred = pd.DataFrame(y_pred_test,columns=['isCategory'])
        df_submit = pd.concat([df_id, df_pred], axis=1)
        df_submit.to_csv('y_test_prediction.csv',index=False)