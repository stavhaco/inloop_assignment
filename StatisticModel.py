from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn import svm, grid_search, datasets
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

class Model(object):

    def accuracy(self,y_pred,y_test):
        conf_matrix = confusion_matrix(y_pred, y_test)
        print conf_matrix
        true_neg, false_pos, false_neg, true_pos = conf_matrix[1][1], conf_matrix[1][0], conf_matrix[0][1], \
                                                   conf_matrix[0][0]
        print "prediction of true negative - correct classification of non-category " + str(round(float(true_neg) / (true_neg + false_neg), 2))
        print "prediction of true positive - correct classifiction of category " + str(round(float(true_pos) / (true_pos + false_pos), 2))



class RandomForest(Model):

    def prediction_model(self,X_train,y_train,X_test,y_test):
        rf = RandomForestRegressor(random_state=42)
        print(rf.get_params())
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        rf = RandomForestRegressor()
        #rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
        #                               random_state=42, n_jobs=-1)
        # grid of parameter combinations ran and provided the following optimized parameters:
        rf_random = RandomForestRegressor(bootstrap=True,min_samples_leaf=4,n_estimators=800,max_features='sqrt',min_samples_split=10,max_depth=50)
        rf_random.fit(X_train, y_train)
        predictions = rf_random.predict(X_test)

        y_pred = predictions
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        return y_pred, rf_random

class SVM(Model):

    def prediction_model(self,X_train,y_train,X_test,y_test):
        iris = datasets.load_iris()
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1.5, 10]}
        svr = svm.SVC()
        clf = grid_search.GridSearchCV(svr, parameters)
        clf.fit(iris.data, iris.target)
        print clf.best_params_
        model = svm.SVC(kernel='linear', C=1.5, gamma=1)
        model.fit(X_train, y_train)
        model.score(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred
