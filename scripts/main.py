from numpy import mean
from numpy import std
from sklearn.datasets import load_diabetes #make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE

# create dataset
X, y = load_diabetes(return_X_y=True)
# configure the cross-validation procedure
cv_outer = KFold(n_splits=5, shuffle=True, random_state=1) #was 10
# enumerate splits
outer_results = list()
for train_ix, test_ix in cv_outer.split(X):
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    # configure the cross-validation procedure
    cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
    # define the model
    model = tree.DecisionTreeRegressor()
    # define search space
    space = dict()
    space['max_depth'] = [3, 4,5]
    space['min_samples_split'] = [2, 4, 6, 8, 10]
    # define search
    search = GridSearchCV(model, space, scoring='neg_mean_squared_error', cv=cv_inner, refit=True)
    # execute search
    result = search.fit(X_train, y_train)
    # get the best performing model fit on the whole training set
    best_model = result.best_estimator_
    # evaluate model on the hold out dataset
    yhat = best_model.predict(X_test)
    # evaluate the model
    mse = MSE(y_test, yhat)
    # store the result
    outer_results.append(mse)
    # report progress
    print('>mse=%.3f, est=%.3f, cfg=%s' % (mse, result.best_score_, result.best_params_))
# summarize the estimated performance of the model
print('MSE: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))

