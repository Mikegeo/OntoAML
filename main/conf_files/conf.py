import numpy as np

""""
For HyperParameter tuning with range and np.arange we go
k = range(2,21) or
k = np.arange(0.05, 1.01, 0.05)
#then for example
parameters  = {'min_sample_split' : k}
"""

algorithms_config_dict = {

    # Classifiers
    'sklearn.naive_bayes.GaussianNB': {
    },

    'sklearn.naive_bayes.BernoulliNB': {
        'BernoulliNB__alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'BernoulliNB__fit_prior': [True, False]
    },

    'sklearn.naive_bayes.MultinomialNB': {
        'MultinomialNB__alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'MultinomialNB__fit_prior': [True, False]
    },

    'sklearn.tree.DecisionTreeClassifier': {
        'DecisionTreeClassifier__criterion': ["gini", "entropy"],
        'DecisionTreeClassifier__max_depth': range(1, 11),
        'DecisionTreeClassifier__min_samples_split': np.arange(2, 21, 1),
        'DecisionTreeClassifier__min_samples_leaf': np.arange(1, 21, 1)
    },

    'sklearn.ensemble.ExtraTreesClassifier': {
        # 'ExtraTreesClassifier__n_estimators': [100],
        'ExtraTreesClassifier__criterion': ["gini", "entropy"],
        'ExtraTreesClassifier__max_features': np.arange(0.05, 1.01, 0.05),
        'ExtraTreesClassifier__min_samples_split': np.arange(2, 21, 1),
        'ExtraTreesClassifier__min_samples_leaf': np.arange(1, 21, 1),
        'ExtraTreesClassifier__bootstrap': [True, False]
    },

    'sklearn.ensemble.RandomForestClassifier': {
        # 'RandomForestClassifier__n_estimators': [100],
        'RandomForestClassifier__criterion': ["gini", "entropy"],
        'RandomForestClassifier__max_features': np.arange(0.05, 1.01, 0.05),
        'RandomForestClassifier__min_samples_split': np.arange(2, 21, 1),
        'RandomForestClassifier__min_samples_leaf':  np.arange(1, 21, 1),
        'RandomForestClassifier__bootstrap': [True, False]
    },

    'sklearn.ensemble.GradientBoostingClassifier': {
       # 'GradientBoostingClassifier__n_estimators': [100],
        'GradientBoostingClassifier__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'GradientBoostingClassifier__max_depth': range(1, 11),
        'GradientBoostingClassifier__min_samples_split': np.arange(2, 21, 2),
        'GradientBoostingClassifier__min_samples_leaf': np.arange(1, 21, 2),
        'GradientBoostingClassifier__subsample': np.arange(0.05, 1.01, 0.05),
       # 'GradientBoostingClassifier__max_features': np.arange(0.05, 1.01, 0.05)
    },

    'sklearn.ensemble.AdaBoostClassifier': {
         'AdaBoostClassifier__n_estimators': [50, 100],
         'AdaBoostClassifier__random_state': [0]
    },

    'sklearn.neighbors.KNeighborsClassifier': {
        'KNeighborsClassifier__n_neighbors': np.arange(1, 101, 1),
        'KNeighborsClassifier__weights': ["uniform", "distance"],
        'KNeighborsClassifier__p': [1, 2]
    },

    'sklearn.svm.LinearSVC': {
        'LinearSVC__loss': ["hinge", "squared_hinge"],
        'LinearSVC__dual': [True, False],
        'LinearSVC__tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'LinearSVC__C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'LinearSVC__penalty': ["l1", "l2"]
    },

    'sklearn.linear_model.LogisticRegression': {
        'LogisticRegression__C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'LogisticRegression__dual': [True, False],
        'LogisticRegression__penalty': ["l1", "l2"]
    },

    'xgboost.XGBClassifier': {
        #'XGBClassifier__n_estimators': [100],
        'XGBClassifier__max_depth': range(1, 11),
        'XGBClassifier__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'XGBClassifier__subsample': np.arange(0.05, 1.01, 0.05),
        'XGBClassifier__min_child_weight': range(1, 21),
        'XGBClassifier__nthread': [1]
    },

    'sklearn.linear_model.SGDClassifier': {
        'SGDClassifier__loss': ['log', 'hinge', 'modified_huber', 'squared_hinge', 'perceptron'],
        'SGDClassifier__penalty': ['elasticnet'],
        'SGDClassifier__alpha': [0.0, 0.01, 0.001],
        'SGDClassifier__learning_rate': ['invscaling', 'constant'],
        'SGDClassifier__fit_intercept': [True, False],
        'SGDClassifier__l1_ratio': [0.25, 0.0, 1.0, 0.75, 0.5],
        'SGDClassifier__eta0': [0.1, 1.0, 0.01],
        'SGDClassifier__power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]
    },

    'sklearn.neural_network.MLPClassifier': {
        'MLPClassifier__activation': ['tanh', 'relu'],
        #'MLPClassifier__alpha': [1e-4, 1e-3, 1e-2, 1e-1],
        'MLPClassifier__learning_rate_init': [1e-3, 1e-2, 1e-1, 0.5, 1.]
    },

    # Preprocesssors
    'sklearn.preprocessing.Binarizer': {
        'Binarizer__threshold': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.decomposition.FastICA': {
        'FastICA__tol': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.cluster.FeatureAgglomeration': {
        'FeatureAgglomeration__linkage': ['ward', 'complete', 'average'],
        'FeatureAgglomeration__affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    },

    'sklearn.preprocessing.MaxAbsScaler': {
    },

    'sklearn.preprocessing.MinMaxScaler': {
    },

    'sklearn.preprocessing.Normalizer': {
        'Normalizer__norm': ['l1', 'l2', 'max']
    },

    'sklearn.kernel_approximation.Nystroem': {
        'Nystroem__kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
        'Nystroem__gamma': np.arange(0.0, 1.01, 0.05),
        'Nystroem__n_components': range(1, 11)
    },

    'sklearn.decomposition.PCA': {
       # 'PCA__svd_solver': ['randomized'],
        'PCA__iterated_power': range(1, 11)
    },

    'sklearn.preprocessing.PolynomialFeatures': {
        'PolynomialFeatures__degree': [2],
        'PolynomialFeatures__include_bias': [False],
        'PolynomialFeatures__interaction_only': [False]
    },

    'sklearn.kernel_approximation.RBFSampler': {
        'RBFSampler__gamma': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.preprocessing.RobustScaler': {
    },

    'sklearn.preprocessing.StandardScaler': {
    },

    'sklearn.preprocessing.OneHotEncoder': {
    },

    'sklearn.impute.SimpleImputer': {
    },

    # Selectors
    'sklearn.feature_selection.SelectFwe': {
        'SelectFwe__alpha': np.arange(0, 0.05, 0.001),
        'SelectFwe__score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },

    # added by me need check # add it to the names of models check it again
    'sklearn.feature_selection.SelectKBest_kr': {
        'SelectKBest_k__f_regression': {
            'SelectKBest_k__k': range(1, 10)
        },
    },

    'sklearn.feature_selection.SelectKBest.chi2': {
        'SelectKBest__chi2': {
            'k': range(1, 20)
        }
    },

    'sklearn.feature_selection.SelectPercentile': {
        'SelectPercentile__percentile': range(1, 100),
        'SelectPercentile__score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },

    'sklearn.feature_selection.VarianceThreshold': {
        'VarianceThreshold__threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },
    # RFE(ExtraTreesClassifier)
    'sklearn.feature_selection.RFE': {
        'RFE(ExtraTreesClassifier)__step': np.arange(0.05, 1.01, 0.05),
        'RFE__estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                # 'RFE(ExtraTreesClassifier)__estimator__n_estimators': [100],
                'RFE(ExtraTreesClassifier)__estimator__criterion': ['gini', 'entropy'],
                'RFE(ExtraTreesClassifier)__estimator__max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    },
    # RFE(SVR)
    'sklearn.feature_selection.RFE1': {
        'RFE(SVR)__step': np.arange(0.05, 1.01, 0.05),
        'RFE__estimator': {
            'sklearn.svm.SVR': {
                'RFE(SVR)__estimator__kernel': ['linear', 'rbf', 'poly'],
                'RFE(SVR)__estimator__C': np.arange(1.5, 10),
                'RFE(SVR)__estimator__gamma': [1e-7, 1e-4],
                'RFE(SVR)__estimator__epsilon': [0.1, 0.2, 0.5, 0.3]
            }
        }
    },
    # SelectFromModel(ExtraTreesClassifier)
    'sklearn.feature_selection.SelectFromModel': {
        'SelectFromModel(ExtraTreesClassifier)__threshold': np.arange(0, 1.01, 0.05),
        'SelectFromModel__estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'SelectFromModel(ExtraTreesClassifier)__estimator__n_estimators': [100],
                'SelectFromModel(ExtraTreesClassifier)__estimator__criterion': ['gini', 'entropy'],
                'SelectFromModel(ExtraTreesClassifier)__estimator__max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    },
    # SelectFromModel_L1(Lasso)
    'sklearn.feature_selection.SelectFromModel_L1': {
        'SelectFromModel_L1__threshold': np.arange(0, 1.01, 0.05),
        'SelectFromModel_L1__estimator': {
            'sklearn.linear_model.LogisticRegression': {
                'SelectFromModel_L1__estimator__C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
                # 'SelectFromModel_L1__estimator__dual': [True, False]
            }
        }
    },
    # elastic net feature selection
    'sklearn.feature_selection.ElasticNet': {
        #'SelectFromModel_ElasticNet__threshold': np.arange(0, 1.01, 0.05),
        'SelectFromModel_ElasticNet__estimator': {
            'sklearn.linear_model.LogisticRegression': {
                'SelectFromModel_ElasticNet__estimator__alpha': [0.01, 0.1, 1.0, 10.0],
                'SelectFromModel_ElasticNet__estimator__l1_ratio': [0.25, 0.5, 0.75]
            }
        }
    },

    # Regression algorithms
    'sklearn.linear_model.ElasticNetCV': {
        'ElasticNetCV__l1_ratio': np.arange(0.0, 1.01, 0.05),
        'ElasticNetCV__tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    },

    'sklearn.ensemble.ExtraTreesRegressor': {
        # 'ExtraTreesRegressor__n_estimators': [100],
        'ExtraTreesRegressor__max_features': np.arange(0.05, 1.01, 0.05),
        'ExtraTreesRegressor__min_samples_split': range(2, 21),
        'ExtraTreesRegressor__min_samples_leaf': range(1, 21),
        'ExtraTreesRegressor__bootstrap': [True, False]
    },

    'sklearn.ensemble.GradientBoostingRegressor': {
        # 'GradientBoostingRegressor__n_estimators': [100],
        'GradientBoostingRegressor__loss': ["squared_error", "lad", "huber", "quantile"],
        'GradientBoostingRegressor__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'GradientBoostingRegressor__max_depth': range(1, 11),
        'GradientBoostingRegressor__min_samples_split': np.arange(2, 21, 2),
        'GradientBoostingRegressor__min_samples_leaf': np.arange(1, 21, 2),
        'GradientBoostingRegressor__subsample': np.arange(0.05, 1.01, 0.05),
        'GradientBoostingRegressor__max_features': np.arange(0.05, 1.01, 0.05),
        # 'GradientBoostingRegressor__alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    },

    'sklearn.ensemble.AdaBoostRegressor': {
        'AdaBoostRegressor__n_estimators': [100],
        'AdaBoostRegressor__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'AdaBoostRegressor__loss': ["linear", "square", "exponential"]
    },

    'sklearn.tree.DecisionTreeRegressor': {
        'DecisionTreeRegressor__max_depth': range(1, 11),
        'DecisionTreeRegressor__min_samples_split': np.arange(2, 21, 2),
        'DecisionTreeRegressor__min_samples_leaf': np.arange(1, 21, 2)
    },

    'sklearn.neighbors.KNeighborsRegressor': {
        'KNeighborsRegressor__n_neighbors': range(1, 101),
        'KNeighborsRegressor__weights': ["uniform", "distance"],
        'KNeighborsRegressor__p': [1, 2]
    },

    'sklearn.linear_model.LassoLarsCV': {
        'LassoLarsCV__normalize': [True, False]
    },

    'sklearn.svm.LinearSVR': {
        'LinearSVR__loss': ["epsilon_insensitive", "squared_epsilon_insensitive"],
        'LinearSVR__dual': [True, False],
        'LinearSVR__tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'LinearSVR__C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'LinearSVR__epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1.]
    },

    'sklearn.ensemble.RandomForestRegressor': {
        # 'RandomForestRegressor__n_estimators': [100],
        'RandomForestRegressor__max_features': np.arange(0.05, 1.01, 0.05),
        'RandomForestRegressor__min_samples_split': np.arange(2, 21, 2),
        'RandomForestRegressor__min_samples_leaf': np.arange(1, 21, 2),
        'RandomForestRegressor__bootstrap': [True, False]
    },

    'sklearn.linear_model.RidgeCV': {
    },

    'xgboost.XGBRegressor': {
        # 'XGBRegressor__n_estimators': [100],
        'XGBRegressor__max_depth': range(1, 11),
        'XGBRegressor__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'XGBRegressor__subsample': np.arange(0.05, 1.01, 0.05),
        'XGBRegressor__min_child_weight': range(1, 21),
        'XGBRegressor__nthread': [1],
        'XGBRegressor__objective': ['reg:squarederror']
    },

    'sklearn.linear_model.SGDRegressor': {
        'SGDRegressor__loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
        'SGDRegressor__penalty': ['elasticnet'],
        'SGDRegressor__alpha': [0.0, 0.01, 0.001],
        'SGDRegressor__learning_rate': ['invscaling', 'constant'],
        'SGDRegressor__fit_intercept': [True, False],
        'SGDRegressor__l1_ratio': [0.25, 0.0, 1.0, 0.75, 0.5],
        'SGDRegressor__eta0': [0.1, 1.0, 0.01],
        'SGDRegressor__power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]
    },

    'sklearn.neural_network.MLPRegressor': {
        'MLPRegressor__activation': ['tanh', 'relu'],
        #'MLPRegressor_alpha': [1e-4, 1e-3, 1e-2, 1e-1],
        'MLPRegressor__learning_rate_init': [1e-3, 1e-2, 1e-1, 0.5, 1.]
    },

    'sklearn.linear_model.LinearRegression': {
    }
}

model_import_conf = {
    'sklearn.ensemble.AdaBoostClassifier': 'from sklearn.ensemble import AdaBoostClassifier',

    'sklearn.ensemble.AdaBoostRegressor': 'from sklearn.ensemble import AdaBoostRegressor',

    'sklearn.feature_selection.f_classif': 'from sklearn.feature_selection import SelectKBest, f_classif',  # ANOVA

    'sklearn.feature_selection.SelectKBest.chi2': 'from sklearn.feature_selection import SelectKBest, chi2',  # chi2

    # CHECK BACKWARD ELIMINATION
    'sklearn.naive_bayes.BernoulliNB': 'from sklearn.naive_bayes import BernoulliNB',

    'sklearn.preprocessing.Binarizer': 'from sklearn.preprocessing import Binarizer',

    # correlation CFS CHECK
    'sklearn.tree.DecisionTreeClassifier': 'from sklearn.tree import DecisionTreeClassifier',

    'sklearn.tree.DecisionTreeRegressor': 'from sklearn.tree import DecisionTreeRegressor',

    'sklearn.linear_model.ElasticNetCV': 'from sklearn.linear_model import ElasticNet',

    'sklearn.metrics.pairwise.euclidean_distances': 'from sklearn.metrics.pairwise import euclidean_distances',# check

    # check exhaustive feature selection
    'sklearn.ensemble.ExtraTreesClassifier': 'from sklearn.ensemble import ExtraTreesClassifier',

    'sklearn.ensemble.ExtraTreesRegressor': 'from sklearn.ensemble import ExtraTreesRegressor',

    'sklearn.decomposition.FastICA': 'from sklearn.decomposition import FastICA',

    'sklearn.cluster.FeatureAgglomeration': 'from sklearn import cluster',

    'sklearn.naive_bayes.GaussianNB': 'from sklearn.naive_bayes import GaussianNB',

    'sklearn.ensemble.GradientBoostingClassifier': 'from sklearn.ensemble import GradientBoostingClassifier',

    'sklearn.ensemble.GradientBoostingRegressor': 'from sklearn.ensemble import GradientBoostingRegressor',

    # chech interative imputer && KBinsDiscretizer

    'sklearn.neighbors.KNeighborsClassifier': 'from sklearn.neighbors import KNeighborsClassifier',

    'sklearn.neighbors.KNeighborsRegressor': 'from sklearn.neighbors import KNeighborsRegressor',

    # check lasso regularization and do the same from select from model extra trees
    'sklearn.linear_model.LassoLarsCV': 'from sklearn.linear_model import LarsCV',

    'sklearn.linear_model.LinearRegression': 'from sklearn.linear_model import LinearRegression',

    'sklearn.svm.LinearSVC': 'from sklearn.svm import LinearSVC',

    'sklearn.svm.LinearSVR': 'from sklearn.svm import LinearSVR',

    'sklearn.linear_model.LogisticRegression': 'from sklearn.linear_model import LogisticRegression',

    'sklearn.preprocessing.MaxAbsScaler': 'from sklearn.preprocessing import MaxAbsScaler',

    'sklearn.preprocessing.MinMaxScaler': 'from sklearn.preprocessing import MinMaxScaler',

    'sklearn.neural_network.MLPClassifier': 'from sklearn.neural_network import MLPClassifier',

    'sklearn.neural_network.MLPRegressor': 'from sklearn.neural_network import MLPRegressor',

    'sklearn.naive_bayes.MultinomialNB': 'from sklearn.naive_bayes import MultinomialNB',

     # mitual info class and regress need primitives

    'sklearn.preprocessing.Normalizer': 'from sklearn.preprocessing import Normalizer',

    'sklearn.kernel_approximation.Nystroem': 'from sklearn.kernel_approximation import Nystroem', # what is this?

    # check oneClassSVM -> OUTLIER DETECTOR

    'sklearn.preprocessing.OneHotEncoder': 'from sklearn.preprocessing import OneHotEncoder',

    'sklearn.decomposition.PCA': 'from sklearn.decomposition import PCA',

    'sklearn.preprocessing.PolynomialFeatures': 'from sklearn.preprocessing import PolynomialFeatures',

    'sklearn.ensemble.RandomForestClassifier': 'from sklearn.ensemble import RandomForestClassifier',

    'sklearn.ensemble.RandomForestRegressor': 'from sklearn.ensemble import RandomForestRegressor',

    'sklearn.kernel_approximation.RBFSampler': 'from sklearn.kernel_approximation import RBFSampler',

    'sklearn.feature_selection.RFE': {'1': 'from sklearn.feature_selection import RFE',
                                       '2': 'from sklearn.ensemble import ExtraTreesClassifier'},

    'sklearn.feature_selection.RFE1': {'1': 'from sklearn.feature_selection import RFE',
                                       '2': 'from sklearn.svm import SVR'},

    'sklearn.linear_model.RidgeCV': 'from sklearn.linear_model import RidgeCV',

    'sklearn.preprocessing.RobustScaler': 'from sklearn.preprocessing import RobustScaler',

    'sklearn.feature_selection.SelectFromModel': {'1': 'from sklearn.feature_selection import SelectFromModel',
                                                  '2': 'from sklearn.ensemble import ExtraTreesClassifier'},

    'sklearn.feature_selection.SelectFromModel_L1': {'1': 'from sklearn.feature_selection import SelectFromModel',
                                                     '2': 'from sklearn.linear_model import LogisticRegression'},

    'sklearn.feature_selection.ElasticNet': {'1': 'from sklearn.feature_selection import SelectFromModel',
                                             '2': 'from sklearn.linear_model import ElasticNet'},

    'sklearn.feature_selection.SelectKBest_kr':
        'from sklearn.feature_selection import SelectKBest, mutual_info_regression',

    'sklearn.feature_selection.SelectFwe': 'from sklearn.feature_selection import SelectFwe',# need check

    'sklearn.feature_selection.SelectPercentile': 'from sklearn.feature_selection import SelectPercentile, f_classif',

    # check sequantial feature selection
    'sklearn.linear_model.SGDClassifier': 'from sklearn.linear_model import SGDClassifier',

    'sklearn.linear_model.SGDRegressor': 'from sklearn.linear_model import SGDRegressor',

    'sklearn.impute.SimpleImputer': 'from sklearn.impute import SimpleImputer',

    'sklearn.preprocessing.StandardScaler': 'from sklearn.preprocessing import StandardScaler',

    # CHECK TSNE

    'sklearn.feature_selection.VarianceThreshold': 'from sklearn.feature_selection import VarianceThreshold',

    # check XGBOOST classifier and regressor

}

model_names_for_pipeline_dict = {
    # Classifiers
    'sklearn.naive_bayes.GaussianNB': 'GaussianNB',
    'sklearn.naive_bayes.BernoulliNB': 'BernoulliNB',
    'sklearn.naive_bayes.MultinomialNB': 'MultinomialNB',
    'sklearn.tree.DecisionTreeClassifier': 'DecisionTreeClassifier',
    'sklearn.ensemble.ExtraTreesClassifier': 'ExtraTreesClassifier',
    'sklearn.ensemble.RandomForestClassifier': 'RandomForestClassifier',
    'sklearn.ensemble.GradientBoostingClassifier': 'GradientBoostingClassifier',
    'sklearn.ensemble.AdaBoostClassifier': 'AdaBoostClassifier',
    'sklearn.neighbors.KNeighborsClassifier': 'KNeighborsClassifier',
    'sklearn.svm.LinearSVC': 'LinearSVC',
    'sklearn.linear_model.LogisticRegression': 'LogisticRegression',
    'xgboost.XGBClassifier': 'XGBClassifier',
    'sklearn.linear_model.SGDClassifier': 'SGDClassifier',
    'sklearn.neural_network.MLPClassifier': 'MLPClassifier',
    # Preprocessors
    'sklearn.preprocessing.Binarizer': 'Binarizer',
    'sklearn.cluster.FeatureAgglomeration': 'FeatureAgglomeration',
    'sklearn.preprocessing.MaxAbsScaler': 'MaxAbsScaler',
    'sklearn.preprocessing.MinMaxScaler': 'MinMaxScaler',
    'sklearn.preprocessing.Normalizer': 'Normalizer',
    'sklearn.kernel_approximation.Nystroem': 'Nystroem',
    'sklearn.preprocessing.PolynomialFeatures': 'PolynomialFeatures',
    'sklearn.kernel_approximation.RBFSampler': 'RBFSampler',
    'sklearn.preprocessing.RobustScaler': 'RobustScaler',
    'sklearn.preprocessing.StandardScaler': 'StandardScaler',
    'sklearn.preprocessing.OneHotEncoder': 'OneHotEncoder',
    'sklearn.impute.SimpleImputer': 'SimpleImputer',
    # Dimensionality reduction
    'sklearn.decomposition.FastICA': 'FastICA',
    'sklearn.decomposition.PCA': {'PCA': 'PCA(svd_solver=\'randomized\')'},
    # Selectors
    'sklearn.feature_selection.SelectKBest.chi2': {'SelectKBest_chi2': 'SelectKBest(chi2)'},
    'sklearn.feature_selection.SelectFwe': 'SelectFwe',
    'sklearn.feature_selection.SelectPercentile': 'SelectPercentile',
    'sklearn.feature_selection.VarianceThreshold': 'VarianceThreshold',
    'sklearn.feature_selection.RFE': {'RFE': 'RFE(ExtraTreesClassifier)'},
    'sklearn.feature_selection.RFE1': {'RFE': 'RFE(SVR)'},
    'sklearn.feature_selection.SelectFromModel': {'SelectFromModel_ExtraTreeClas':
                                                      'SelectFromModel(ExtraTreesClassifier)'},
    'sklearn.feature_selection.SelectFromModel_L1': {'SelectFromModel_L1': 'SelectFromModel(LogisticRegression(penalty'
                                                                           '=\'l1\', solver=\'liblinear\'))'},
    'sklearn.feature_selection.SelectKBest_kr': {'SelectKBest_k': 'SelectKBest(score_func=mutual_info_regression)'},
    'sklearn.feature_selection.ElasticNet': {'SelectFromModel_ElasticNet': 'SelectFromModel(ElasticNet())'},

    # Regressors
    'sklearn.linear_model.ElasticNetCV': 'ElasticNetCV',
    'sklearn.ensemble.ExtraTreesRegressor': 'ExtraTreesRegressor',
    'sklearn.ensemble.GradientBoostingRegressor': 'GradientBoostingRegressor',
    'sklearn.ensemble.AdaBoostRegressor': 'AdaBoostRegressor',
    'sklearn.tree.DecisionTreeRegressor': 'DecisionTreeRegressor',
    'sklearn.neighbors.KNeighborsRegressor': 'KNeighborsRegressor',
    'sklearn.linear_model.LassoLarsCV': 'LassoLarsCV',
    'sklearn.svm.LinearSVR': 'LinearSVR',
    'sklearn.ensemble.RandomForestRegressor': 'RandomForestRegressor',
    'sklearn.linear_model.RidgeCV': 'RidgeCV',
    'xgboost.XGBRegressor': 'XGBRegressor',
    'sklearn.linear_model.SGDRegressor': 'SGDRegressor',
    'sklearn.neural_network.MLPRegressor': 'MLPRegressor',
    'sklearn.linear_model.LinearRegression': 'LinearRegression'
}

algorithms_explainability_dict = {
    # Preprocessors
    'sklearn.preprocessing.Binarizer': 'Binarizer',
    'sklearn.cluster.FeatureAgglomeration': 'FeatureAgglomeration',
    'sklearn.preprocessing.MaxAbsScaler': 'MaxAbsScaler',
    'sklearn.preprocessing.Normalizer': 'Normalizer',
    'sklearn.kernel_approximation.Nystroem': 'Nystroem',
    'sklearn.preprocessing.PolynomialFeatures': 'PolynomialFeatures',
    'sklearn.kernel_approximation.RBFSampler': 'RBFSampler',
    'sklearn.preprocessing.MinMaxScaler': 'Many machine learning algorithms perform better or converge faster when '
                                          'features are on a relatively similar scale and/or close to normally '
                                          'distributed. This Pipeline uses MinMaxScaler algorithm, which will '
                                          'transform each value in the column proportionally within the range [0,1]. '
                                          'It’s non-distorting.',
    'sklearn.preprocessing.RobustScaler': 'Many machine learning models perform better or converge faster when '
                                          'features are on a relatively similar scale and/or close to normally '
                                          'distributed. This Pipeline uses Robust-Scaler algorithm, which is '
                                          'recommended if the the data-set has several outliers. This method scale the'
                                          'features but in this case using statistics that are robust to outliers.',
    'sklearn.preprocessing.StandardScaler': 'Many machine learning algorithms perform better or converge faster when '
                                            'features are on a relatively similar scale and/or close to normally '
                                            'distributed. This Pipeline uses Standard-Scaler algorithm which follows '
                                            'Standard Normal Distribution (SND). Therefore, it transforms each value '
                                            'in the column to range about the mean 0 and standard deviation 1, ie, '
                                            'each value will be normalised by subtracting the mean and dividing by '
                                            'standard deviation. ',
    'sklearn.feature_selection.ElasticNet': 'ElasticNet is a type of linear regression that combines L1 (Lasso) and'
                                            ' L2 (Ridge) regularization. It is commonly used for feature selection '
                                            'because it can shrink the coefficients of less important features to zero,'
                                            ' effectively removing them from the model. This can improve the models '
                                            'performance and reduce overfitting',
    'sklearn.preprocessing.OneHotEncoder': 'The Pipeline is using One-Hot-Encoding to encode the categorical values'
                                           'of the data-set before pass them to model, most of the models need One-hot-'
                                           'encoding, this algorithm transforms the value from a category to numerical.',
    'sklearn.impute.SimpleImputer': 'The Pipeline is using Simple-Imputer to impute the missing values of the data-set'
                                    'before pass them to the model.',
    # Dimensionality reduction
    'sklearn.decomposition.FastICA': 'FastICA',
    'sklearn.decomposition.PCA': 'This Pipeline contains PCA algorithm (is a well-known unsupervised dimensionality'
                                 ' reduction technique that constructs relevant features/variables through linear '
                                 '(linear PCA) or non-linear (kernel PCA) combinations of the original variables '
                                 '(features).) PCA technique is particularly useful in processing data where '
                                 'multi-colinearity exists between the features/variables. PCA can be used when '
                                 'the dimensions of the input features are high (e.g. a lot of variables).',
    # Selectors
    'sklearn.feature_selection.SelectFwe': 'SelectFwe',
    'sklearn.feature_selection.SelectPercentile': 'SelectPercentile',
    'sklearn.feature_selection.VarianceThreshold': 'VarianceThreshold',
    'sklearn.feature_selection.RFE': {'RFE': 'RFE(ExtraTreesClassifier)'},
    'sklearn.feature_selection.RFE1': {'RFE': 'RFE(SVR)'},
    'sklearn.feature_selection.SelectFromModel': {'SelectFromModel_ExtraTreeClas':
                                                  'SelectFromModel(ExtraTreesClassifier)'},
    'sklearn.feature_selection.SelectFromModel_L1': 'The Pipeline contains Lasso feature selection. In statistics and '
                                                    'machine learning, lasso (least absolute '
                                                    'shrinkage and selection operator; also Lasso or LASSO) is a '
                                                    'regression analysis method that performs both variable selection '
                                                    'and regularization in order to enhance the prediction accuracy '
                                                    'and interpretability of the resulting statistical model.',
    'sklearn.feature_selection.SelectKBest.chi2': ' This Pipeline contains Chi-squared (Chi2) feature selection, '
                                                  'because the data-set contains too many categorical features. ',
    # Models
    # Classifiers
    'sklearn.naive_bayes.GaussianNB': 'GaussianNB',
    'xgboost.XGBClassifier': 'XGBClassifier',
    'sklearn.linear_model.SGDClassifier': 'SGDClassifier',
    'sklearn.naive_bayes.MultinomialNB': 'MultinomialNB',
    'sklearn.ensemble.AdaBoostClassifier': 'AdaBoostClassifier',
    'sklearn.naive_bayes.BernoulliNB': 'The Pipeline has BernoulliNB model which is a Naive Bayes classifier for '
                                       ' multivariate Bernoulli models. This model has been used because the user '
                                       'selected the "Speed" option and the machine learning problem is classification.'
    ,
    'sklearn.tree.DecisionTreeClassifier': 'The Pipeline has a DecisionTreeClassifier model. This model has been used '
                                           'because the user selected the "Interpretability" option and the machine '
                                           'learning problem is classification.',
    'sklearn.ensemble.ExtraTreesClassifier': 'ExtraTreesClassifier',
    'sklearn.ensemble.RandomForestClassifier': 'This Pipeline has a RandomForestClassifier model. This model has been '
                                               'used because the user selected the "Accuracy" option and the machine '
                                               'learning problem is classification.',
    'sklearn.ensemble.GradientBoostingClassifier': 'The Pipeline has a GradientBoostingClassifier model. This model has'
                                                   ' been used because the user selected the "Accuracy" option'
                                                   ' and the machine learning problem is classification.',
    'sklearn.neighbors.KNeighborsClassifier': 'The Pipeline has a KNeighborsClassifier model. This model has been used '
                                           'because the user selected the "Interpretability" option and the machine '
                                           'learning problem is classification.',
    'sklearn.svm.LinearSVC': 'This Pipeline has a LinearSVC model. This model has been used because the user selected '
                             'the "Speed" option and the machine learning problem is classification.',
    'sklearn.linear_model.LogisticRegression': 'The Pipeline has a LogisticRegression model. This model has been used '
                                           'because the user selected the "Interpretability" option and the machine '
                                           'learning problem is classification.',
    'sklearn.neural_network.MLPClassifier': 'This Pipeline has a MLPClassifier model. This model has'
                                                   ' been used because the user selected the "Accuracy" option'
                                                   ' and the machine learning problem is classification.',
    # Regressors
    'sklearn.linear_model.ElasticNetCV': 'ElasticNetCV',
    'sklearn.ensemble.ExtraTreesRegressor': 'ExtraTreesRegressor',
    'sklearn.ensemble.AdaBoostRegressor': 'AdaBoostRegressor',
    'sklearn.linear_model.LassoLarsCV': 'LassoLarsCV',
    'sklearn.linear_model.RidgeCV': 'RidgeCV',
    'xgboost.XGBRegressor': 'XGBRegressor',
    'sklearn.linear_model.SGDRegressor': 'SGDRegressor',
    'sklearn.ensemble.GradientBoostingRegressor': 'This Pipeline has a GradientBoostingRegressor model. This model has '
                                                  'been used because the user selected the "Accuracy" option and the '
                                                  'machine learning problem is regression.',
    'sklearn.tree.DecisionTreeRegressor': 'This Pipeline has a DecisionTreeRegressor model. This model has been '
                                           'used because the user selected the "Interpretability" option and the '
                                          'machine learning problem is regression.',
    'sklearn.neighbors.KNeighborsRegressor': 'This Pipeline has a KNeighborsRegressor model. This model has been '
                                           'used because the user selected the "Interpretability" option and the '
                                          'machine learning problem is regression.',
    'sklearn.svm.LinearSVR': 'This Pipeline has a LinearSVR model. This model has been used because the user selected '
                             'the "Speed" option and the machine learning problem is regression.',
    'sklearn.ensemble.RandomForestRegressor': 'This Pipeline has a RandomForestRegressor model. This model has been '
                                              'used because the user selected the "Accuracy" option and the machine '
                                              'learning problem is regression.',
    'sklearn.linear_model.LinearRegression': 'This Pipeline has a LinerRegression model. This model has been used '
                                             'because the user selected the "Interpretability" option and the machine '
                                             'learning problem is regression.',
    'sklearn.neural_network.MLPRegressor': 'This Pipeline has a MLPRegressor model. This model has been '
                                              'used because the user selected the "Accuracy" option and the machine '
                                              'learning problem is regression.',
    # Hyper-Parameter tuning model
    'RandomizedSearchCV': 'The Randomized Search hyper-parameter tuning was used in this Pipeline because the '
                          'parameters list was too large for Grid Search and the search for the best hyper-parameters'
                          'would take too long.',
    'GridSearchCV': 'The Grid Search hyper-parameter tuning was used in this Pipeline because the parameter list number'
                    ' was 9 or less, and an exhaustive Grid Search can be run. '
    # explainable model

}
