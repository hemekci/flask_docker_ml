import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib



iris = datasets.load_iris() #Loading the dataset
iris.keys()


iris = pd.DataFrame(
    data= np.c_[iris['data'], iris['target']],
    columns= iris['feature_names'] + ['target']
    )

iris.head(10)

#eg: Setosa(0), Versicolor(1),  Virginica(2)


species = []

for i in range(len(iris['target'])):
    if iris['target'][i] == 0:
        species.append("setosa")
    elif iris['target'][i] == 1:
        species.append('versicolor')
    else:
        species.append('virginica')

iris['species'] = species

iris.groupby('species').size()
iris.describe()

classifiers = [('LR', LogisticRegression()),
               ('KNN', KNeighborsClassifier()),
               # ("SVC", SVC()),
               ("CART", DecisionTreeClassifier()),
               ("RF", RandomForestClassifier()),
               ('Adaboost', AdaBoostClassifier()),
               ('GBM', GradientBoostingClassifier()),
               #('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')),
               #('LightGBM', LGBMClassifier()),
               # ('CatBoost', CatBoostClassifier(verbose=False))
               ]

y = iris["species"]
X = iris.drop(["target", "species"], axis=1)

# Label encoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Save Encoder file
joblib.dump(encoder, "saved_models/0_iris_label_encoder.pkl")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.7, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

LR = LogisticRegression()
scores = cross_validate(LR, X_train, y_train, cv=5, scoring=["roc_auc_ovr"])


# Evaluate the model using cross_validate()  ROC_AUC_OVR used for multiple classes
for name, classifier in classifiers:
    cv_results = cross_validate(classifier, X_train, y_train, cv=3, scoring=["roc_auc_ovr"])
    print(f"AUC_OVR: {round(cv_results['test_roc_auc_ovr'].mean(), 4)} ({name}) ")


# Save Model

index = 1
for name, classifier in classifiers:
    classifier.fit(X_train, y_train)
    joblib.dump(classifier, f"saved_models/{index}_{name}.pkl")
    index = index+1

# make predictions
# Read models
encoder_loaded = joblib.load("saved_models/0_iris_label_encoder.pkl")
classifier_loaded = joblib.load("saved_models/1_LR.pkl")

# Prediction set
X_manual_test = [[4.0, 4.0, 4.0, 4.0]]
print("X_manual_test", X_manual_test)


prediction_raw = classifier_loaded.predict(X_manual_test)
print("prediction_raw", prediction_raw)

prediction_real = encoder_loaded.inverse_transform(classifier_loaded.predict(X_manual_test))
print("Real prediction", prediction_real)

