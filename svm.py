# This file is showing how model process data form raw to training,
# for modelling there is saperate file
# importing Required libraries
import logging
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import lightgbm as lgb
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# log file initialization 
logging.basicConfig(filename='debug.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')

logging.debug(' Model.py File execution started ')

# loading database with pandas library
df = pd.read_csv("./dataset/train.csv")
logging.debug(' Database Loaded ')

df = df.drop(['sl_no','salary'], axis=1)
df = df.apply(lambda x: x.fillna(0))
col_names = df.columns
category_col = ['ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation','status']

labelEncoder = preprocessing.LabelEncoder()

mapping_dict = {}
for col in category_col:
	df[col] = labelEncoder.fit_transform(df[col])

	le_name_mapping = dict(zip(labelEncoder.classes_,
							labelEncoder.transform(labelEncoder.classes_)))

	mapping_dict[col] = le_name_mapping

logging.debug('Database Pre-processing is Finished')

# model featuring
X = df[['gender',
 'ssc_p',
 'ssc_b',
 'hsc_p',
 'hsc_b',
 'hsc_s',
 'degree_p',
 'degree_t',
 'workex',
 'etest_p',
 'specialisation',
 'mba_p']]
y = df['status']

# Data Spliting For model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# summarize class distribution
print("Before oversampling: ",Counter(y_train))

# define oversampling strategy
SMOTE = SMOTE()

# fit and apply the transform
X_train_SMOTE, y_train_SMOTE = SMOTE.fit_resample(X_train, y_train)

# summarize class distribution
print("After oversampling: ",Counter(y_train_SMOTE))

# model fitting using LGBMClassifier
from sklearn.svm import SVC
svcclassifier = SVC(kernel='linear')
svcclassifier.fit(X_train, y_train)
y_pred = svcclassifier.predict(X_test)
print(y_pred)
#clf = lgb.svmClassifier()
#clf.fit(X_train_SMOTE, y_train_SMOTE)

# Printing Accuracy
predictions_e = svcclassifier.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, predictions_e))


# pkl export & finish log
pickle.dump(svcclassifier, open("model.pkl", "wb"))
logging.debug(' Execution of Model.py is finished ')


# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Model: XGBoost
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train_smote, y_train_smote)
xgb_predictions = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_predictions) * 100
print("XGBoost Accuracy: {:.2f}%".format(xgb_accuracy))
print("\nClassification Report:\n", classification_report(y_test, xgb_predictions))

# Plot confusion matrix
ConfusionMatrixDisplay.from_estimator(xgb_model, X_test, y_test, cmap="Blues", display_labels=["Not Placed", "Placed"])
plt.title("Confusion Matrix")
plt.show()

# Save the XGBoost model
pickle.dump(xgb_model, open("xgboost_model.pkl", "wb"))
print("XGBoost model saved as 'xgboost_model.pkl'.")




