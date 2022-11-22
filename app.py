import numpy as np
import pandas as pd

import sklearn
import sklearn.metrics
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, precision_score, recall_score, f1_score, classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve

from scipy import stats as st
from random import randrange
from matplotlib import pyplot as plt
from scipy.special import softmax

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from catboost import Pool
from sklearn.ensemble import RandomForestClassifier


import optuna

import shap

import gradio as gr

import random

#Read and redefine data.

data = pd.read_csv("acdf_imputed.csv", index_col = 0)
variables = ['SEX', 'INOUT', 'TRANST', 'AGE', 'SURGSPEC', 'HEIGHT', 'WEIGHT', 'DIABETES', 'SMOKE', 'DYSPNEA', 'FNSTATUS2', 'VENTILAT', 'HXCOPD', 'ASCITES', 'HXCHF', 'HYPERMED', 'RENAFAIL', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDDIS', 'TRANSFUS', 'PRSODM', 'PRBUN', 'PRCREAT', 'PRWBC', 'PRHCT', 'PRPLATE', 'ASACLAS', 'READMISSION1', 'BMI', 'RACE', 'LEVELS', 'LOS', 'DISCHARGE']
data = data[variables]

data['SEX'] = data['SEX'].replace(['male'], 'Male')
data['SEX'] = data['SEX'].replace(['female'], 'Female')
data['SEX'] = data['SEX'].replace(['non-binary'], 'Non-Binary')

print(data.columns)

#Define outcomes.

x = data
y1 = data.pop('LOS')
y2 = data.pop('DISCHARGE')
y3 = data.pop('READMISSION1')
y1 = (y1 == "Yes").astype(int)
y2 = (y2 == "Yes").astype(int)
y3 = (y3 == "Yes").astype(int)

categorical_columns = list(x.select_dtypes('object').columns)

x = x.astype({col: "category" for col in categorical_columns})

#Prepare data for LOS (y1).
y1_data_xgb = xgb.DMatrix(x, label=y1, enable_categorical=True)
y1_data_lgb = lgb.Dataset(x, label=y1)
y1_data_cb = Pool(data=x, label=y1, cat_features=categorical_columns)

#Prepare data for DISCHARGE (y2).
y2_data_xgb = xgb.DMatrix(x, label=y2, enable_categorical=True)
y2_data_lgb = lgb.Dataset(x, label=y2)
y2_data_cb = Pool(data=x, label=y2, cat_features=categorical_columns)

#Prepare data for READMISSION (y3).
y3_data_xgb = xgb.DMatrix(x, label=y3, enable_categorical=True)
y3_data_lgb = lgb.Dataset(x, label=y3)
y3_data_cb = Pool(data=x, label=y3, cat_features=categorical_columns)

#Prepare data for Random Forest models.
x_rf = x
categorical_columns = list(x_rf.select_dtypes('category').columns)
x_rf = x_rf.astype({col: "category" for col in categorical_columns})
le = sklearn.preprocessing.LabelEncoder()
for col in categorical_columns:
        x_rf[col] = le.fit_transform(x_rf[col].astype(str))
d1 = dict.fromkeys(x_rf.select_dtypes(np.int64).columns, str)
x_rf = x_rf.astype(d1)

#Assign unique values as answer options.

unique_sex = list(data["SEX"].unique())
unique_inout = list(data["INOUT"].unique())
unique_transt = list(data["TRANST"].unique())
unique_surgspec = list(data["SURGSPEC"].unique())
unique_diabetes = list(data["DIABETES"].unique())
unique_smoke = list(data["SMOKE"].unique())
unique_dyspnea = list(data["DYSPNEA"].unique())
unique_fnstatus2 = list(data["FNSTATUS2"].unique())
unique_ventilat = list(data["VENTILAT"].unique())
unique_hxcopd = list(data["HXCOPD"].unique())
unique_ascites = list(data["ASCITES"].unique())
unique_hxchf = list(data["HXCHF"].unique())
unique_hypermed = list(data["HYPERMED"].unique())
unique_renafail = list(data["RENAFAIL"].unique())
unique_dialysis = list(data["DIALYSIS"].unique())
unique_discancr = list(data["DISCANCR"].unique())
unique_wndinf = list(data["WNDINF"].unique())
unique_steroid = list(data["STEROID"].unique())
unique_wtloss = list(data["WTLOSS"].unique())
unique_bleeddis = list(data["BLEEDDIS"].unique())
unique_transfus = list(data["TRANSFUS"].unique())
unique_asaclas = list(data["ASACLAS"].unique())
unique_race = list(data["RACE"].unique())
unique_levels = list(data["LEVELS"].unique())


#Assign hyperparameters.

y1_xgb_params = {'objective': 'binary:logistic', 'booster': 'gbtree', 'lambda': 2.130200881607855e-06, 'alpha': 1.1952639351742128e-08, 'max_depth': 1, 'eta': 0.7368548077735989, 'gamma': 2.792491017336799e-08, 'grow_policy': 'depthwise'}
y2_xgb_params = {'objective': 'binary:logistic', 'booster': 'gbtree', 'lambda': 2.130200881607855e-06, 'alpha': 1.1952639351742128e-08, 'max_depth': 1, 'eta': 0.7368548077735989, 'gamma': 2.792491017336799e-08, 'grow_policy': 'depthwise'}
y3_xgb_params = {'objective': 'binary:logistic', 'booster': 'gbtree', 'lambda': 2.130200881607855e-06, 'alpha': 1.1952639351742128e-08, 'max_depth': 1, 'eta': 0.7368548077735989, 'gamma': 2.792491017336799e-08, 'grow_policy': 'depthwise'}

y1_lgb_params = {'objective': 'binary', 'boosting_type': 'gbdt', 'lambda_l1': 2.4979101276990208, 'lambda_l2': 3.039019408265432e-08, 'num_leaves': 2, 'feature_fraction': 0.49911918163841645, 'bagging_fraction': 0.6263675478864141, 'bagging_freq': 7, 'min_child_samples': 94}
y2_lgb_params = {'objective': 'binary', 'boosting_type': 'gbdt', 'lambda_l1': 9.733358026911517e-06, 'lambda_l2': 0.07114217378879342, 'num_leaves': 2, 'feature_fraction': 0.8005516906569979, 'bagging_fraction': 0.796375315175025, 'bagging_freq': 7, 'min_child_samples': 34}
y3_lgb_params = {'objective': 'binary', 'boosting_type': 'gbdt', 'lambda_l1': 1.0493476627183768e-05, 'lambda_l2': 0.00011086160340957947, 'num_leaves': 2, 'feature_fraction': 0.5039171555750599, 'bagging_fraction': 0.4619254427343027, 'bagging_freq': 2, 'min_child_samples': 99}

y1_cb_params = {'objective': 'CrossEntropy', 'colsample_bylevel': 0.05888279602035283, 'depth': 5, 'boosting_type': 'Plain', 'bootstrap_type': 'Bernoulli', 'subsample': 0.14630926431770105}
y2_cb_params = {'objective': 'CrossEntropy', 'colsample_bylevel': 0.04799694910292889, 'depth': 9, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}
y3_cb_params = {'objective': 'Logloss', 'colsample_bylevel': 0.07294555922953751, 'depth': 6, 'boosting_type': 'Plain', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 8.377424880761083}

y1_rf_params = {'criterion': 'gini', 'bootstrap': 'auto', 'max_features': 'log2', 'max_depth': 35, 'n_estimators': 500, 'min_samples_leaf': 3, 'min_samples_split': 5}
y2_rf_params = {'criterion': 'gini', 'bootstrap': 'auto', 'max_features': 'log2', 'max_depth': 70, 'n_estimators': 1500, 'min_samples_leaf': 1, 'min_samples_split': 9}
y3_rf_params = {'criterion': 'entropy', 'bootstrap': 'auto', 'max_features': 'sqrt', 'max_depth': 67, 'n_estimators': 1200, 'min_samples_leaf': 4, 'min_samples_split': 7}


#Modeling for y1/LOS.

y1_model_xgb = xgb.train(params=y1_xgb_params, dtrain=y1_data_xgb)
y1_explainer_xgb = shap.TreeExplainer(y1_model_xgb)

y1_model_lgb = lgb.train(params=y1_lgb_params, train_set=y1_data_lgb)
y1_explainer_lgb = shap.TreeExplainer(y1_model_lgb)

y1_model_cb = cb.train(pool=y1_data_cb, params=y1_cb_params)
y1_explainer_cb = shap.TreeExplainer(y1_model_cb)

from sklearn.ensemble import RandomForestClassifier as rf
y1_rf = rf(**y1_rf_params)
y1_model_rf = y1_rf.fit(x_rf, y1)
y1_explainer_rf = shap.TreeExplainer(y1_model_rf)


#Modeling for y2/COMP.

y2_model_xgb = xgb.train(params=y2_xgb_params, dtrain=y2_data_xgb)
y2_explainer_xgb = shap.TreeExplainer(y2_model_xgb)

y2_model_lgb = lgb.train(params=y2_lgb_params, train_set=y2_data_lgb)
y2_explainer_lgb = shap.TreeExplainer(y2_model_lgb)

y2_model_cb = cb.train(pool=y2_data_cb, params=y2_cb_params)
y2_explainer_cb = shap.TreeExplainer(y2_model_cb)

from sklearn.ensemble import RandomForestClassifier as rf
y2_rf = rf(**y2_rf_params)
y2_model_rf = y2_rf.fit(x_rf, y2)
y2_explainer_rf = shap.TreeExplainer(y2_model_rf)


#Modeling for y3/DISCHARGE.

y3_model_xgb = xgb.train(params=y3_xgb_params, dtrain=y3_data_xgb)
y3_explainer_xgb = shap.TreeExplainer(y3_model_xgb)

y3_model_lgb = lgb.train(params=y3_lgb_params, train_set=y3_data_lgb)
y3_explainer_lgb = shap.TreeExplainer(y3_model_lgb)

y3_model_cb = cb.train(pool=y3_data_cb, params=y3_cb_params)
y3_explainer_cb = shap.TreeExplainer(y3_model_cb)

from sklearn.ensemble import RandomForestClassifier as rf
y3_rf = rf(**y3_rf_params)
y3_model_rf = y3_rf.fit(x_rf, y3)
y3_explainer_rf = shap.TreeExplainer(y3_model_rf)


#Define predict for y1/LOS.

def y1_predict_xgb(*args):
    df_xgb = pd.DataFrame([args], columns=x.columns)
    df_xgb = df_xgb.astype({col: "category" for col in categorical_columns})
    pos_pred = y1_model_xgb.predict(xgb.DMatrix(df_xgb, enable_categorical=True))
    return {"Prolonged LOS": float(pos_pred[0]), "Not Prolonged LOS": 1 - float(pos_pred[0])}

def y1_predict_lgb(*args):
    df = pd.DataFrame([args], columns=data.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    pos_pred = y1_model_lgb.predict(df)
    return {"Prolonged LOS": float(pos_pred[0]), "Not Prolonged LOS": 1 - float(pos_pred[0])}

def y1_predict_cb(*args):
    df_cb = pd.DataFrame([args], columns=x.columns)
    df_cb = df_cb.astype({col: "category" for col in categorical_columns})
    pos_pred = y1_model_cb.predict(Pool(df_cb, cat_features = categorical_columns), prediction_type='Probability')
    return {"Prolonged LOS": float(pos_pred[0][1]), "Not Prolonged LOS": float(pos_pred[0][0])}

def y1_predict_rf(*args):
    df = pd.DataFrame([args], columns=x_rf.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    d = dict.fromkeys(df.select_dtypes(np.int64).columns, np.int32)
    df = df.astype(d)
    pos_pred = y1_model_rf.predict_proba(df)
    return {"Prolonged LOS": float(pos_pred[0][1]), "Not Prolonged LOS": float(pos_pred[0][0])}


#Define predict for y2/DISCHARGE.

def y2_predict_xgb(*args):
    df_xgb = pd.DataFrame([args], columns=x.columns)
    df_xgb = df_xgb.astype({col: "category" for col in categorical_columns})
    pos_pred = y2_model_xgb.predict(xgb.DMatrix(df_xgb, enable_categorical=True))
    return {"Non-home Discharge": float(pos_pred[0]), "Home Discharge": 1 - float(pos_pred[0])}

def y2_predict_lgb(*args):
    df = pd.DataFrame([args], columns=data.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    pos_pred = y2_model_lgb.predict(df)
    return {"Non-home Discharge": float(pos_pred[0]), "Home Discharge": 1 - float(pos_pred[0])}

def y2_predict_cb(*args):
    df_cb = pd.DataFrame([args], columns=x.columns)
    df_cb = df_cb.astype({col: "category" for col in categorical_columns})
    pos_pred = y2_model_cb.predict(Pool(df_cb, cat_features = categorical_columns), prediction_type='Probability')
    return {"Non-home Discharge": float(pos_pred[0][1]), "Home Discharge": float(pos_pred[0][0])}

def y2_predict_rf(*args):
    df = pd.DataFrame([args], columns=x_rf.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    d = dict.fromkeys(df.select_dtypes(np.int64).columns, np.int32)
    df = df.astype(d)
    pos_pred = y2_model_rf.predict_proba(df)
    return {"Non-home Discharge": float(pos_pred[0][1]), "Home Discharge": float(pos_pred[0][0])}


#Define predict for y3/READMISSION.

def y3_predict_xgb(*args):
    df_xgb = pd.DataFrame([args], columns=x.columns)
    df_xgb = df_xgb.astype({col: "category" for col in categorical_columns})
    pos_pred = y3_model_xgb.predict(xgb.DMatrix(df_xgb, enable_categorical=True))
    return {"No Readmission": float(pos_pred[0]), "Readmission": 1 - float(pos_pred[0])}

def y3_predict_lgb(*args):
    df = pd.DataFrame([args], columns=data.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    pos_pred = y3_model_lgb.predict(df)
    return {"No Readmission": float(pos_pred[0]), "Readmission": 1 - float(pos_pred[0])}

def y3_predict_cb(*args):
    df_cb = pd.DataFrame([args], columns=x.columns)
    df_cb = df_cb.astype({col: "category" for col in categorical_columns})
    pos_pred = y3_model_cb.predict(Pool(df_cb, cat_features = categorical_columns), prediction_type='Probability')
    return {"No Readmission": float(pos_pred[0][1]), "Readmission": float(pos_pred[0][0])}

def y3_predict_rf(*args):
    df = pd.DataFrame([args], columns=x_rf.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    d = dict.fromkeys(df.select_dtypes(np.int64).columns, np.int32)
    df = df.astype(d)
    pos_pred = y3_model_rf.predict_proba(df)


#Define interpret for y1/LOS.

def y1_interpret_xgb(*args):
    df = pd.DataFrame([args], columns=x.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y1_explainer_xgb.shap_values(xgb.DMatrix(df, enable_categorical=True))
    scores_desc = list(zip(shap_values[0], x.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m

def y1_interpret_lgb(*args):
    df = pd.DataFrame([args], columns=x.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y1_explainer_lgb.shap_values(df)
    scores_desc = list(zip(shap_values[0][0], x.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m

def y1_interpret_cb(*args):
    df = pd.DataFrame([args], columns=x.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y1_explainer_cb.shap_values(Pool(df, cat_features = categorical_columns))
    scores_desc = list(zip(shap_values[0], x.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m

def y1_interpret_rf(*args):
    df = pd.DataFrame([args], columns=x_rf.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y1_explainer_rf.shap_values(df)
    scores_desc = list(zip(shap_values[0][0], x_rf.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m


#Define interpret for y2/DISCHARGE.

def y2_interpret_xgb(*args):
    df = pd.DataFrame([args], columns=x.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y2_explainer_xgb.shap_values(xgb.DMatrix(df, enable_categorical=True))
    scores_desc = list(zip(shap_values[0], x.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m

def y2_interpret_lgb(*args):
    df = pd.DataFrame([args], columns=x.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y2_explainer_lgb.shap_values(df)
    scores_desc = list(zip(shap_values[0][0], x.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m

def y2_interpret_cb(*args):
    df = pd.DataFrame([args], columns=x.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y2_explainer_cb.shap_values(Pool(df, cat_features = categorical_columns))
    scores_desc = list(zip(shap_values[0], x.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m

def y2_interpret_rf(*args):
    df = pd.DataFrame([args], columns=x_rf.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y2_explainer_rf.shap_values(df)
    scores_desc = list(zip(shap_values[0][0], x_rf.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m


#Define interpret for y3/READMISSION.

def y3_interpret_xgb(*args):
    df = pd.DataFrame([args], columns=x.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y3_explainer_xgb.shap_values(xgb.DMatrix(df, enable_categorical=True))
    scores_desc = list(zip(shap_values[0], x.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m

def y3_interpret_lgb(*args):
    df = pd.DataFrame([args], columns=x.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y3_explainer_lgb.shap_values(df)
    scores_desc = list(zip(shap_values[0][0], x.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m

def y3_interpret_cb(*args):
    df = pd.DataFrame([args], columns=x.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y3_explainer_cb.shap_values(Pool(df, cat_features = categorical_columns))
    scores_desc = list(zip(shap_values[0], x.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m

def y3_interpret_rf(*args):
    df = pd.DataFrame([args], columns=x_rf.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y3_explainer_rf.shap_values(df)
    scores_desc = list(zip(shap_values[0][0], x_rf.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m

with gr.Blocks() as demo:
    
    gr.Markdown(
        """ 
    """
    )
        
    gr.Markdown(
        """
    # Outcome Prediction for ACDF Surgery
    """
    )

    with gr.Tab('Length of Stay'):
        
        gr.Markdown(
            """
         
        ### Prolonged Length of Stay Prediction Model for ACDF Surgery
        """
        )

    
        with gr.Row():

            with gr.Column():

                AGE = gr.Slider(label="Age", minimum=17, maximum=99, step=1, randomize=True)

                SEX = gr.Radio(
                    label="Sex",
                    choices=unique_sex,
                    type='index',
                    value=lambda: random.choice(unique_sex),
                )

                RACE = gr.Radio(
                    label="Race",
                    choices=unique_race,
                    type='index',
                    value=lambda: random.choice(unique_race),
                )

                HEIGHT = gr.Slider(label="Height (in meters)", minimum=1.0, maximum=2.25, step=0.01, randomize=True)

                WEIGHT = gr.Slider(label="Weight (in kilograms)", minimum=20, maximum=200, step=1, randomize=True)

                BMI = gr.Slider(label="BMI", minimum=10, maximum=70, step=1, randomize=True)

                TRANST = gr.Radio(
                    label="Transfer Status",
                    choices=unique_transt,
                    type='index',
                    value=lambda: random.choice(unique_transt),
                )
                
                INOUT = gr.Radio(
                    label="Inpatient or Outpatient",
                    choices=unique_inout,
                    type='index',
                    value=lambda: random.choice(unique_inout),
                )

                SURGSPEC = gr.Radio(
                    label="Surgical Specialty",
                    choices=unique_surgspec,
                    type='index',
                    value=lambda: random.choice(unique_surgspec),
                )

                SMOKE = gr.Radio(
                    label="Smoking Status",
                    choices=unique_smoke,
                    type='index',
                    value=lambda: random.choice(unique_smoke),
                )

                DIABETES = gr.Radio(
                    label="Diabetes",
                    choices=unique_diabetes,
                    type='index',
                    value=lambda: random.choice(unique_diabetes),
                )

                DYSPNEA = gr.Radio(
                    label="Dyspnea",
                    choices=unique_dyspnea,
                    type='index',
                    value=lambda: random.choice(unique_dyspnea),
                )
                
                VENTILAT = gr.Radio(
                    label="Ventilator Dependency",
                    choices=unique_ventilat,
                    type='index',
                    value=lambda: random.choice(unique_ventilat),
                )

                HXCOPD = gr.Radio(
                    label="History of COPD",
                    choices=unique_hxcopd,
                    type='index',
                    value=lambda: random.choice(unique_hxcopd),
                )

                ASCITES = gr.Radio(
                    label="Ascites",
                    choices=unique_ascites,
                    type='index',
                    value=lambda: random.choice(unique_ascites),
                )

                HXCHF = gr.Radio(
                    label="History of Congestive Heart Failure",
                    choices=unique_hxchf,
                    type='index',
                    value=lambda: random.choice(unique_hxchf),
                )

                HYPERMED = gr.Radio(
                    label="Hypertension Despite Medication",
                    choices=unique_hypermed,
                    type='index',
                    value=lambda: random.choice(unique_hypermed),
                )

                RENAFAIL = gr.Radio(
                    label="Renal Failure",
                    choices=unique_renafail,
                    type='index',
                    value=lambda: random.choice(unique_renafail),
                )

                DIALYSIS = gr.Radio(
                    label="Dialysis",
                    choices=unique_dialysis,
                    type='index',
                    value=lambda: random.choice(unique_dialysis),
                )

                STEROID = gr.Radio(
                    label="Steroid",
                    choices=unique_steroid,
                    type='index',
                    value=lambda: random.choice(unique_steroid),
                )

                WTLOSS = gr.Radio(
                    label="Weight Loss",
                    choices=unique_wtloss,
                    type='index',
                    value=lambda: random.choice(unique_wtloss),
                )

                BLEEDDIS = gr.Radio(
                    label="Bleeding Disorder",
                    choices=unique_bleeddis,
                    type='index',
                    value=lambda: random.choice(unique_bleeddis),
                )

                TRANSFUS = gr.Radio(
                    label="Transfusion",
                    choices=unique_transfus,
                    type='index',
                    value=lambda: random.choice(unique_transfus),
                )
                
                WNDINF = gr.Radio(
                    label="Wound Infection",
                    choices=unique_wndinf,
                    type='index',
                    value=lambda: random.choice(unique_wndinf),
                )

                DISCANCR = gr.Radio(
                    label="Disseminated Cancer",
                    choices=unique_discancr,
                    type='index',
                    value=lambda: random.choice(unique_discancr),
                )

                FNSTATUS2 = gr.Radio(
                    label="Functional Status",
                    choices=unique_fnstatus2,
                    type='index',
                    value=lambda: random.choice(unique_fnstatus2),
                )

                PRSODM = gr.Slider(label="Sodium", minimum=min(x['PRSODM']), maximum=max(x['PRSODM']), step=1, randomize=True)

                PRBUN = gr.Slider(label="BUN", minimum=min(x['PRBUN']), maximum=max(x['PRBUN']), step=1, randomize=True)

                PRCREAT = gr.Slider(label="Creatine", minimum=min(x['PRCREAT']),maximum=max(x['PRCREAT']), step=0.1, randomize=True)

                PRWBC = gr.Slider(label="WBC", minimum=min(x['PRWBC']), maximum=max(x['PRWBC']), step=0.1, randomize=True)

                PRHCT = gr.Slider(label="Hematocrit", minimum=min(x['PRHCT']), maximum=max(x['PRHCT']), step=0.1, randomize=True)

                PRPLATE = gr.Slider(label="Platelet", minimum=min(x['PRPLATE']), maximum=max(x['PRPLATE']), step=1, randomize=True)

                ASACLAS = gr.Radio(
                    label="ASA Class",
                    choices=unique_asaclas,
                    type='index',
                    value=lambda: random.choice(unique_asaclas),

                )

                LEVELS = gr.Radio(
                    label="Levels",
                    choices=unique_levels,
                    type='index',
                    value=lambda: random.choice(unique_levels),
                )

            with gr.Column():

                with gr.Row():
                    y1_predict_btn_xgb = gr.Button(value="Predict (XGBoost)")
                    y1_predict_btn_lgb = gr.Button(value="Predict (LightGBM)")
                    y1_predict_btn_cb = gr.Button(value="Predict (CatBoost)")
                    y1_predict_btn_rf = gr.Button(value="Predict (Random Forest)")
                label = gr.Label()

                with gr.Row():
                    y1_interpret_btn_xgb = gr.Button(value="Explain (XGBoost)")
                    y1_interpret_btn_lgb = gr.Button(value="Explain (LightGBM)")
                    y1_interpret_btn_cb = gr.Button(value="Explain (CatBoost)")
                    y1_interpret_btn_rf = gr.Button(value="Explain (Random Forest)") 

                plot = gr.Plot()

                y1_predict_btn_xgb.click(
                    y1_predict_xgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y1_predict_btn_lgb.click(
                    y1_predict_lgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y1_predict_btn_cb.click(
                    y1_predict_cb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y1_predict_btn_rf.click(
                    y1_predict_rf,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y1_interpret_btn_xgb.click(
                    y1_interpret_xgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )

                y1_interpret_btn_lgb.click(
                    y1_interpret_lgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )

                y1_interpret_btn_cb.click(
                    y1_interpret_cb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )

                y1_interpret_btn_rf.click(
                    y1_interpret_rf,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )
                
    with gr.Tab('Non-home Discharge'):
        
        gr.Markdown(
            """
         
        ### Non-home Discharge Prediction Model for ACDF Surgery
        """
        )

    
        with gr.Row():

            with gr.Column():

                AGE = gr.Slider(label="Age", minimum=17, maximum=99, step=1, randomize=True)

                SEX = gr.Radio(
                    label="Sex",
                    choices=unique_sex,
                    type='index',
                    value=lambda: random.choice(unique_sex),
                )

                RACE = gr.Radio(
                    label="Race",
                    choices=unique_race,
                    type='index',
                    value=lambda: random.choice(unique_race),
                )

                HEIGHT = gr.Slider(label="Height (in meters)", minimum=1.0, maximum=2.25, step=0.01, randomize=True)

                WEIGHT = gr.Slider(label="Weight (in kilograms)", minimum=20, maximum=200, step=1, randomize=True)

                BMI = gr.Slider(label="BMI", minimum=10, maximum=70, step=1, randomize=True)

                TRANST = gr.Radio(
                    label="Transfer Status",
                    choices=unique_transt,
                    type='index',
                    value=lambda: random.choice(unique_transt),
                )
                
                INOUT = gr.Radio(
                    label="Inpatient or Outpatient",
                    choices=unique_inout,
                    type='index',
                    value=lambda: random.choice(unique_inout),
                )

                SURGSPEC = gr.Radio(
                    label="Surgical Specialty",
                    choices=unique_surgspec,
                    type='index',
                    value=lambda: random.choice(unique_surgspec),
                )

                SMOKE = gr.Radio(
                    label="Smoking Status",
                    choices=unique_smoke,
                    type='index',
                    value=lambda: random.choice(unique_smoke),
                )

                DIABETES = gr.Radio(
                    label="Diabetes",
                    choices=unique_diabetes,
                    type='index',
                    value=lambda: random.choice(unique_diabetes),
                )

                DYSPNEA = gr.Radio(
                    label="Dyspnea",
                    choices=unique_dyspnea,
                    type='index',
                    value=lambda: random.choice(unique_dyspnea),
                )
                
                VENTILAT = gr.Radio(
                    label="Ventilator Dependency",
                    choices=unique_ventilat,
                    type='index',
                    value=lambda: random.choice(unique_ventilat),
                )

                HXCOPD = gr.Radio(
                    label="History of COPD",
                    choices=unique_hxcopd,
                    type='index',
                    value=lambda: random.choice(unique_hxcopd),
                )

                ASCITES = gr.Radio(
                    label="Ascites",
                    choices=unique_ascites,
                    type='index',
                    value=lambda: random.choice(unique_ascites),
                )

                HXCHF = gr.Radio(
                    label="History of Congestive Heart Failure",
                    choices=unique_hxchf,
                    type='index',
                    value=lambda: random.choice(unique_hxchf),
                )

                HYPERMED = gr.Radio(
                    label="Hypertension Despite Medication",
                    choices=unique_hypermed,
                    type='index',
                    value=lambda: random.choice(unique_hypermed),
                )

                RENAFAIL = gr.Radio(
                    label="Renal Failure",
                    choices=unique_renafail,
                    type='index',
                    value=lambda: random.choice(unique_renafail),
                )

                DIALYSIS = gr.Radio(
                    label="Dialysis",
                    choices=unique_dialysis,
                    type='index',
                    value=lambda: random.choice(unique_dialysis),
                )

                STEROID = gr.Radio(
                    label="Steroid",
                    choices=unique_steroid,
                    type='index',
                    value=lambda: random.choice(unique_steroid),
                )

                WTLOSS = gr.Radio(
                    label="Weight Loss",
                    choices=unique_wtloss,
                    type='index',
                    value=lambda: random.choice(unique_wtloss),
                )

                BLEEDDIS = gr.Radio(
                    label="Bleeding Disorder",
                    choices=unique_bleeddis,
                    type='index',
                    value=lambda: random.choice(unique_bleeddis),
                )

                TRANSFUS = gr.Radio(
                    label="Transfusion",
                    choices=unique_transfus,
                    type='index',
                    value=lambda: random.choice(unique_transfus),
                )
                
                WNDINF = gr.Radio(
                    label="Wound Infection",
                    choices=unique_wndinf,
                    type='index',
                    value=lambda: random.choice(unique_wndinf),
                )

                DISCANCR = gr.Radio(
                    label="Disseminated Cancer",
                    choices=unique_discancr,
                    type='index',
                    value=lambda: random.choice(unique_discancr),
                )

                FNSTATUS2 = gr.Radio(
                    label="Functional Status",
                    choices=unique_fnstatus2,
                    type='index',
                    value=lambda: random.choice(unique_fnstatus2),
                )

                PRSODM = gr.Slider(label="Sodium", minimum=min(x['PRSODM']), maximum=max(x['PRSODM']), step=1, randomize=True)

                PRBUN = gr.Slider(label="BUN", minimum=min(x['PRBUN']), maximum=max(x['PRBUN']), step=1, randomize=True)

                PRCREAT = gr.Slider(label="Creatine", minimum=min(x['PRCREAT']),maximum=max(x['PRCREAT']), step=0.1, randomize=True)

                PRWBC = gr.Slider(label="WBC", minimum=min(x['PRWBC']), maximum=max(x['PRWBC']), step=0.1, randomize=True)

                PRHCT = gr.Slider(label="Hematocrit", minimum=min(x['PRHCT']), maximum=max(x['PRHCT']), step=0.1, randomize=True)

                PRPLATE = gr.Slider(label="Platelet", minimum=min(x['PRPLATE']), maximum=max(x['PRPLATE']), step=1, randomize=True)

                ASACLAS = gr.Radio(
                    label="ASA Class",
                    choices=unique_asaclas,
                    type='index',
                    value=lambda: random.choice(unique_asaclas),

                )

                LEVELS = gr.Radio(
                    label="Levels",
                    choices=unique_levels,
                    type='index',
                    value=lambda: random.choice(unique_levels),
                )

            with gr.Column():

                with gr.Row():
                    y2_predict_btn_xgb = gr.Button(value="Predict (XGBoost)")
                    y2_predict_btn_lgb = gr.Button(value="Predict (LightGBM)")
                    y2_predict_btn_cb = gr.Button(value="Predict (CatBoost)")
                    y2_predict_btn_rf = gr.Button(value="Predict (Random Forest)")
                label = gr.Label()

                with gr.Row():
                    y2_interpret_btn_xgb = gr.Button(value="Explain (XGBoost)")
                    y2_interpret_btn_lgb = gr.Button(value="Explain (LightGBM)")
                    y2_interpret_btn_cb = gr.Button(value="Explain (CatBoost)")
                    y2_interpret_btn_rf = gr.Button(value="Explain (Random Forest)") 

                plot = gr.Plot()

                y2_predict_btn_xgb.click(
                    y2_predict_xgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y2_predict_btn_lgb.click(
                    y2_predict_lgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y2_predict_btn_cb.click(
                    y2_predict_cb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y2_predict_btn_rf.click(
                    y2_predict_rf,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y2_interpret_btn_xgb.click(
                    y2_interpret_xgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )

                y2_interpret_btn_lgb.click(
                    y2_interpret_lgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )

                y2_interpret_btn_cb.click(
                    y2_interpret_cb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )

                y2_interpret_btn_rf.click(
                    y2_interpret_rf,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )
                
    with gr.Tab('Readmission'):
        
        gr.Markdown(
            """
         
        ### Readmission Prediction Model for ACDF Surgery
        """
        )

    
        with gr.Row():

            with gr.Column():

                AGE = gr.Slider(label="Age", minimum=17, maximum=99, step=1, randomize=True)

                SEX = gr.Radio(
                    label="Sex",
                    choices=unique_sex,
                    type='index',
                    value=lambda: random.choice(unique_sex),
                )

                RACE = gr.Radio(
                    label="Race",
                    choices=unique_race,
                    type='index',
                    value=lambda: random.choice(unique_race),
                )

                HEIGHT = gr.Slider(label="Height (in meters)", minimum=1.0, maximum=2.25, step=0.01, randomize=True)

                WEIGHT = gr.Slider(label="Weight (in kilograms)", minimum=20, maximum=200, step=1, randomize=True)

                BMI = gr.Slider(label="BMI", minimum=10, maximum=70, step=1, randomize=True)

                TRANST = gr.Radio(
                    label="Transfer Status",
                    choices=unique_transt,
                    type='index',
                    value=lambda: random.choice(unique_transt),
                )
                
                INOUT = gr.Radio(
                    label="Inpatient or Outpatient",
                    choices=unique_inout,
                    type='index',
                    value=lambda: random.choice(unique_inout),
                )

                SURGSPEC = gr.Radio(
                    label="Surgical Specialty",
                    choices=unique_surgspec,
                    type='index',
                    value=lambda: random.choice(unique_surgspec),
                )

                SMOKE = gr.Radio(
                    label="Smoking Status",
                    choices=unique_smoke,
                    type='index',
                    value=lambda: random.choice(unique_smoke),
                )

                DIABETES = gr.Radio(
                    label="Diabetes",
                    choices=unique_diabetes,
                    type='index',
                    value=lambda: random.choice(unique_diabetes),
                )

                DYSPNEA = gr.Radio(
                    label="Dyspnea",
                    choices=unique_dyspnea,
                    type='index',
                    value=lambda: random.choice(unique_dyspnea),
                )
                
                VENTILAT = gr.Radio(
                    label="Ventilator Dependency",
                    choices=unique_ventilat,
                    type='index',
                    value=lambda: random.choice(unique_ventilat),
                )

                HXCOPD = gr.Radio(
                    label="History of COPD",
                    choices=unique_hxcopd,
                    type='index',
                    value=lambda: random.choice(unique_hxcopd),
                )

                ASCITES = gr.Radio(
                    label="Ascites",
                    choices=unique_ascites,
                    type='index',
                    value=lambda: random.choice(unique_ascites),
                )

                HXCHF = gr.Radio(
                    label="History of Congestive Heart Failure",
                    choices=unique_hxchf,
                    type='index',
                    value=lambda: random.choice(unique_hxchf),
                )

                HYPERMED = gr.Radio(
                    label="Hypertension Despite Medication",
                    choices=unique_hypermed,
                    type='index',
                    value=lambda: random.choice(unique_hypermed),
                )

                RENAFAIL = gr.Radio(
                    label="Renal Failure",
                    choices=unique_renafail,
                    type='index',
                    value=lambda: random.choice(unique_renafail),
                )

                DIALYSIS = gr.Radio(
                    label="Dialysis",
                    choices=unique_dialysis,
                    type='index',
                    value=lambda: random.choice(unique_dialysis),
                )

                STEROID = gr.Radio(
                    label="Steroid",
                    choices=unique_steroid,
                    type='index',
                    value=lambda: random.choice(unique_steroid),
                )

                WTLOSS = gr.Radio(
                    label="Weight Loss",
                    choices=unique_wtloss,
                    type='index',
                    value=lambda: random.choice(unique_wtloss),
                )

                BLEEDDIS = gr.Radio(
                    label="Bleeding Disorder",
                    choices=unique_bleeddis,
                    type='index',
                    value=lambda: random.choice(unique_bleeddis),
                )

                TRANSFUS = gr.Radio(
                    label="Transfusion",
                    choices=unique_transfus,
                    type='index',
                    value=lambda: random.choice(unique_transfus),
                )
                
                WNDINF = gr.Radio(
                    label="Wound Infection",
                    choices=unique_wndinf,
                    type='index',
                    value=lambda: random.choice(unique_wndinf),
                )

                DISCANCR = gr.Radio(
                    label="Disseminated Cancer",
                    choices=unique_discancr,
                    type='index',
                    value=lambda: random.choice(unique_discancr),
                )

                FNSTATUS2 = gr.Radio(
                    label="Functional Status",
                    choices=unique_fnstatus2,
                    type='index',
                    value=lambda: random.choice(unique_fnstatus2),
                )

                PRSODM = gr.Slider(label="Sodium", minimum=min(x['PRSODM']), maximum=max(x['PRSODM']), step=1, randomize=True)

                PRBUN = gr.Slider(label="BUN", minimum=min(x['PRBUN']), maximum=max(x['PRBUN']), step=1, randomize=True)

                PRCREAT = gr.Slider(label="Creatine", minimum=min(x['PRCREAT']),maximum=max(x['PRCREAT']), step=0.1, randomize=True)

                PRWBC = gr.Slider(label="WBC", minimum=min(x['PRWBC']), maximum=max(x['PRWBC']), step=0.1, randomize=True)

                PRHCT = gr.Slider(label="Hematocrit", minimum=min(x['PRHCT']), maximum=max(x['PRHCT']), step=0.1, randomize=True)

                PRPLATE = gr.Slider(label="Platelet", minimum=min(x['PRPLATE']), maximum=max(x['PRPLATE']), step=1, randomize=True)

                ASACLAS = gr.Radio(
                    label="ASA Class",
                    choices=unique_asaclas,
                    type='index',
                    value=lambda: random.choice(unique_asaclas),

                )

                LEVELS = gr.Radio(
                    label="Levels",
                    choices=unique_levels,
                    type='index',
                    value=lambda: random.choice(unique_levels),
                )

            with gr.Column():

                with gr.Row():
                    y3_predict_btn_xgb = gr.Button(value="Predict (XGBoost)")
                    y3_predict_btn_lgb = gr.Button(value="Predict (LightGBM)")
                    y3_predict_btn_cb = gr.Button(value="Predict (CatBoost)")
                    y3_predict_btn_rf = gr.Button(value="Predict (Random Forest)")
                label = gr.Label()

                with gr.Row():
                    y3_interpret_btn_xgb = gr.Button(value="Explain (XGBoost)")
                    y3_interpret_btn_lgb = gr.Button(value="Explain (LightGBM)")
                    y3_interpret_btn_cb = gr.Button(value="Explain (CatBoost)")
                    y3_interpret_btn_rf = gr.Button(value="Explain (Random Forest)") 

                plot = gr.Plot()

                y3_predict_btn_xgb.click(
                    y3_predict_xgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y3_predict_btn_lgb.click(
                    y3_predict_lgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y3_predict_btn_cb.click(
                    y3_predict_cb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y3_predict_btn_rf.click(
                    y3_predict_rf,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y3_interpret_btn_xgb.click(
                    y3_interpret_xgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )

                y3_interpret_btn_lgb.click(
                    y3_interpret_lgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )

                y3_interpret_btn_cb.click(
                    y3_interpret_cb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )

                y3_interpret_btn_rf.click(
                    y3_interpret_rf,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )

demo.launch()
