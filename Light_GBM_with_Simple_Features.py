# Forked from excellent kernel : https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('input/application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('input/application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    #df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    #df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    #df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    #df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    #df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    #df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    #df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    #df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    #df['NEW_PHONE_TO_EMPLOY_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    #df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['TH_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / (1 + df['AMT_INCOME_TOTAL'])
    df['TH_CREDIT_TO_ANNUITY_TO_INCOME_RATIO'] = df['NEW_CREDIT_TO_ANNUITY_RATIO'] / (1 + df['AMT_INCOME_TOTAL'])
    
    df['TH_CREDIT_TO_EMPLOY_RATIO'] = df['AMT_CREDIT'] / df['DAYS_EMPLOYED']
    df['TH_ANNUITY_TO_EMPLOY_RATIO'] = df['AMT_ANNUITY'] / df['DAYS_EMPLOYED']
    df['TH_ANNUITY_TO_GOODS_RATIO'] = df['AMT_ANNUITY'] / df['AMT_GOODS_PRICE']
    #df['TH_CREDIT_TO_BIRTH_RATIO'] = df['AMT_CREDIT'] / df['DAYS_BIRTH']
    #df['TH_CREDIT_TO_EXT_SOURCES_MEAN_RATIO'] = df['AMT_CREDIT'] / df['NEW_EXT_SOURCES_MEAN']
    df['TH_CREDIT_TO_EXT_SOURCE_1_RATIO'] = df['AMT_CREDIT'] / df['EXT_SOURCE_1']
    df['TH_CREDIT_TO_EXT_SOURCE_2_RATIO'] = df['AMT_CREDIT'] / df['EXT_SOURCE_2']
    df['TH_CREDIT_TO_EXT_SOURCE_3_RATIO'] = df['AMT_CREDIT'] / df['EXT_SOURCE_3']
    #df['TH_CREDIT_TO_EXT_SOURCES_PROD_RATIO'] = df['AMT_CREDIT'] / (df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3'])
    df['TH_ANNUIT_TO_EXT_SOURCE_1_RATIO'] = df['AMT_ANNUITY'] / df['EXT_SOURCE_1']
    df['TH_ANNUIT_TO_EXT_SOURCE_2_RATIO'] = df['AMT_ANNUITY'] / df['EXT_SOURCE_2']
    df['TH_ANNUIT_TO_EXT_SOURCE_3_RATIO'] = df['AMT_ANNUITY'] / df['EXT_SOURCE_3']
    df['TH_EXT_SOURCE_1_TO_EMPLOY_RATIO'] = df['EXT_SOURCE_1'] / df['DAYS_EMPLOYED']
    df['TH_EXT_SOURCE_2_TO_EMPLOY_RATIO'] = df['EXT_SOURCE_2'] / df['DAYS_EMPLOYED']
    df['TH_EXT_SOURCE_3_TO_EMPLOY_RATIO'] = df['EXT_SOURCE_3'] / df['DAYS_EMPLOYED']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # Some simple new features (percentages)
    #df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    #df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    #df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    #df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    #df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
    #dropcolum=['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4',
    #'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',
    #'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 
    #'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
    #'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16',
    #'FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',
    #'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
    #df= df.drop(dropcolum,axis=1)
    
    del test_df
    gc.collect()
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('input/bureau.csv', nrows = num_rows)
    bb = pd.read_csv('input/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    cols = active_agg.columns.tolist()
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    
    for e in cols:
        bureau_agg['NEW_RATIO_BURO_' + e[0] + "_" + e[1].upper()] = bureau_agg['ACTIVE_' + e[0] + "_" + e[1].upper()] / bureau_agg['CLOSED_' + e[0] + "_" + e[1].upper()]
    
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('input/previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    cols = approved_agg.columns.tolist()
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    
    for e in cols:
        prev_agg['NEW_RATIO_PREV_' + e[0] + "_" + e[1].upper()] = prev_agg['APPROVED_' + e[0] + "_" + e[1].upper()] / prev_agg['REFUSED_' + e[0] + "_" + e[1].upper()]
    
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('input/POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg
    
# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('input/installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('input/credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified = False, debug= False):
    df2 = df[['NEW_CREDIT_TO_ANNUITY_RATIO',\
'NEW_EXT_SOURCES_MEAN',\
'DAYS_BIRTH',\
'EXT_SOURCE_2',\
'EXT_SOURCE_3',\
'NEW_CREDIT_TO_GOODS_RATIO',\
'DAYS_ID_PUBLISH',\
'APPROVED_CNT_PAYMENT_MEAN',\
'INSTAL_DAYS_ENTRY_PAYMENT_MAX',\
'TH_ANNUIT_TO_EXT_SOURCE_3_RATIO',\
'TH_ANNUITY_TO_GOODS_RATIO',\
'NEW_ANNUITY_TO_INCOME_RATIO',\
'TH_ANNUITY_TO_EMPLOY_RATIO',\
'DAYS_REGISTRATION',\
'INSTAL_DPD_MEAN',\
'ACTIVE_DAYS_CREDIT_MAX',\
'AMT_ANNUITY',\
'PREV_CNT_PAYMENT_MEAN',\
'ACTIVE_DAYS_CREDIT_ENDDATE_MIN',\
'INSTAL_DBD_SUM',\
'INSTAL_AMT_PAYMENT_SUM',\
'EXT_SOURCE_1',\
'REGION_POPULATION_RELATIVE',\
'DAYS_LAST_PHONE_CHANGE',\
'DAYS_EMPLOYED',\
'TH_ANNUIT_TO_EXT_SOURCE_1_RATIO',\
'NEW_EMPLOY_TO_BIRTH_RATIO',\
'CLOSED_DAYS_CREDIT_MAX',\
'TH_CREDIT_TO_EMPLOY_RATIO',\
'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN',\
'TH_EXT_SOURCE_2_TO_EMPLOY_RATIO',\
'NEW_SCORES_STD',\
'OWN_CAR_AGE',\
'TH_CREDIT_TO_EXT_SOURCE_3_RATIO',\
'TH_ANNUIT_TO_EXT_SOURCE_2_RATIO',\
'APPROVED_DAYS_DECISION_MAX',\
'BURO_DAYS_CREDIT_MAX',\
'TH_EXT_SOURCE_3_TO_EMPLOY_RATIO',\
'TH_CREDIT_TO_EXT_SOURCE_2_RATIO',\
'CLOSED_DAYS_CREDIT_ENDDATE_MAX',\
'AMT_GOODS_PRICE',\
'INSTAL_AMT_PAYMENT_MIN',\
'ACTIVE_DAYS_CREDIT_UPDATE_MEAN',\
'INSTAL_DBD_MEAN',\
'TH_CREDIT_TO_EXT_SOURCE_1_RATIO',\
'BURO_AMT_CREDIT_SUM_MEAN',\
'BURO_DAYS_CREDIT_ENDDATE_MAX',\
'TH_EXT_SOURCE_1_TO_EMPLOY_RATIO',\
'POS_MONTHS_BALANCE_SIZE',\
'INSTAL_DBD_MAX',\
'ACTIVE_DAYS_CREDIT_ENDDATE_MEAN',\
'PREV_APP_CREDIT_PERC_VAR',\
'NEW_RATIO_BURO_DAYS_CREDIT_MAX',\
'INSTAL_DAYS_ENTRY_PAYMENT_MEAN',\
'ACTIVE_AMT_CREDIT_SUM_SUM',\
'POS_NAME_CONTRACT_STATUS_Active_MEAN',\
'TH_CREDIT_TO_INCOME_RATIO',\
'POS_MONTHS_BALANCE_MEAN',\
'ACTIVE_DAYS_CREDIT_ENDDATE_MAX',\
'PREV_AMT_ANNUITY_MEAN',\
'ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN',\
'TH_CREDIT_TO_ANNUITY_TO_INCOME_RATIO',\
'PREV_DAYS_DECISION_MAX',\
'BURO_AMT_CREDIT_SUM_DEBT_MEAN',\
'NEW_RATIO_BURO_DAYS_CREDIT_ENDDATE_MAX',\
'CLOSED_DAYS_CREDIT_VAR',\
'CLOSED_AMT_CREDIT_SUM_MEAN',\
'BURO_AMT_CREDIT_SUM_SUM',\
'AMT_CREDIT',\
'APPROVED_CNT_PAYMENT_SUM',\
'INSTAL_PAYMENT_DIFF_MAX',\
'BURO_DAYS_CREDIT_MEAN',\
'INSTAL_DAYS_ENTRY_PAYMENT_SUM',\
'ACTIVE_DAYS_CREDIT_MEAN',\
'APPROVED_APP_CREDIT_PERC_VAR',\
'INSTAL_PAYMENT_PERC_MEAN',\
'BURO_DAYS_CREDIT_VAR',\
'INSTAL_AMT_PAYMENT_MEAN',\
'PREV_HOUR_APPR_PROCESS_START_MEAN',\
'ACTIVE_AMT_CREDIT_SUM_MEAN',\
'PREV_APP_CREDIT_PERC_MEAN',\
'INSTAL_PAYMENT_DIFF_MEAN',\
'BURO_AMT_CREDIT_SUM_MAX',\
'APPROVED_AMT_ANNUITY_MEAN',\
'NEW_RATIO_BURO_DAYS_CREDIT_UPDATE_MEAN',\
'BURO_DAYS_CREDIT_ENDDATE_MEAN',\
'POS_NAME_CONTRACT_STATUS_Completed_MEAN',\
'INSTAL_AMT_INSTALMENT_MAX',\
'PREV_NAME_YIELD_GROUP_high_MEAN',\
'CODE_GENDER',\
'CLOSED_AMT_CREDIT_SUM_SUM',\
'CLOSED_DAYS_CREDIT_UPDATE_MEAN',\
'ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN',\
'ACTIVE_DAYS_CREDIT_VAR',\
'PREV_APP_CREDIT_PERC_MIN',\
'INSTAL_AMT_INSTALMENT_SUM',\
'ACTIVE_AMT_CREDIT_SUM_DEBT_SUM',\
'PREV_DAYS_DECISION_MEAN',\
'NEW_RATIO_BURO_AMT_CREDIT_SUM_SUM',\
'PREV_AMT_ANNUITY_MIN',\
'ACTIVE_AMT_CREDIT_SUM_MAX',\
'INSTAL_AMT_PAYMENT_MAX',\
'APPROVED_APP_CREDIT_PERC_MEAN',\
'APPROVED_AMT_DOWN_PAYMENT_MAX',\
'NEW_RATIO_BURO_DAYS_CREDIT_ENDDATE_MIN',\
'PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN',\
'NEW_RATIO_PREV_DAYS_DECISION_MAX',\
'NEW_RATIO_BURO_AMT_CREDIT_SUM_MAX',\
'AMT_INCOME_TOTAL',\
'PREV_NAME_TYPE_SUITE_nan_MEAN',\
'POS_MONTHS_BALANCE_MAX',\
'POS_SK_DPD_DEF_MEAN',\
'NEW_RATIO_BURO_AMT_CREDIT_SUM_MEAN',\
'NEW_RATIO_BURO_DAYS_CREDIT_MEAN',\
'CLOSED_AMT_CREDIT_SUM_MAX',\
'APPROVED_DAYS_DECISION_MEAN',\
'ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN',\
'CC_CNT_DRAWINGS_ATM_CURRENT_MEAN',\
'APPROVED_HOUR_APPR_PROCESS_START_MEAN',\
'NEW_RATIO_BURO_DAYS_CREDIT_MIN',\
'APPROVED_APP_CREDIT_PERC_MIN',\
'CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN',\
'BURO_STATUS_0_MEAN_MEAN',\
'INSTAL_PAYMENT_DIFF_SUM',\
'APPROVED_AMT_CREDIT_MIN',\
'APPROVED_AMT_ANNUITY_MIN',\
'BURO_DAYS_CREDIT_MIN',\
'BURO_DAYS_CREDIT_UPDATE_MEAN',\
'INSTAL_PAYMENT_PERC_VAR',\
'NEW_RATIO_BURO_DAYS_CREDIT_ENDDATE_MEAN',\
'PREV_AMT_DOWN_PAYMENT_MAX',\
'PREV_DAYS_DECISION_MIN',\
'ACTIVE_AMT_CREDIT_SUM_DEBT_MAX',\
'INSTAL_PAYMENT_PERC_SUM',\
'CLOSED_DAYS_CREDIT_ENDDATE_MEAN',\
'ACTIVE_DAYS_CREDIT_MIN',\
'APPROVED_DAYS_DECISION_MIN',\
'TOTALAREA_MODE',\
'INSTAL_AMT_INSTALMENT_MEAN',\
'BURO_AMT_CREDIT_SUM_DEBT_MAX',\
'APPROVED_AMT_CREDIT_MAX',\
'PREV_APP_CREDIT_PERC_MAX',\
'PREV_RATE_DOWN_PAYMENT_MEAN',\
'BURO_CREDIT_TYPE_Credit card_MEAN',\
'BURO_AMT_CREDIT_SUM_DEBT_SUM',\
'NAME_FAMILY_STATUS_Married',\
'APPROVED_APP_CREDIT_PERC_MAX',\
'APPROVED_AMT_ANNUITY_MAX',\
'CLOSED_DAYS_CREDIT_MIN',\
'APPROVED_AMT_GOODS_PRICE_MIN',\
'PREV_AMT_DOWN_PAYMENT_MEAN',\
'HOUR_APPR_PROCESS_START',\
'PREV_AMT_ANNUITY_MAX',\
'BURO_DAYS_CREDIT_ENDDATE_MIN',\
'CLOSED_DAYS_CREDIT_MEAN',\
'NEW_RATIO_BURO_DAYS_CREDIT_VAR',\
'PREV_NAME_YIELD_GROUP_middle_MEAN',\
'PREV_AMT_APPLICATION_MEAN',\
'PREV_WEEKDAY_APPR_PROCESS_START_THURSDAY_MEAN',\
'PREV_NAME_CONTRACT_STATUS_Refused_MEAN',\
'PREV_CNT_PAYMENT_SUM',\
'CLOSED_DAYS_CREDIT_ENDDATE_MIN',\
'CC_CNT_DRAWINGS_CURRENT_VAR',\
'PREV_AMT_GOODS_PRICE_MIN',\
'NEW_RATIO_PREV_DAYS_DECISION_MIN',\
'PREV_AMT_CREDIT_MEAN',\
'PREV_NAME_YIELD_GROUP_low_normal_MEAN',\
'INSTAL_PAYMENT_DIFF_VAR',\
'REGION_RATING_CLIENT_W_CITY',\
'APPROVED_RATE_DOWN_PAYMENT_MAX',\
'PREV_WEEKDAY_APPR_PROCESS_START_MONDAY_MEAN',\
'APPROVED_AMT_APPLICATION_MIN',\
'BURO_CREDIT_TYPE_Consumer credit_MEAN',\
'PREV_WEEKDAY_APPR_PROCESS_START_FRIDAY_MEAN',\
'PREV_WEEKDAY_APPR_PROCESS_START_SATURDAY_MEAN',\
'INSTAL_DPD_MAX',\
'CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN',\
'BURO_CREDIT_ACTIVE_Closed_MEAN',\
'REFUSED_DAYS_DECISION_MAX',\
'CC_AMT_PAYMENT_CURRENT_MEAN',\
'PREV_RATE_DOWN_PAYMENT_MAX',\
'INSTAL_COUNT',\
'PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN',\
'APPROVED_AMT_DOWN_PAYMENT_MEAN',\
'BURO_AMT_CREDIT_SUM_LIMIT_MEAN',\
'PREV_PRODUCT_COMBINATION_Cash X-Sell: low_MEAN',\
'NEW_RATIO_PREV_HOUR_APPR_PROCESS_START_MEAN',\
'CC_CNT_DRAWINGS_CURRENT_MEAN',\
'INSTAL_DPD_SUM',\
'PREV_NAME_CLIENT_TYPE_New_MEAN',\
'PREV_NAME_YIELD_GROUP_low_action_MEAN',\
'APPROVED_AMT_APPLICATION_MAX',\
'APPROVED_AMT_CREDIT_MEAN',\
'APPROVED_RATE_DOWN_PAYMENT_MEAN',\
'CC_CNT_DRAWINGS_ATM_CURRENT_VAR',\
'NEW_RATIO_PREV_AMT_ANNUITY_MAX',\
'PREV_NAME_PAYMENT_TYPE_XNA_MEAN',\
'ACTIVE_MONTHS_BALANCE_SIZE_MEAN',\
'PREV_CHANNEL_TYPE_Country-wide_MEAN',\
'PREV_CHANNEL_TYPE_Credit and cash offices_MEAN',\
'FLAG_WORK_PHONE',\
'BURO_CREDIT_TYPE_Microloan_MEAN',\
'NEW_RATIO_PREV_CNT_PAYMENT_SUM',\
'PREV_AMT_GOODS_PRICE_MEAN',\
'PREV_NAME_YIELD_GROUP_XNA_MEAN',\
'PREV_WEEKDAY_APPR_PROCESS_START_WEDNESDAY_MEAN',\
'NEW_RATIO_PREV_APP_CREDIT_PERC_MAX',\
'PREV_AMT_CREDIT_MIN',\
'PREV_AMT_CREDIT_MAX',\
'BURO_STATUS_1_MEAN_MEAN',\
'PREV_NAME_SELLER_INDUSTRY_Consumer electronics_MEAN',\
'POS_SK_DPD_MEAN',\
'BURO_CREDIT_ACTIVE_Active_MEAN',\
'PREV_WEEKDAY_APPR_PROCESS_START_TUESDAY_MEAN',\
'DEF_30_CNT_SOCIAL_CIRCLE',\
'REFUSED_AMT_ANNUITY_MIN',\
'PREV_HOUR_APPR_PROCESS_START_MAX',\
'PREV_RATE_DOWN_PAYMENT_MIN',\
'NEW_RATIO_PREV_DAYS_DECISION_MEAN',\
'PREV_CHANNEL_TYPE_Stone_MEAN',\
'NEW_RATIO_PREV_APP_CREDIT_PERC_MEAN',\
'REFUSED_DAYS_DECISION_MEAN',\
'PREV_NAME_CLIENT_TYPE_Refreshed_MEAN',\
'NEW_RATIO_PREV_CNT_PAYMENT_MEAN',\
'BURO_STATUS_X_MEAN_MEAN',\
'BURO_CREDIT_TYPE_Mortgage_MEAN',\
'YEARS_BEGINEXPLUATATION_MEDI',\
'PREV_NAME_PRODUCT_TYPE_walk-in_MEAN',\
'REFUSED_DAYS_DECISION_MIN',\
'PREV_PRODUCT_COMBINATION_POS household with interest_MEAN',\
'PREV_PRODUCT_COMBINATION_Cash Street: low_MEAN',\
'APPROVED_HOUR_APPR_PROCESS_START_MAX',\
'LANDAREA_AVG',\
'NEW_RATIO_PREV_AMT_ANNUITY_MIN',\
'APARTMENTS_MODE',\
'YEARS_BEGINEXPLUATATION_MODE',\
'APPROVED_AMT_APPLICATION_MEAN',\
'PREV_NAME_PAYMENT_TYPE_Cash through the bank_MEAN',\
'PREV_AMT_APPLICATION_MIN',\
'PREV_NAME_PRODUCT_TYPE_XNA_MEAN',\
'BURO_MONTHS_BALANCE_SIZE_SUM',\
'NAME_EDUCATION_TYPE_Higher education',\
'NEW_RATIO_PREV_HOUR_APPR_PROCESS_START_MAX',\
'LANDAREA_MODE',\
'PREV_NAME_CONTRACT_STATUS_Approved_MEAN',\
'OBS_60_CNT_SOCIAL_CIRCLE',\
'NEW_RATIO_PREV_HOUR_APPR_PROCESS_START_MIN',\
'INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE',\
'REFUSED_APP_CREDIT_PERC_MIN',\
'REFUSED_AMT_CREDIT_MIN',\
'REFUSED_HOUR_APPR_PROCESS_START_MEAN',\
'YEARS_BEGINEXPLUATATION_AVG',\
'PREV_PRODUCT_COMBINATION_Cash X-Sell: high_MEAN',\
'LIVINGAREA_MODE',\
'APARTMENTS_AVG',\
'NEW_RATIO_PREV_AMT_ANNUITY_MEAN',\
'BURO_MONTHS_BALANCE_SIZE_MEAN',\
'APPROVED_AMT_DOWN_PAYMENT_MIN',\
'PREV_AMT_APPLICATION_MAX',\
'BURO_STATUS_C_MEAN_MEAN',\
'PREV_NAME_TYPE_SUITE_Family_MEAN',\
'PREV_AMT_DOWN_PAYMENT_MIN',\
'PREV_NAME_GOODS_CATEGORY_Mobile_MEAN',\
'BASEMENTAREA_AVG',\
'CLOSED_MONTHS_BALANCE_SIZE_MEAN',\
'NEW_RATIO_PREV_AMT_GOODS_PRICE_MEAN',\
'CLOSED_AMT_ANNUITY_MAX',\
'APPROVED_AMT_GOODS_PRICE_MEAN',\
'APARTMENTS_MEDI',\
'BASEMENTAREA_MODE',\
'LIVINGAREA_AVG',\
'LANDAREA_MEDI',\
'PREV_NAME_CASH_LOAN_PURPOSE_XNA_MEAN',\
'BURO_AMT_ANNUITY_MEAN',\
'FLAG_DOCUMENT_3',\
'CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN',\
'OBS_30_CNT_SOCIAL_CIRCLE',\
'CC_AMT_CREDIT_LIMIT_ACTUAL_SUM',\
'ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM',\
'CC_AMT_DRAWINGS_ATM_CURRENT_MEAN',\
'PREV_NAME_CLIENT_TYPE_Repeater_MEAN',\
'PREV_NAME_CONTRACT_STATUS_Canceled_MEAN',\
'CLOSED_MONTHS_BALANCE_SIZE_SUM',\
'PREV_NAME_PRODUCT_TYPE_x-sell_MEAN',\
'CC_AMT_PAYMENT_CURRENT_SUM',\
'ORGANIZATION_TYPE_Self-employed',\
'NONLIVINGAREA_AVG',\
'PREV_WEEKDAY_APPR_PROCESS_START_SUNDAY_MEAN',\
'REFUSED_APP_CREDIT_PERC_MEAN',\
'APPROVED_RATE_DOWN_PAYMENT_MIN',\
'NEW_RATIO_PREV_APP_CREDIT_PERC_MIN',\
'NEW_RATIO_PREV_AMT_GOODS_PRICE_MIN',\
'NEW_RATIO_PREV_AMT_CREDIT_MIN',\
'APPROVED_AMT_GOODS_PRICE_MAX',\
'NEW_RATIO_BURO_MONTHS_BALANCE_SIZE_MEAN',\
'PREV_NAME_PORTFOLIO_Cash_MEAN',\
'REFUSED_APP_CREDIT_PERC_VAR',\
'NEW_RATIO_PREV_AMT_CREDIT_MAX',\
'POS_SK_DPD_DEF_MAX',\
'APPROVED_HOUR_APPR_PROCESS_START_MIN',\
'BURO_AMT_CREDIT_SUM_LIMIT_SUM',\
'CC_AMT_INST_MIN_REGULARITY_VAR',\
'NEW_RATIO_PREV_APP_CREDIT_PERC_VAR',\
'ACTIVE_MONTHS_BALANCE_SIZE_SUM',\
'NONLIVINGAREA_MODE',\
'PREV_PRODUCT_COMBINATION_POS mobile with interest_MEAN',\
'AMT_REQ_CREDIT_BUREAU_QRT',\
'LIVINGAREA_MEDI',\
'NEW_RATIO_BURO_MONTHS_BALANCE_MIN_MIN',\
'DEF_60_CNT_SOCIAL_CIRCLE',\
'PREV_CHANNEL_TYPE_Contact center_MEAN',\
'PREV_CHANNEL_TYPE_AP+ (Cash loan)_MEAN',\
'POS_NAME_CONTRACT_STATUS_Signed_MEAN',\
'BASEMENTAREA_MEDI',\
'PREV_PRODUCT_COMBINATION_Cash X-Sell: middle_MEAN',\
'NEW_RATIO_BURO_MONTHS_BALANCE_SIZE_SUM',\
'PREV_NAME_PORTFOLIO_XNA_MEAN',\
'AMT_REQ_CREDIT_BUREAU_YEAR',\
'BURO_CREDIT_TYPE_Car loan_MEAN',\
'REFUSED_CNT_PAYMENT_MEAN',\
'NAME_EDUCATION_TYPE_Secondary / secondary special',\
'PREV_HOUR_APPR_PROCESS_START_MIN',\
'BURO_AMT_ANNUITY_MAX',\
'REFUSED_AMT_ANNUITY_MAX',\
'CC_AMT_CREDIT_LIMIT_ACTUAL_VAR',\
'ACTIVE_AMT_ANNUITY_MEAN',\
'NEW_RATIO_PREV_AMT_APPLICATION_MEAN',\
'POS_COUNT',\
'CC_AMT_DRAWINGS_ATM_CURRENT_SUM',\
'REG_CITY_NOT_LIVE_CITY',\
'REFUSED_AMT_ANNUITY_MEAN',\
'PREV_NAME_GOODS_CATEGORY_Audio/Video_MEAN',\
'CC_AMT_DRAWINGS_ATM_CURRENT_MAX',\
'ACTIVE_AMT_ANNUITY_MAX',\
'PREV_NAME_GOODS_CATEGORY_Computers_MEAN',\
'PREV_CODE_REJECT_REASON_XAP_MEAN',\
'REFUSED_AMT_GOODS_PRICE_MEAN',\
'PREV_NAME_CONTRACT_TYPE_Cash loans_MEAN',\
'CC_CNT_DRAWINGS_CURRENT_MAX',\
'NEW_RATIO_PREV_AMT_GOODS_PRICE_MAX',\
'CC_AMT_PAYMENT_CURRENT_MAX',\
'CC_AMT_DRAWINGS_ATM_CURRENT_VAR',\
'PREV_NAME_SELLER_INDUSTRY_XNA_MEAN',\
'CC_AMT_PAYMENT_CURRENT_VAR',\
'PREV_NAME_GOODS_CATEGORY_Consumer Electronics_MEAN',\
'POS_SK_DPD_MAX',\
'PREV_PRODUCT_COMBINATION_POS industry with interest_MEAN',\
'PREV_PRODUCT_COMBINATION_Cash_MEAN',\
'BURO_AMT_CREDIT_SUM_OVERDUE_MEAN',\
'CC_AMT_PAYMENT_CURRENT_MIN',\
'TARGET','SK_ID_CURR','index']]
    del df
    gc.collect()
    # Divide in training/validation and test data
    train_df = df2[df2['TARGET'].notnull()]
    test_df = df2[df2['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    print("All features: ")
    #for c in df.columns: print(c)
    del df2
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=8,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=50, #34,
            colsample_bytree=0.88, #0.9497036,
            subsample=0.8715623,
            subsample_freq=1,
            max_depth=9, #8
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.05, #0.0222415,
            min_child_weight=39.3259775,
            random_state=0,
            silent=-1,
            verbose=-1, )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)
    display_importances(feature_importance_df)
    return feature_importance_df

# Display/plot feature importance
def display_importances(feature_importance_df_):
    c200 = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:200].index
    for c in c200: print(c)
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:35].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances21.png')

def main(debug = False):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(df, num_folds= 5, stratified= True, debug= debug)

if __name__ == "__main__":
    submission_file_name = "submission_lgbm21.csv"
    with timer("Full model run"):
        main()

