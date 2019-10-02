import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn import cross_validation, decomposition, grid_search
from sklearn.preprocessing import LabelEncoder

####################################################
# Functions                                        #
####################################################
# Remove outliers
def remove_outliers(df, column, min_val, max_val):
    col_values = df[column].values
    df[column] = np.where(np.logical_or(col_values<=min_val, col_values>=max_val), np.NaN, col_values)
   
    return df

# Home made One Hot Encoder
def convert_to_binary(df, column_to_convert):
    categories = list(df[column_to_convert].drop_duplicates())
   
    for category in categories:
        cat_name = str(category).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "").lower()
        col_name = column_to_convert[:5] + '_' + cat_name[:10]
        df[col_name] = 0
        df.loc[(df[column_to_convert] == category), col_name] = 1
   
    return df

# Count occurrences of value in a column
def convert_to_counts(df, id_col, column_to_convert):
    id_list = df[id_col].drop_duplicates()
   
    df_counts = df.loc[:,[id_col, column_to_convert]]
    df_counts['count'] = 1
    df_counts = df_counts.groupby(by=[id_col, column_to_convert], as_index=False, sort=False).sum()
   
    new_df = df_counts.pivot(index=id_col, columns=column_to_convert, values='count')
    new_df = new_df.fillna(0)
   
    # Rename Columns
    categories = list(df[column_to_convert].drop_duplicates())
    for category in categories:
        cat_name = str(category).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "").lower()
        col_name = column_to_convert + '_' + cat_name
        new_df.rename(columns = {category:col_name}, inplace=True)
       
    return new_df

####################################################
# Cleaning                                         #
####################################################
# Import data
print("Reading in data...")
tr_filepath = "./train_users_2.csv"
df_train = pd.read_csv(tr_filepath, header=0, index_col=None)
te_filepath = "./test_users.csv"
df_test = pd.read_csv(te_filepath, header=0, index_col=None)

# Combine into one dataset
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

# Change Dates to consistent format
print("Fixing timestamps...")
df_all['date_account_created'] =  pd.to_datetime(df_all['date_account_created'], format='%Y-%m-%d')
df_all['timestamp_first_active'] =  pd.to_datetime(df_all['timestamp_first_active'], format='%Y%m%d%H%M%S')
df_all['date_account_created'].fillna(df_all.timestamp_first_active, inplace=True)

# Remove date_first_booking column
df_all.drop('date_first_booking', axis=1, inplace=True)

# Fixing age column
print("Fixing age column...")
df_all = remove_outliers(df=df_all, column='age', min_val=15, max_val=90)
df_all['age'].fillna(-1, inplace=True)

# Fill first_affiliate_tracked column
print("Filling first_affiliate_tracked column...")
df_all['first_affiliate_tracked'].fillna(-1, inplace=True)

####################################################
# Data Transformation                              #
####################################################
# One Hot Encoding
print("One Hot Encoding categorical data...")
columns_to_convert = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']

for column in columns_to_convert:
    df_all = convert_to_binary(df=df_all, column_to_convert=column)
    df_all.drop(column, axis=1, inplace=True)

####################################################
# Feature Extraction                               #
####################################################
# Add new date related fields
print("Adding new fields...")
df_all['day_account_created'] = df_all['date_account_created'].dt.weekday
df_all['month_account_created'] = df_all['date_account_created'].dt.month
df_all['quarter_account_created'] = df_all['date_account_created'].dt.quarter
df_all['year_account_created'] = df_all['date_account_created'].dt.year
df_all['hour_first_active'] = df_all['timestamp_first_active'].dt.hour
df_all['day_first_active'] = df_all['timestamp_first_active'].dt.weekday
df_all['month_first_active'] = df_all['timestamp_first_active'].dt.month
df_all['quarter_first_active'] = df_all['timestamp_first_active'].dt.quarter
df_all['year_first_active'] = df_all['timestamp_first_active'].dt.year
df_all['created_less_active'] = (df_all['date_account_created'] - df_all['timestamp_first_active']).dt.days

# Drop unnecessary columns
columns_to_drop = ['date_account_created', 'timestamp_first_active', 'date_first_booking', 'country_destination']
for column in columns_to_drop:
    if column in df_all.columns:
        df_all.drop(column, axis=1, inplace=True)

####################################################
# Add data from sessions.csv                       #
####################################################
# Import sessions data
s_filepath = "./sessions.csv"
sessions = pd.read_csv(s_filepath, header=0, index_col=False)

# Determine primary device
print("Determing primary device...")
sessions_device = sessions.loc[:, ['user_id', 'device_type', 'secs_elapsed']]
aggregated_lvl1 = sessions_device.groupby(['user_id', 'device_type'], as_index=False, sort=False).aggregate(np.sum)
idx = aggregated_lvl1.groupby(['user_id'], sort=False)['secs_elapsed'].transform(max) == aggregated_lvl1['secs_elapsed']
df_primary = pd.DataFrame(aggregated_lvl1.loc[idx , ['user_id', 'device_type', 'secs_elapsed']])
df_primary.rename(columns = {'device_type':'primary_device', 'secs_elapsed':'primary_secs'}, inplace=True)
df_primary = convert_to_binary(df=df_primary, column_to_convert='primary_device')
df_primary.drop('primary_device', axis=1, inplace=True)

# Determine Secondary device
print("Determing secondary device...")
remaining = aggregated_lvl1.drop(aggregated_lvl1.index[idx])
idx = remaining.groupby(['user_id'], sort=False)['secs_elapsed'].transform(max) == remaining['secs_elapsed']
df_secondary = pd.DataFrame(remaining.loc[idx , ['user_id', 'device_type', 'secs_elapsed']])
df_secondary.rename(columns = {'device_type':'secondary_device', 'secs_elapsed':'secondary_secs'}, inplace=True)
df_secondary = convert_to_binary(df=df_secondary, column_to_convert='secondary_device')
df_secondary.drop('secondary_device', axis=1, inplace=True)

# Aggregate and combine actions taken columns
print("Aggregating actions taken...")
session_actions = sessions.loc[:,['user_id', 'action', 'action_type', 'action_detail']]
columns_to_convert = ['action', 'action_type', 'action_detail']
session_actions = session_actions.fillna('not provided')
first = True

for column in columns_to_convert:
    print("Converting " + column + " column...")
    current_data = convert_to_counts(df=session_actions, id_col='user_id', column_to_convert=column)

    # If first loop, current data becomes existing data, otherwise merge existing and current
    if first:
        first = False
        actions_data = current_data
    else:
        actions_data = pd.concat([actions_data, current_data], axis=1, join='inner')

# Merge device datasets
print("Combining results...")
df_primary.set_index('user_id', inplace=True)
df_secondary.set_index('user_id', inplace=True)
device_data = pd.concat([df_primary, df_secondary], axis=1, join="outer")

# Merge device and actions datasets
combined_results = pd.concat([device_data, actions_data], axis=1, join='outer')
df_sessions = combined_results.fillna(0)

# Merge user and session datasets
df_all.set_index('id', inplace=True)
df_all = pd.concat([df_all, df_sessions], axis=1, join='inner')

####################################################
# Building Model                                   #
####################################################
# Prepare training data for modelling
df_train.set_index('id', inplace=True)
df_train = pd.concat([df_train['country_destination'], df_all], axis=1, join='inner')

id_train = df_train.index.values
labels = df_train['country_destination']
le = LabelEncoder()
y = le.fit_transform(labels)
X = df_train.drop('country_destination', axis=1, inplace=False)

# Training model
print("Training model...")

# Grid Search - Used to find best combination of parameters
XGB_model = xgb.XGBClassifier(objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)
param_grid = {'max_depth': [3, 4], 'learning_rate': [0.1, 0.3], 'n_estimators': [25, 50]}
model = grid_search.GridSearchCV(estimator=XGB_model, param_grid=param_grid, scoring='accuracy', verbose=10, n_jobs=1, iid=True, refit=True, cv=3)

model.fit(X, y)
print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

####################################################
# Make predictions                                 #
####################################################
print("Making predictions...")

# Prepare test data for prediction
df_test.set_index('id', inplace=True)
df_test = pd.merge(df_test.loc[:,['date_first_booking']], df_all, how='left', left_index=True, right_index=True, sort=False)
X_test = df_test.drop('date_first_booking', axis=1, inplace=False)
X_test = X_test.fillna(-1)
id_test = df_test.index.values

# Make predictions
y_pred = model.predict_proba(X_test)

#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

#Generate submission
print("Outputting final results...")
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('./submission.csv',index=False)
