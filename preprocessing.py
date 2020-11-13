import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import joblib

# Individual pre-processing and training functions
# ================================================

#loading the source dataset from the path
def load_data(df_path):
    # Function loads data for training
    print("data loaded successfully")
    return pd.read_csv(df_path)

#replacing the question marks in the data
def replace_ques(data):
    return data.replace('?', np.nan)

#retain only the first cabin if more than 1 are available per passenger
def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return np.nan

# extracts the title (Mr, Ms, etc) from the name variable
def get_title(passenger):
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'

# cast numerical variables as floats
def numerical_to_floatcasting(df,var):
    return df[var].astype('float')

#droppin irrelevant columns
def drop_columns(df,colu):
    df.drop(labels=colu,axis=1,inplace=True)
    return df

#splitting data into X(independent) and y(dependent)
def split_X_y(df):
    X=df.copy()
    index_y=df.index
    y=pd.DataFrame(data=df['survived'], columns=['survived'])
    X=X.drop('survived',axis=1)
    print('X,y created')
    return X,y

def drop_non_alphabetic(df_x,var):
    return df_x[var].replace('[^a-zA-Z]','')
#replace Nan numeric values to mean
def replace_missing_num(X,VARS_NUM_MS):

    return X[VARS_NUM_MS].fillna(X[VARS_NUM_MS].mean())

#replace missing categorial
def replace_missing_cat(X,VARS_CAT_MS):

    return X[VARS_CAT_MS].fillna('Missing')

def remove_rare_labels(df, var, frequent_labels):
    return np.where(df[var].isin(frequent_labels), df[var], 'Rare')

def encoding(X):
    return pd.get_dummies(X,drop_first=True)


def scaling_featur_matix(X):
    cols=X.columns
    index_x=X.index
    scaler = StandardScaler()
    scaler.fit(X[cols])
    a=scaler.transform(X[cols])
    df=pd.DataFrame(a,index=index_x,columns=cols)
    return df

def divide_train_test(X,y):
    # Function divides data set in train and test
    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size=0.1,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test

def train_model(X_train,y_train,output_path):
    logreg = LogisticRegression(random_state=0)
    logreg.fit(X_train, y_train)
    joblib.dump(logreg, output_path)
    return None

def predict(X_test, model):
    model = joblib.load(model)
    return model.predict(X_test)
