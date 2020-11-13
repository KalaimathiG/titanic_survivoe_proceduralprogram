import numpy as np
import pandas as pd
import preprocessing as pf
import config

import warnings
warnings.simplefilter(action='ignore')

# ================================================
# TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL

data = pf.load_data(config.PATH_TO_DATASET)

data=pf.replace_ques(data)

data[config.CABIN]=data[config.CABIN].apply(pf.get_first_cabin)

data['title']=data['name'].apply(pf.get_title)

for var in config.NUMERICAL_TO_FLOAT:
    data[var]=pf.numerical_to_floatcasting(data,var)

data=pf.drop_columns(data,config.DROP_LABELS)

X,y=pf.split_X_y(data)

X[config.CABIN]=pf.drop_non_alphabetic(X,config.CABIN)

for var in config.VARS_NUM_MS:
    X[var]=pf.replace_missing_num(X,var)

for var in config.VARS_CAT_MS:
    X[var]=pf.replace_missing_cat(X,var)

for var in config.VARS_CAT:
    X[var]=pf.remove_rare_labels(X,var,config.FREQUENT_LABELS)

X=pf.encoding(X)

X=pf.scaling_featur_matix(X)

X_train, X_test, y_train, y_test=pf.divide_train_test(X,y)

X_train.to_csv('X_train.csv', index=False)

X_test.to_csv('X_test.csv', index=False)

y_train.to_csv('y_train.csv', index=False)

y_test.to_csv('y_test.csv', index=False)

pf.train_model(X_train,y_train,config.OUTPUT_MODEL_PATH)


print('Finished training')
