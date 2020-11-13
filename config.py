# ====   PATHS ===================

PATH_TO_DATASET = 'titanic_survival.csv'
OUTPUT_SCALER_PATH = 'scaler.pkl'
OUTPUT_MODEL_PATH = 'logistic_regression.pkl'
X_TRAIN="X_train.csv"
X_TEST="X_test.csv"
Y_TRAIN="y_train.csv"
Y_TEST="y_test.csv"
#===========Feature groups=========#

TARGET='survived'

#categorical variable to get cabin details only
CABIN=['cabin']

#variable to get the title details
NAME=['name']

#numerical columns to be converted to float
NUMERICAL_TO_FLOAT=['age','fare']

#columns to be dropped
DROP_LABELS=['name','ticket', 'boat', 'body','home.dest']

#list of numerical values to be processed
VARS_NUM=['pclass','age', 'sibsp', 'parch', 'fare']

#list of selected categorial columns
VARS_CAT=['sex', 'cabin', 'embarked', 'title']

#numerical columns with Nan
VARS_NUM_MS=['age', 'fare']

#missing cat drop_columns
VARS_CAT_MS=['cabin', 'embarked']

#frequent labels and colums to be changed to roc_auc_score
FREQUENT_LABELS={
    'sex':['female', 'male'],
    'cabin':['A', 'B', 'C', 'D', 'E', 'F', 'G', 'Missing', 'T'],
    'embarked':['C', 'Missing', 'Q', 'S'],
    'title':['Master', 'Miss', 'Mr', 'Mrs', 'Other']
}

COLUMN_NAMES_AFTER_DUMMIES=['pclass', 'survived', 'age', 'sibsp', 'parch', 'fare', 'sex_male',
       'cabin_B', 'cabin_C', 'cabin_D', 'cabin_E', 'cabin_F', 'cabin_G',
       'cabin_Missing', 'cabin_T', 'embarked_Missing', 'embarked_Q',
       'embarked_S', 'title_Miss', 'title_Mr', 'title_Mrs', 'title_Other']
