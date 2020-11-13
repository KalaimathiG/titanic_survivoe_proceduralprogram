
import preprocessing as pf
import config

if __name__ == '__main__':


    import numpy as np
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.metrics import confusion_matrix
    import warnings
    warnings.simplefilter(action='ignore')

    # Load data
    X_test = pf.load_data(config.X_TEST)
    y_test=pf.load_data(config.Y_TEST)
    y_pred=pf.predict(X_test,config.OUTPUT_MODEL_PATH)

    #confusion_matrix = confusion_matrix(y_test, y_pred)
    #print('Confusion_matrix=',confusion_matrix)

    logit_roc_auc = roc_auc_score(y_test, y_pred)
    print('roc_auc_score=',logit_roc_auc)

    print('accuracy_score=',accuracy_score(y_test,y_pred))
