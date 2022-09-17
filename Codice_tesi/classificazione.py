import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import BernoulliNB
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

NewMexOFF = pd.read_csv('NewMexOFF.csv', sep=",")
NewMexON_OFF = pd.read_csv('NewMexON_OFF.csv', sep=",")
NewMexON = pd.read_csv('NewMexON_54.csv', sep=',')

NewMexOFF_media=NewMexOFF[['AF3_media','AF4_media', 'AF7_media', 'AF8_media', 'AFz_media' ,'C1_media',  'C2_media', 'C3_media','C4_media','C5_media','C6_media','CP1_media','CP2_media'
     ,'CP3_media','CP4_media','CP5_media','CP6_media','Cz_media' ,'F1_media','F2_media','F3_media','F4_media','F5_media','F6_media' ,'F7_media','F8_media','FC1_media','FC2_media','FC3_media' ,'FC4_media','FC5_media','FC6_media','FCz_media','FT10_media',
    'FT7_media','FT8_media','FT9_media','Fp1_media','Fp2_media','Fz_media','O1_media','O2_media','Oz_media','P1_media','P2_media',
     'P3_media','P4_media','P5_media','P6_media','P7_media','P8_media','PO3_media','PO4_media','PO7_media','PO8_media','POz_media',
     'T7_media','T8_media','TP10_media','TP7_media','TP8_media','TP9_media','PD']]


NewMexOFF_varianza=NewMexOFF[['AF3_varianza','AF4_varianza', 'AF7_varianza', 'AF8_varianza', 'AFz_varianza' ,'C1_varianza',  'C2_varianza', 'C3_varianza','C4_varianza','C5_varianza','C6_varianza','CP1_varianza','CP2_varianza'
     ,'CP3_varianza','CP4_varianza','CP5_varianza','CP6_varianza','Cz_varianza' ,'F1_varianza','F2_varianza','F3_varianza','F4_varianza','F5_varianza','F6_varianza' ,'F7_varianza','F8_varianza','FC1_varianza','FC2_varianza','FC3_varianza' ,'FC4_varianza','FC5_varianza','FC6_varianza','FCz_varianza','FT10_varianza',
    'FT7_varianza','FT8_varianza','FT9_varianza','Fp1_varianza','Fp2_varianza','Fz_varianza','O1_varianza','O2_varianza','Oz_varianza','P1_varianza','P2_varianza',
     'P3_varianza','P4_varianza','P5_varianza','P6_varianza','P7_varianza','P8_varianza','PO3_varianza','PO4_varianza','PO7_varianza','PO8_varianza','POz_varianza',
     'T7_varianza','T8_varianza','TP10_varianza','TP7_varianza','TP8_varianza','TP9_varianza','PD']]


NewMexOFF_max_min=NewMexOFF[['AF3_max','AF3_min', 'AF4_max', 'AF4_min',  'AF7_max', 'AF7_min',  'AF8_max', 'AF8_min' ,
     'AFz_max','AFz_min','C1_max','C1_min', 'C2_max', 'C2_min','C3_max', 'C3_min',
    'C4_max','C4_min','C5_max','C5_min','C6_max','C6_min','CP1_max','CP1_min','CP2_max','CP2_min'
     ,'CP3_max','CP3_min','CP4_max','CP4_min','CP5_max','CP5_min','CP6_max','CP6_min','Cz_max','Cz_min'
     ,'F1_max','F1_min','F2_max','F2_min','F3_max','F3_min','F4_max','F4_min','F5_max','F5_min','F6_max','F6_min'
     ,'F7_max','F7_min','F8_max','F8_min','FC1_max','FC1_min','FC2_max','FC2_min','FC3_max','FC3_min'
     ,'FC4_max','FC4_min','FC5_max','FC5_min','FC6_max','FC6_min','FCz_max','FCz_min','FT10_max','FT10_min'
     ,'FT7_max','FT7_min','FT8_max','FT8_min','FT9_max','FT9_min','Fp1_max','Fp1_min','Fp2_max','Fp2_min',
     'Fz_max','Fz_min','O1_max','O1_min','O2_max','O2_min','Oz_max','Oz_min','P1_max','P1_min','P2_max','P2_min',
     'P3_max','P3_min','P4_max','P4_min','P5_max','P5_min','P6_max','P6_min','P7_max','P7_min','P8_max','P8_min'
     ,'PO3_max','PO3_min','PO4_max','PO4_min','PO7_max','PO7_min','PO8_max','PO8_min','POz_max','POz_min'
     ,'T7_max','T7_min','T8_max','T8_min','TP10_max','TP10_min','TP7_max','TP7_min','TP8_max','TP8_min'
     ,'TP9_max','TP9_min','PD']]

NewMexON_media=NewMexON[['AF3_media','AF4_media', 'AF7_media', 'AF8_media', 'AFz_media' ,'C1_media',  'C2_media', 'C3_media','C4_media','C5_media','C6_media','CP1_media','CP2_media'
     ,'CP3_media','CP4_media','CP5_media','CP6_media','Cz_media' ,'F1_media','F2_media','F3_media','F4_media','F5_media','F6_media' ,'F7_media','F8_media','FC1_media','FC2_media','FC3_media' ,'FC4_media','FC5_media','FC6_media','FCz_media','FT10_media',
    'FT7_media','FT8_media','FT9_media','Fp1_media','Fp2_media','Fz_media','O1_media','O2_media','Oz_media','P1_media','P2_media',
     'P3_media','P4_media','P5_media','P6_media','P7_media','P8_media','PO3_media','PO4_media','PO7_media','PO8_media','POz_media',
     'T7_media','T8_media','TP10_media','TP7_media','TP8_media','TP9_media','PD']]


NewMexON_varianza=NewMexON[['AF3_varianza','AF4_varianza', 'AF7_varianza', 'AF8_varianza', 'AFz_varianza' ,'C1_varianza',  'C2_varianza', 'C3_varianza','C4_varianza','C5_varianza','C6_varianza','CP1_varianza','CP2_varianza'
     ,'CP3_varianza','CP4_varianza','CP5_varianza','CP6_varianza','Cz_varianza' ,'F1_varianza','F2_varianza','F3_varianza','F4_varianza','F5_varianza','F6_varianza' ,'F7_varianza','F8_varianza','FC1_varianza','FC2_varianza','FC3_varianza' ,'FC4_varianza','FC5_varianza','FC6_varianza','FCz_varianza','FT10_varianza',
    'FT7_varianza','FT8_varianza','FT9_varianza','Fp1_varianza','Fp2_varianza','Fz_varianza','O1_varianza','O2_varianza','Oz_varianza','P1_varianza','P2_varianza',
     'P3_varianza','P4_varianza','P5_varianza','P6_varianza','P7_varianza','P8_varianza','PO3_varianza','PO4_varianza','PO7_varianza','PO8_varianza','POz_varianza',
     'T7_varianza','T8_varianza','TP10_varianza','TP7_varianza','TP8_varianza','TP9_varianza','PD']]


NewMexON_max_min=NewMexON[['AF3_max','AF3_min', 'AF4_max', 'AF4_min',  'AF7_max', 'AF7_min',  'AF8_max', 'AF8_min' ,
     'AFz_max','AFz_min','C1_max','C1_min', 'C2_max', 'C2_min','C3_max', 'C3_min',
    'C4_max','C4_min','C5_max','C5_min','C6_max','C6_min','CP1_max','CP1_min','CP2_max','CP2_min'
     ,'CP3_max','CP3_min','CP4_max','CP4_min','CP5_max','CP5_min','CP6_max','CP6_min','Cz_max','Cz_min'
     ,'F1_max','F1_min','F2_max','F2_min','F3_max','F3_min','F4_max','F4_min','F5_max','F5_min','F6_max','F6_min'
     ,'F7_max','F7_min','F8_max','F8_min','FC1_max','FC1_min','FC2_max','FC2_min','FC3_max','FC3_min'
     ,'FC4_max','FC4_min','FC5_max','FC5_min','FC6_max','FC6_min','FCz_max','FCz_min','FT10_max','FT10_min'
     ,'FT7_max','FT7_min','FT8_max','FT8_min','FT9_max','FT9_min','Fp1_max','Fp1_min','Fp2_max','Fp2_min',
     'Fz_max','Fz_min','O1_max','O1_min','O2_max','O2_min','Oz_max','Oz_min','P1_max','P1_min','P2_max','P2_min',
     'P3_max','P3_min','P4_max','P4_min','P5_max','P5_min','P6_max','P6_min','P7_max','P7_min','P8_max','P8_min'
     ,'PO3_max','PO3_min','PO4_max','PO4_min','PO7_max','PO7_min','PO8_max','PO8_min','POz_max','POz_min'
     ,'T7_max','T7_min','T8_max','T8_min','TP10_max','TP10_min','TP7_max','TP7_min','TP8_max','TP8_min'
     ,'TP9_max','TP9_min','PD']]

NewMexON_OFF_media=NewMexON_OFF[['AF3_media','AF4_media', 'AF7_media', 'AF8_media', 'AFz_media' ,'C1_media',  'C2_media', 'C3_media','C4_media','C5_media','C6_media','CP1_media','CP2_media'
     ,'CP3_media','CP4_media','CP5_media','CP6_media','Cz_media' ,'F1_media','F2_media','F3_media','F4_media','F5_media','F6_media' ,'F7_media','F8_media','FC1_media','FC2_media','FC3_media' ,'FC4_media','FC5_media','FC6_media','FCz_media','FT10_media',
    'FT7_media','FT8_media','FT9_media','Fp1_media','Fp2_media','Fz_media','O1_media','O2_media','Oz_media','P1_media','P2_media',
     'P3_media','P4_media','P5_media','P6_media','P7_media','P8_media','PO3_media','PO4_media','PO7_media','PO8_media','POz_media',
     'T7_media','T8_media','TP10_media','TP7_media','TP8_media','TP9_media','Levodopa','PD']]


NewMexON_OFF_varianza=NewMexON_OFF[['AF3_varianza','AF4_varianza', 'AF7_varianza', 'AF8_varianza', 'AFz_varianza' ,'C1_varianza',  'C2_varianza', 'C3_varianza','C4_varianza','C5_varianza','C6_varianza','CP1_varianza','CP2_varianza'
     ,'CP3_varianza','CP4_varianza','CP5_varianza','CP6_varianza','Cz_varianza' ,'F1_varianza','F2_varianza','F3_varianza','F4_varianza','F5_varianza','F6_varianza' ,'F7_varianza','F8_varianza','FC1_varianza','FC2_varianza','FC3_varianza' ,'FC4_varianza','FC5_varianza','FC6_varianza','FCz_varianza','FT10_varianza',
    'FT7_varianza','FT8_varianza','FT9_varianza','Fp1_varianza','Fp2_varianza','Fz_varianza','O1_varianza','O2_varianza','Oz_varianza','P1_varianza','P2_varianza',
     'P3_varianza','P4_varianza','P5_varianza','P6_varianza','P7_varianza','P8_varianza','PO3_varianza','PO4_varianza','PO7_varianza','PO8_varianza','POz_varianza',
     'T7_varianza','T8_varianza','TP10_varianza','TP7_varianza','TP8_varianza','TP9_varianza','Levodopa','PD']]


NewMexON_OFF_max_min=NewMexON_OFF[['AF3_max','AF3_min', 'AF4_max', 'AF4_min',  'AF7_max', 'AF7_min',  'AF8_max', 'AF8_min' ,
     'AFz_max','AFz_min','C1_max','C1_min', 'C2_max', 'C2_min','C3_max', 'C3_min',
    'C4_max','C4_min','C5_max','C5_min','C6_max','C6_min','CP1_max','CP1_min','CP2_max','CP2_min'
     ,'CP3_max','CP3_min','CP4_max','CP4_min','CP5_max','CP5_min','CP6_max','CP6_min','Cz_max','Cz_min'
     ,'F1_max','F1_min','F2_max','F2_min','F3_max','F3_min','F4_max','F4_min','F5_max','F5_min','F6_max','F6_min'
     ,'F7_max','F7_min','F8_max','F8_min','FC1_max','FC1_min','FC2_max','FC2_min','FC3_max','FC3_min'
     ,'FC4_max','FC4_min','FC5_max','FC5_min','FC6_max','FC6_min','FCz_max','FCz_min','FT10_max','FT10_min'
     ,'FT7_max','FT7_min','FT8_max','FT8_min','FT9_max','FT9_min','Fp1_max','Fp1_min','Fp2_max','Fp2_min',
     'Fz_max','Fz_min','O1_max','O1_min','O2_max','O2_min','Oz_max','Oz_min','P1_max','P1_min','P2_max','P2_min',
     'P3_max','P3_min','P4_max','P4_min','P5_max','P5_min','P6_max','P6_min','P7_max','P7_min','P8_max','P8_min'
     ,'PO3_max','PO3_min','PO4_max','PO4_min','PO7_max','PO7_min','PO8_max','PO8_min','POz_max','POz_min'
     ,'T7_max','T7_min','T8_max','T8_min','TP10_max','TP10_min','TP7_max','TP7_min','TP8_max','TP8_min'
     ,'TP9_max','TP9_min','Levodopa','PD']]





pd.set_option("display.max_rows", None, "display.max_columns", None)

dataset=[NewMexOFF_media,NewMexOFF_varianza,NewMexOFF_max_min,NewMexON_media,NewMexON_varianza,NewMexON_max_min,
          NewMexON_OFF_media,NewMexON_OFF_varianza,NewMexON_OFF_max_min]

names=['NewMexOFF_media','NewMexOFF_varianza','NewMexOFF_max_min','NewMexON_media','NewMexON_varianza','NewMexON_max_min',
         ' NewMexON_OFF_media','NewMexON_OFF_varianza','NewMexON_OFF_max_min']

for name, df in zip(names,dataset):
    print('__________________________________________________')
    print(name)
    print('__________________________________________________')
    def classificationFunction(df):
        car=df.shape[1] - 1
        numpy_array = df.values
        x = numpy_array[:, : car ]
        y = numpy_array[:, car]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        names = ["RandomForrest", "ExtraTrees", "MLPClassifier", "XGBClassifier", "DecisionTree", "SVC",
                 "Naive Bayes", "K-Nearest Neighbor", "CatBoostClassifier", "QDA", "AdaBoostClassifier",
                 "GradientBoostingClassifier"]

        classifiers = [
            RandomForestClassifier(class_weight='balanced', random_state=1, n_estimators=100, max_features=1),
            ExtraTreesClassifier(class_weight='balanced', random_state=1),
            MLPClassifier(hidden_layer_sizes=(7, 5, 3, 2), random_state=1),
            XGBClassifier(learning_rate=0.01, use_label_encoder=False, eval_metric= 'logloss'),
            DecisionTreeClassifier(class_weight='balanced', max_depth=10),
            SVC(gamma=2, C=1),
            GaussianNB(),
            KNeighborsClassifier(3),
            CatBoostClassifier(max_depth=5, verbose=False),
            QuadraticDiscriminantAnalysis(),
            AdaBoostClassifier(),
            GradientBoostingClassifier(n_estimators=20, learning_rate=0.01, max_features=0.7, max_depth=2,
                                       random_state=0)]

        result = pd.DataFrame(
            columns=["Classifier", "Accuracy", "Precision", "Recall", "FScore",
                     "K-Cross Validation best result Accuracy",
                     "K-Cross Validation best result Precision", "K-Cross Validation best result Recall",
                     "K-Cross Validation best result FScore"])

        for name, classifier in zip(names, classifiers):
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)
            pr, rc, fs, sup = metrics.precision_recall_fscore_support(y_test, y_pred, average='macro')
            report = classification_report(y_pred, y_test)
            disp = plot_confusion_matrix(classifier, x_test, y_test, cmap=plt.cm.BuPu, normalize=None)
            disp.ax_.set_title("Confusion matrix for " + name)
            print()
            print(name, "\n")
            print(report)
            print(name, "Confusion Matrix: \n", disp.confusion_matrix)
            print()
            print('Accuracy score: ', round(accuracy_score(y_test, y_pred), 3), ' Precision', round(pr, 3), ' Recall',
                  round(rc, 3), ' FScore', round(fs, 3))
            cvA = cross_val_score(classifier, x_test, y_test, cv=10, scoring='accuracy')
            cvP = cross_val_score(classifier, x_test, y_test, cv=10, scoring='precision_macro')
            cvR = cross_val_score(classifier, x_test, y_test, cv=10, scoring='recall_macro')
            cvF = cross_val_score(classifier, x_test, y_test, cv=10, scoring='f1_macro')
            print("\n", name, "K cross validation: ")
            print("Score Accuracy: ", cvA)
            print("Mean Accuracy: ", round(np.mean(cvA),3))
            print("Score Precision: ", cvP)
            print("Mean Precision: ", round(np.mean(cvP),3))
            print("Score Recall: ", cvR)
            print("Mean Recall: ", round(np.mean(cvR),3))
            print("Score FScore: ", cvF)
            print("Mean FScore: ", round(np.mean(cvF),3))
            print("\n\n")
            result = result.append({"Classifier": name, "Accuracy": round(metrics.accuracy_score(y_test, y_pred), 4),
                                    "Precision": round(pr, 3), "Recall": round(rc, 3), "FScore": round(fs, 3),
                                    "K-Cross Validation best result Accuracy": round(max(cvA), 3),
                                    "K-Cross Validation best result Precision": round(max(cvP), 3),
                                    "K-Cross Validation best result Recall": round(max(cvR), 3),
                                    "K-Cross Validation best result FScore": round(max(cvF), 3)
                                    }, ignore_index=True)

        print(result)
        return result


    classificationFunction(df)

    # Cross-validate
    def display_accuracy_scores(pipeline, x, y):
        scores = cross_val_score(pipeline,x, y, cv=10, scoring='accuracy')
        print('Scores\t:', scores)
        print('Mean\t:', scores.mean())
        print('SD\t:', scores.std())

    print("###########################################################################")