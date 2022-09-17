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

NewMexON_OFF = pd.read_csv('NewMexON_OFF.csv', sep=",")






NewMexON_OFF=NewMexON_OFF[['AF3_media','AF3_varianza','AF3_max','AF3_min','AF4_media','AF4_varianza', 'AF4_max', 'AF4_min', 'AF7_media','AF7_varianza', 'AF7_max', 'AF7_min', 'AF8_media', 'AF8_varianza', 'AF8_max', 'AF8_min' ,
     'AFz_media' ,'AFz_varianza','AFz_max','AFz_min', 'C1_media', 'C1_varianza','C1_max','C1_min', 'C2_media', 'C2_varianza', 'C2_max', 'C2_min', 'C3_media', 'C3_varianza', 'C3_max', 'C3_min'
     ,'C4_media','C4_varianza','C4_max','C4_min','C5_media','C5_varianza','C5_max','C5_min','C6_media','C6_varianza','C6_max','C6_min','CP1_media','CP1_varianza','CP1_max','CP1_min','CP2_media','CP2_varianza','CP2_max','CP2_min'
     ,'CP3_media','CP3_varianza','CP3_max','CP3_min','CP4_media','CP4_varianza','CP4_max','CP4_min','CP5_media','CP5_varianza','CP5_max','CP5_min','CP6_media','CP6_varianza','CP6_max','CP6_min','Cz_media','Cz_varianza','Cz_max','Cz_min'
     ,'F1_media','F1_varianza','F1_max','F1_min','F2_media','F2_varianza','F2_max','F2_min','F3_media','F3_varianza','F3_max','F3_min','F4_media','F4_varianza','F4_max','F4_min','F5_media','F5_varianza','F5_max','F5_min','F6_media','F6_varianza','F6_max','F6_min'
     ,'F7_media','F7_varianza','F7_max','F7_min','F8_media','F8_varianza','F8_max','F8_min','FC1_media','FC1_varianza','FC1_max','FC1_min','FC2_media','FC2_varianza','FC2_max','FC2_min','FC3_media','FC3_varianza','FC3_max','FC3_min'
     ,'FC4_media','FC4_varianza','FC4_max','FC4_min','FC5_media','FC5_varianza','FC5_max','FC5_min','FC6_media','FC6_varianza','FC6_max','FC6_min','FCz_media','FCz_varianza','FCz_max','FCz_min','FT10_media','FT10_varianza','FT10_max','FT10_min'
     ,'FT7_media','FT7_varianza','FT7_max','FT7_min','FT8_media','FT8_varianza','FT8_max','FT8_min','FT9_media','FT9_varianza','FT9_max','FT9_min','Fp1_media','Fp1_varianza','Fp1_max','Fp1_min','Fp2_media','Fp2_varianza','Fp2_max','Fp2_min'
     ,'Fz_media','Fz_varianza','Fz_max','Fz_min','O1_media','O1_varianza','O1_max','O1_min','O2_media','O2_varianza','O2_max','O2_min','Oz_media','Oz_varianza','Oz_max','Oz_min','P1_media','P1_varianza','P1_max','P1_min','P2_media','P2_varianza','P2_max','P2_min'
     ,'P3_media','P3_varianza','P3_max','P3_min','P4_media','P4_varianza','P4_max','P4_min','P5_media','P5_varianza','P5_max','P5_min','P6_media','P6_varianza','P6_max','P6_min','P7_media','P7_varianza','P7_max','P7_min','P8_media','P8_varianza','P8_max','P8_min'
     ,'PO3_media','PO3_varianza','PO3_max','PO3_min','PO4_media','PO4_varianza','PO4_max','PO4_min','PO7_media','PO7_varianza','PO7_max','PO7_min','PO8_media','PO8_varianza','PO8_max','PO8_min','POz_media','POz_varianza','POz_max','POz_min'
     ,'T7_media','T7_varianza','T7_max','T7_min','T8_media','T8_varianza','T8_max','T8_min','TP10_media','TP10_varianza','TP10_max','TP10_min','TP7_media','TP7_varianza','TP7_max','TP7_min','TP8_media','TP8_varianza','TP8_max','TP8_min'
     ,'TP9_media','TP9_varianza','TP9_max','TP9_min','Levodopa','PD']]



dataset=[NewMexON_OFF]
names=['NewMexON_OFF']


pd.set_option("display.max_rows", None, "display.max_columns", None)
dataset=[NewMexON_OFF]

names=['NewMexON_OFF']

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