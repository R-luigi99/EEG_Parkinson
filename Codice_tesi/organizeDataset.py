import pickle
import csv


file = open("NewMex_OFF_media.pkl", "rb")
NewMex_OFF_media = pickle.load(file)
file = open("NewMex_ON_media.pkl", "rb") #off
NewMex_ON_media = pickle.load(file)
file = open("NewMex_OFF_varianza.pkl", "rb")
NewMex_OFF_varianza = pickle.load(file)
file = open("NewMex_ON_varianza.pkl", "rb")
NewMex_ON_varianza = pickle.load(file)
file = open("NewMex_OFF_max.pkl", "rb")
NewMex_OFF_max = pickle.load(file)
file = open("NewMex_ON_max.pkl", "rb")
NewMex_ON_max = pickle.load(file)
file = open("NewMex_OFF_min.pkl", "rb")
NewMex_OFF_min = pickle.load(file)
file = open("NewMex_ON_min.pkl", "rb")
NewMex_ON_min = pickle.load(file)


keys=['AF3_media','AF3_varianza','AF3_max','AF3_min','AF4_media','AF4_varianza', 'AF4_max', 'AF4_min', 'AF7_media','AF7_varianza', 'AF7_max', 'AF7_min', 'AF8_media', 'AF8_varianza', 'AF8_max', 'AF8_min' ,
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
     ,'TP9_media','TP9_varianza','TP9_max','TP9_min','Levodopa','PD']

NewMex_ON_DF = {key: [] for key in keys}

#costruisco iterativamente il dataset

for j in range(62):
    for i in range(27):
        NewMex_ON_DF[NewMex_ON_media['Channel_location'][j] + '_media'].append(NewMex_ON_media['EEG'][j][1][i])
        NewMex_ON_DF[NewMex_ON_varianza['Channel_location'][j] + '_varianza'].append(NewMex_ON_varianza['EEG'][j][1][i])
        NewMex_ON_DF[NewMex_ON_max['Channel_location'][j] + '_max'].append(NewMex_ON_max['EEG'][j][1][i])
        NewMex_ON_DF[NewMex_ON_min['Channel_location'][j] + '_min'].append(NewMex_ON_min['EEG'][j][1][i])
        NewMex_ON_DF['PD'].append(1)
        NewMex_ON_DF['Levodopa'].append(1)

NewMex_ON_DF['PD'][27:] = []

NewMex_OFF_DF = {key: [] for key in keys}

for j in range(62):
    for i in range(27):
        NewMex_OFF_DF[NewMex_OFF_media['Channel_location'][j] + '_media'].append(NewMex_OFF_media['EEG'][j][0][i])
        NewMex_OFF_DF[NewMex_OFF_media['Channel_location'][j] + '_media'].append(NewMex_OFF_media['EEG'][j][1][i])
        NewMex_OFF_DF[NewMex_OFF_varianza['Channel_location'][j] + '_varianza'].append(NewMex_OFF_varianza['EEG'][j][0][i])
        NewMex_OFF_DF[NewMex_OFF_varianza['Channel_location'][j] + '_varianza'].append(NewMex_OFF_varianza['EEG'][j][1][i])
        NewMex_OFF_DF[NewMex_OFF_max['Channel_location'][j] + '_max'].append(NewMex_OFF_max['EEG'][j][0][i])
        NewMex_OFF_DF[NewMex_OFF_max['Channel_location'][j] + '_max'].append(NewMex_OFF_max['EEG'][j][1][i])
        NewMex_OFF_DF[NewMex_OFF_min['Channel_location'][j] + '_min'].append(NewMex_OFF_min['EEG'][j][0][i])
        NewMex_OFF_DF[NewMex_OFF_min['Channel_location'][j] + '_min'].append(NewMex_OFF_min['EEG'][j][1][i])
        NewMex_OFF_DF['PD'].append(0)
        NewMex_OFF_DF['PD'].append(1)
        NewMex_OFF_DF['Levodopa'].append(0)
        NewMex_OFF_DF['Levodopa'].append(0)

NewMex_OFF_DF['PD'][54:] = []

with open("NewMexON.csv", "w") as outfile2:

    writer2 = csv.writer(outfile2)

    writer2.writerow(NewMex_ON_DF.keys())

    writer2.writerows(zip(*NewMex_ON_DF.values()))



with open("NewMexOFF.csv", "w") as outfile:

    writer = csv.writer(outfile)

    writer.writerow(NewMex_OFF_DF.keys())

    writer.writerows(zip(*NewMex_OFF_DF.values()))


