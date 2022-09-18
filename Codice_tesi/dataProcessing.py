import copy
import pickle
import numpy
from pymatreader import read_mat


NewMex_OFF= read_mat('EEG_Jim_rest_OFF_Unsegmented_WithAllChannels.mat')
NewMex_ON= read_mat('EEG_Jim_rest_Unsegmented_WithAllChannelsEYESCLOSED.mat')


#selezione dei canali più comuni

Pz=NewMex_OFF['Channel_location'].index('Pz')
NewMex_OFF['Channel_location'].pop(NewMex_OFF['Channel_location'].index('Pz'))
NewMex_OFF['EEG'].pop(Pz)

VEOG=NewMex_OFF['Channel_location'].index('VEOG')
NewMex_OFF['Channel_location'].pop(NewMex_OFF['Channel_location'].index('VEOG'))
NewMex_OFF['EEG'].pop(VEOG)

X=NewMex_OFF['Channel_location'].index('X')
NewMex_OFF['Channel_location'].pop(NewMex_OFF['Channel_location'].index('X'))
NewMex_OFF['EEG'].pop(X)

Y=NewMex_OFF['Channel_location'].index('Y')
NewMex_OFF['Channel_location'].pop(NewMex_OFF['Channel_location'].index('Y'))
NewMex_OFF['EEG'].pop(Y)

Z=NewMex_OFF['Channel_location'].index('Z')
NewMex_OFF['Channel_location'].pop(NewMex_OFF['Channel_location'].index('Z'))
NewMex_OFF['EEG'].pop(Z)


Pz=NewMex_ON['Channel_location'].index('Pz')
NewMex_ON['Channel_location'].pop(NewMex_ON['Channel_location'].index('Pz'))
NewMex_ON['EEG'].pop(Pz)

VEOG=NewMex_ON['Channel_location'].index('VEOG')
NewMex_ON['Channel_location'].pop(NewMex_ON['Channel_location'].index('VEOG'))
NewMex_ON['EEG'].pop(VEOG)

X=NewMex_ON['Channel_location'].index('X')
NewMex_ON['Channel_location'].pop(NewMex_ON['Channel_location'].index('X'))
NewMex_ON['EEG'].pop(X)

Y=NewMex_ON['Channel_location'].index('Y')
NewMex_ON['Channel_location'].pop(NewMex_ON['Channel_location'].index('Y'))
NewMex_ON['EEG'].pop(Y)

Z=NewMex_ON['Channel_location'].index('Z')
NewMex_ON['Channel_location'].pop(NewMex_ON['Channel_location'].index('Z'))
NewMex_ON['EEG'].pop(Z)



#ordino i canali
size=len(NewMex_OFF['Channel_location'])
for step in range(size):
    min_idx = step

    for i in range(step + 1, size):


        if NewMex_OFF['Channel_location'][i] < NewMex_OFF['Channel_location'][min_idx]:
            min_idx = i

    # put min at the correct position
    (NewMex_OFF['Channel_location'][step], NewMex_OFF['Channel_location'][min_idx]) = (NewMex_OFF['Channel_location'][min_idx],NewMex_OFF['Channel_location'][step])
    (NewMex_OFF['EEG'][step], NewMex_OFF['EEG'][min_idx]) = (NewMex_OFF['EEG'][min_idx],NewMex_OFF['EEG'][step])


size=len(NewMex_ON['Channel_location'])
for step in range(size):
    min_idx = step

    for i in range(step + 1, size):


        if NewMex_ON['Channel_location'][i] < NewMex_ON['Channel_location'][min_idx]:
            min_idx = i

    # put min at the correct position
    (NewMex_ON['Channel_location'][step], NewMex_ON['Channel_location'][min_idx]) = (NewMex_ON['Channel_location'][min_idx],NewMex_ON['Channel_location'][step])
    (NewMex_ON['EEG'][step], NewMex_ON['EEG'][min_idx]) = (NewMex_ON['EEG'][min_idx], NewMex_ON['EEG'][step])


#scambio ordine 0 e 1 per PD e controllo

for j in range(62):
    (NewMex_ON['EEG'][j][0], NewMex_ON['EEG'][j][1]) = (NewMex_ON['EEG'][j][1], NewMex_ON['EEG'][j][0])

for j in range(62):
    (NewMex_OFF['EEG'][j][0], NewMex_OFF['EEG'][j][1]) = (NewMex_OFF['EEG'][j][1], NewMex_OFF['EEG'][j][0])



NewMex_ON_media=NewMex_ON
NewMex_OFF_media=NewMex_OFF
NewMex_OFF_varianza=copy.deepcopy(NewMex_OFF)
NewMex_OFF_min=copy.deepcopy(NewMex_OFF)
NewMex_OFF_max=copy.deepcopy(NewMex_OFF)
NewMex_ON_varianza=copy.deepcopy(NewMex_ON)
NewMex_ON_min=copy.deepcopy(NewMex_ON)
NewMex_ON_max=copy.deepcopy(NewMex_ON)


#media dei valori

for j in range(62):
    for i in range(27):
        NewMex_OFF_media['EEG'][j][0][i]=numpy.mean(NewMex_OFF_media['EEG'][j][0][i])
        NewMex_OFF_media['EEG'][j][1][i]=numpy.mean(NewMex_OFF_media['EEG'][j][1][i])

for j in range(62):
    for i in range(27):
        NewMex_ON_media['EEG'][j][0][i]=numpy.mean(NewMex_ON_media['EEG'][j][0][i])
        NewMex_ON_media['EEG'][j][1][i]=numpy.mean(NewMex_ON_media['EEG'][j][1][i])

for j in range(62):
    for i in range(27):
        NewMex_OFF_varianza['EEG'][j][0][i]=numpy.var(NewMex_OFF_varianza['EEG'][j][0][i])
        NewMex_OFF_varianza['EEG'][j][1][i]=numpy.var(NewMex_OFF_varianza['EEG'][j][1][i])

for j in range(62):
    for i in range(27):
        NewMex_ON_varianza['EEG'][j][0][i]=numpy.var(NewMex_ON_varianza['EEG'][j][0][i])
        NewMex_ON_varianza['EEG'][j][1][i]=numpy.var(NewMex_ON_varianza['EEG'][j][1][i])

for j in range(62):
    for i in range(27):
        NewMex_OFF_max['EEG'][j][0][i]=numpy.max(NewMex_OFF_max['EEG'][j][0][i])
        NewMex_OFF_max['EEG'][j][1][i]=numpy.max(NewMex_OFF_max['EEG'][j][1][i])

for j in range(62):
    for i in range(27):
        NewMex_ON_max['EEG'][j][0][i]=numpy.max(NewMex_ON_max['EEG'][j][0][i])
        NewMex_ON_max['EEG'][j][1][i]=numpy.max(NewMex_ON_max['EEG'][j][1][i])


for j in range(62):
    for i in range(27):
        NewMex_OFF_min['EEG'][j][0][i]=numpy.min(NewMex_OFF_min['EEG'][j][0][i])
        NewMex_OFF_min['EEG'][j][1][i]=numpy.min(NewMex_OFF_min['EEG'][j][1][i])

for j in range(62):
    for i in range(27):
        NewMex_ON_min['EEG'][j][0][i]=numpy.min(NewMex_ON_min['EEG'][j][0][i])
        NewMex_ON_min['EEG'][j][1][i]=numpy.min(NewMex_ON_min['EEG'][j][1][i])


file = open("NewMex_OFF_media.pkl","wb")
pickle.dump(NewMex_OFF_media, file)
file.close()

file = open("NewMex_ON_media.pkl","wb")
pickle.dump(NewMex_ON_media, file)
file.close()


file = open("NewMex_OFF_varianza.pkl","wb")
pickle.dump(NewMex_OFF_varianza, file)
file.close()


file = open("NewMex_ON_varianza.pkl","wb")
pickle.dump(NewMex_ON_varianza, file)
file.close()


file = open("NewMex_OFF_max.pkl","wb")
pickle.dump(NewMex_OFF_max, file)
file.close()


file = open("NewMex_ON_max.pkl","wb")
pickle.dump(NewMex_ON_max, file)
file.close()


file = open("NewMex_OFF_min.pkl","wb")
pickle.dump(NewMex_OFF_min, file)
file.close()


file = open("NewMex_ON_min.pkl","wb")
pickle.dump(NewMex_ON_min, file)
file.close()
