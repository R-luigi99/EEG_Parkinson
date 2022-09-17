import pandas as pd

test1 = pd.read_csv('NewMexON.csv', sep=",")
test2 = pd.read_csv('NewMexOFF.csv', sep=",")


merge=pd.merge(test1,test2, how='outer')


merge.to_csv('NewMexON_OFF.csv', index=False)



