import kmapper as km
import pandas as pd
import numpy as np
import os
import sklearn
from sklearn import datasets

# Set path and obtain files
PATH = "C:/Users/Gebruiker/Documents/Programming/Python/Elections"
os.chdir(PATH)
files = os.listdir('./data_election')
#files = ['Uitslag_2002.csv']
for f in files:
    # Get file name and year
    name, _ = os.path.splitext(f)
    _, year = name.split('_') 

    # Data read
    df = pd.read_csv('./data_election/Uitslag_{}.csv'.format(year), 
                     sep=';', 
                     index_col=0)
    dg = pd.read_csv('./data_population/Populatie_{}.csv'.format(year),
                     sep=';',
                     index_col=0)
    difference = set(df.index).symmetric_difference(set(dg.index))
    print(difference)
    #print('the year',year)
    #print(dg.aantal.shape)
    #print(df.index.shape)
    #print(difference)
