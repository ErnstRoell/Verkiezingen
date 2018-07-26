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
#files = ['Uitslag_2017.csv']
for f in files:
    # Get file name
    name, _ = os.path.splitext(f)
    _, year = name.split('_') 

    # Data read
    df = pd.read_csv('./data_election/{}'.format(f), sep=';', index_col=0)
    df.sort_index(inplace=True)

    data = df.as_matrix()
    labels = df.index

    # Initialize
    mapper = km.KeplerMapper()

    projected_data = mapper.fit_transform(data,
                                          projection=sklearn.manifold.TSNE()
                                          )

    # Create dictionary called 'graph' with nodes, edges and meta-information
    graph = mapper.map(projected_data,
                       overlap_perc=.5,
                       nr_cubes=20)

    dg = pd.read_csv('./data_population/Populatie_{}.csv'.format(year),sep=';', index_col=0)
    dg.sort_index(inplace=True)

    weight = np.zeros(df.shape[1])
    w_parties = np.array([.75,-.5,-.125,.5,-1,0,-.5-.5,.25,-.5])
    weight[:9]=w_parties
    result = np.dot(mapper.inverse,weight)

    result = np.interp(result,(result.min(),result.max()),(-1,1))
    def color(ids):
        val = np.average(result[ids])
        return int(30*np.interp(val,(result.min(),result.max()),(0,1)))

    def size(ids):
        return np.sum(dg.aantal[ids])

    def type(ids):
        return "circle"

    # Visualize it
    mapper.visualize(graph, 
                     path_html="./html/{}.html".format(name),
                     X=mapper.inverse,
                     color_function=color,
                     size_function=size,
                     type_function=type,
                     custom_tooltips=labels
                     )
