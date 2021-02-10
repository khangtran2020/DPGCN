import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
import networkx as nx

def read_data(dataset, prefix='data'):
    datapath = os.path.join(prefix, dataset)
    feature_matrix = np.loadtxt(datapath + '/' + dataset + '.content', dtype=str, delimiter='\t')
    list_edge = np.loadtxt(datapath + '/' + dataset + '.cites', dtype=str, delimiter='\t')
    return feature_matrix, list_edge

def preprocess_data(feat_matrix, ls_edge, reindex = True):
    le = preprocessing.LabelEncoder()
    feat_df = pd.DataFrame(feat_matrix)
    ls_edge_df = pd.DataFrame(ls_edge)
    idx = feat_df.columns[0]
    label = feat_df.columns[-1]
    feat_df['id'] = feat_df[idx]
    feat_df['label'] = feat_df[label]
    feat_df.drop([idx,label], axis=1, inplace = True)
    le.fit(feat_df['id'])
    feat_df['id'] = le.transform(feat_df['id'])
    feat_df.sort_values('id', ascending=True, inplace=True)
    feat_df.index = feat_df['id']
    feat_df.drop('id',axis=1,inplace=True)
    for col in ls_edge_df.columns:
        ls_edge_df[col] = le.transform(ls_edge_df[col])
    label = le.fit_transform(feat_df['label'])
    label = tf.keras.utils.to_categorical(label)
    feat_df.drop('label', axis=1,inplace=True)
    feat_df = feat_df.astype(int)
    return feat_df, ls_edge_df, label

def create_adjacency_matrix(list_edge_df):
    G=nx.from_pandas_edgelist(list_edge_df, list_edge_df.columns[0], list_edge_df.columns[-1])
    return nx.adjacency_matrix(G).todense()

def preprocess_adjacency_matrix(adjacency_matrix):
    # take degree matrix
    adjacency_matrix = adjacency_matrix + np.identity(adjacency_matrix.shape[0])
    degree = []
    for val in np.sum(adjacency_matrix > 0, axis = 1):
        degree.append(val[0,0])
    degree_matrix = 1/np.sqrt(np.diag(degree))
    degree_matrix[degree_matrix==np.inf] = 0
    return degree_matrix*adjacency_matrix*degree_matrix