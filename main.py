from utilities import *

dataset = 'cora'
feature_matrix, list_edge = read_data(dataset)
feat_df, list_edge_df, label = preprocess_data(feature_matrix, list_edge)
adjacency_matrix = create_adjacency_matrix(list_edge_df)
adjacency_matrix = preprocess_adjacency_matrix(adjacency_matrix)
print(feat_df.values)
print(adjacency_matrix)
print(label)