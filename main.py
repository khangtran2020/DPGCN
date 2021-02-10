from utilities import *
from loader import Loader


dataset = 'cora'
feature_matrix, list_edge = read_data(dataset)
feat_df, list_edge_df, label = preprocess_data(feature_matrix, list_edge)
adjacency_matrix = create_adjacency_matrix(list_edge_df)
adjacency_matrix = preprocess_adjacency_matrix(adjacency_matrix)
# print(feat_df)
# print(adjacency_matrix)
# print(label)
mask_tr, mask_va, mask_te = masking(feature_matrix=feat_df, num_train=140, num_valid=500, num_test=1000)
# print(np.sum(mask_tr))
# print(np.sum(mask_va))
# print(np.sum(mask_te))


loader = Loader(feature_matrix=feat_df, adjacency_matrix=adjacency_matrix,label=label,sample_weights=mask_tr)

print(loader.load())