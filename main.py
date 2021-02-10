from utilities import *
from loader import Loader
from layers import GraphConv
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


dataset = 'cora'
learning_rate = 1e-2
seed = 0
epochs = 200
patience = 50
channels = 16
dropout = 0.2

feature_matrix, list_edge = read_data(dataset)
feat_df, list_edge_df, label = preprocess_data(feature_matrix, list_edge)
adjacency_matrix = create_adjacency_matrix(list_edge_df)
adjacency_matrix = preprocess_adjacency_matrix(adjacency_matrix)
mask_tr, mask_va, mask_te = masking(feature_matrix=feat_df, num_train=140, num_valid=500, num_test=1000)
N = feat_df.shape[0]         # Number of nodes in the graph
F = feat_df.shape[1]   # Original size of node features
n_out = label.shape[1]
print(N, F, n_out)


x_in = Input(shape=(F,))
a_in = Input((N,))

gc_1 = GraphConv(channels,
               activation='relu')([x_in, a_in])
do_2 = Dropout(dropout)(gc_1)
gc_2 = GraphConv(n_out,
               activation='softmax')([do_2, a_in])
model = tf.keras.models.Model([x_in,a_in], gc_2)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()

loader_tr = Loader(feature_matrix=feat_df, adjacency_matrix=adjacency_matrix,label=label,sample_weights=mask_tr)
loader_va = Loader(feature_matrix=feat_df, adjacency_matrix=adjacency_matrix,label=label,sample_weights=mask_va)
model.fit(loader_tr.load(),
          steps_per_epoch=1,
          validation_data=loader_va.load(),
          validation_steps=1,
          epochs=epochs,
          callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)])
# print(loader.load())