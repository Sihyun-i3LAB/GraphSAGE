import numpy as np
import tensorflow as tf
from tensorflow import keras
import networkx as nx
import pandas as pd
import numpy as np
import stellargraph as sg
from stellargraph import StellarGraph
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, MeanPoolingAggregator, MeanAggregator, MaxPoolingAggregator
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
import matplotlib.pyplot as plt
from stellargraph import IndexedArray
from sklearn import model_selection
from sklearn.preprocessing import Normalizer, MinMaxScaler, RobustScaler
scaler = RobustScaler()
import warnings
import sys
warnings.filterwarnings('ignore')

model = tf.keras.models.load_model("model/graphSAGE.h5", custom_objects={'MaxPoolingAggregator':MaxPoolingAggregator})
print(model.summary())

Space_adj_file = "data/Space_adj.csv"
Space_feature_file = "data/Space_feature.csv"
Space_to_elements_file = "data/Space_element_adj.csv"

    # Edge import
edge_data = pd.read_csv(
    Space_adj_file,
    sep=",",  # tab-separated
    header=None,  # no heading row
    names=["target", "source"],  # set our own names for the columns
)
#print("Edge data is:\n", edge_data)

    # Node import
node_data = pd.read_csv(
    Space_feature_file,
    sep=",",  # tab-separated
    header=0,
)

    # Space - element import
space_element_data = pd.read_csv(
    Space_to_elements_file,
    sep=",",
    header=None
)

    # add space-element adj feature to node data
element_list = space_element_data.iloc[:, :4]
element_list = element_list.rename(columns={0: 's1', 1: 's1_label', 2: 's2', 3: 's2_label'})

id2type = dict(zip(element_list['s1'], element_list['s1_label']))
id2type_to_add = dict(zip(element_list['s2'], element_list['s2_label']))
id2type.update(id2type_to_add)

column_names = list(np.unique(element_list['s2_label']))
adj_feat = pd.DataFrame(0, index=node_data['space_id'], columns=column_names)

for i in element_list.index:

    s1_id = element_list.iloc[i]['s1']
    s2_id = element_list.iloc[i]['s2']

    try:
        s1_adj_type = id2type[s2_id]
        adj_feat.loc[s1_id, s1_adj_type] += 1

    except KeyError:
        continue

adj_feat_pre = adj_feat.copy()
adj_feat_pre[adj_feat_pre != 0] = 1
# adj_feat_pre = adj_feat_pre.drop(['othersWoodDoor'], axis=1)
# print(adj_feat_pre)

    # Node raw data reshape
space_feature_raw_content = node_data[['space_id', 'areas', 'volumes', 'perimeter', 'X', 'Y', 'Z', 'aspect_ratio', 'boundary_line', 'surface_areas', 'ax1s', 'label']]
#print("Node data is: \n", space_feature_raw_content.head())
space_feature_raw_content = pd.concat([space_feature_raw_content, adj_feat_pre.reset_index(drop=True)], axis=1)
# print(space_feature_raw_content)

    # Node data without labels
space_content_no_subject = space_feature_raw_content.drop(columns="label")
# print("Node data without labels:\n", space_content_no_subject)

    # Only node feature data to array for Graph and make Index by space_id
feature_array = np.array(space_content_no_subject.drop(columns="space_id"))
# print(feature_array)

    # Scaling
robust = scaler.fit_transform(feature_array)
# print("robusted features:\n", robust)
# pd.DataFrame(robust).to_csv("robusted.csv")

feature_array = robust
indexed_array = IndexedArray(feature_array, index=node_data["space_id"])

space_graph = StellarGraph(indexed_array, edge_data, node_type_default="spaces", edge_type_default="edges")
# print(space_graph.info())

    # True labels = space_subject make series with index(space_id)
space_subject = pd.Series(space_feature_raw_content["label"])
space_subject.index = node_data["space_id"]
# print(space_subject)
set(space_subject)

    # Counts the subject
from collections import Counter
print(Counter(space_subject))

    #Converting to numerica arrays
target_encoding = preprocessing.LabelBinarizer()
test_targets = target_encoding.fit_transform(space_subject)

batch_size = 32
num_samples = [4]

generator = GraphSAGENodeGenerator(space_graph, batch_size, num_samples)
test_gen = generator.flow(space_subject.index, test_targets)

test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

test_predictions = model.predict(test_gen)
node_predictions = target_encoding.inverse_transform(test_predictions)

df_test = pd.DataFrame({"Predicted": node_predictions, "True": space_subject})
df_test.to_csv('space_graphSAGE_pool_2.csv')

from sklearn.metrics import confusion_matrix

y_true = space_subject
y_pred = node_predictions

cm = pd.DataFrame(confusion_matrix(y_true, y_pred))
cmv = cm.values
acc = [np.round((cmv[i][i])/(np.sum(cmv[i])), 2) for i in range(len(cm))]
cm['acc by class']= acc
print(cm)