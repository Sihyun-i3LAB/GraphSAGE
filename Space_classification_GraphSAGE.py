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
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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

    # Index setting - space_id
# space_content_str_subject = space_feature_raw_content.set_index("space_id")
# print("Node data with index:\n", space_content_str_subject)

    # Node data without labels
space_content_no_subject = space_feature_raw_content.drop(columns="label")
# print("Node data without labels:\n", space_content_no_subject)

    #Only node feature data to array for Graph and make Index by space_id
feature_array = np.array(space_content_no_subject.drop(columns="space_id"))
# print(feature_array)

    # Scaling
robust = scaler.fit_transform(feature_array)
print("robusted features:\n", robust)
# pd.DataFrame(robust).to_csv("robusted.csv")

feature_array = robust
indexed_array = IndexedArray(feature_array, index=node_data["space_id"])

space_graph = StellarGraph(indexed_array, edge_data, node_type_default="spaces", edge_type_default="edges")
# print(space_graph.info())

    #True labels = space_subject make series with index(space_id)
space_subject = pd.Series(space_feature_raw_content["label"])
space_subject.index = node_data["space_id"]
# print(space_subject)

set(space_subject)

    #Splitting the data
train_subjects, test_subjects = model_selection.train_test_split(
    space_subject, train_size=0.6, test_size=0.4, stratify=space_subject, shuffle=True)
# print(train_subjects)
# print(test_subjects)

val_subjects, test_subjects = model_selection.train_test_split(
    test_subjects, train_size=0.25, test_size=0.75, stratify=test_subjects, shuffle=True)

    #Counts the subject
from collections import Counter
print(Counter(train_subjects))

    #Converting to numerica arrays
target_encoding = preprocessing.LabelBinarizer()
train_targets = target_encoding.fit_transform(train_subjects)
# print("train_targets:\n", train_targets)
val_targets = target_encoding.transform(val_subjects)
# print("val_targets:\n", val_targets)
test_targets = target_encoding.transform(test_subjects)
# print("test_targets:\n", test_targets)

    #Creating the GraphSAGE model
batch_size = 32
num_samples = [4]
generator = GraphSAGENodeGenerator(space_graph, batch_size, num_samples)
train_gen = generator.flow(train_subjects.index, train_targets, shuffle=True)
# print(train_subjects.index)

graphsage_model = GraphSAGE(
    layer_sizes=[32], generator=generator, aggregator=MaxPoolingAggregator, bias=True, dropout=0.5
)

x_inp, x_out = graphsage_model.in_out_tensors()
prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

model = Model(inputs=x_inp, outputs=prediction)
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.005),
    loss=losses.categorical_crossentropy,
    metrics=["acc"],
)
print(model.summary())

test_gen = generator.flow(test_subjects.index, test_targets)
val_gen = generator.flow(val_subjects.index, val_targets)

from tensorflow.keras.callbacks import EarlyStopping
es_callback = EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)

history = model.fit(
    train_gen, epochs=150, validation_data=val_gen, verbose=2, shuffle=False, callbacks=[es_callback]
)

model.save('model/graphSAGE.h5')

sg.utils.plot_history(history)

test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

test_predictions = model.predict(test_gen)
node_predictions = target_encoding.inverse_transform(test_predictions)

df_test = pd.DataFrame({"Predicted": node_predictions, "True": test_subjects})
df_test.to_csv('space_graphSAGE_pool_2.csv')

    #Confusion matrix
from sklearn.metrics import confusion_matrix

y_true = test_subjects
y_pred = node_predictions

cm = pd.DataFrame(confusion_matrix(y_true, y_pred))
cmv = cm.values
acc = [np.round((cmv[i][i])/(np.sum(cmv[i])), 2) for i in range(len(cm))]
cm['acc by class']= acc
print(cm)

    #graphML
all_nodes = space_subject.index
all_mapper = generator.flow(all_nodes)
all_predictions = model.predict(all_mapper)

node_predictions = target_encoding.inverse_transform(all_predictions)
df = pd.DataFrame({"Predicted": node_predictions, "True": space_subject})

Gnx = space_graph.to_networkx(feature_attr=None)

for nid, pred, true in zip(df.index, df["Predicted"], df["True"]):
    Gnx.nodes[nid]["subject"] = true
    Gnx.nodes[nid]["PREDICTED_subject"] = pred.split("=")[-1]

for nid in train_subjects.index:
    Gnx.nodes[nid]["isTrain"] = True

for nid in test_subjects.index:
    Gnx.nodes[nid]["isTrain"] = False

for nid in Gnx.nodes():
    Gnx.nodes[nid]["isCorrect"] = (
        Gnx.nodes[nid]["subject"] == Gnx.nodes[nid]["PREDICTED_subject"]
    )

#pred_fname = "pred_n={}.graphml".format(num_samples)
nx.write_graphml(Gnx, "data/graphml/space_test_1.graphml")

    #T-SNE with all data
embedding_model = Model(inputs=x_inp, outputs=x_out)
emb = embedding_model.predict(all_mapper)
emb.shape

X = emb
y = np.argmax(target_encoding.transform(space_subject), axis=1)

if X.shape[1] > 2:
    transform = TSNE  # or PCA

    trans = transform(n_components=2)
    emb_transformed = pd.DataFrame(trans.fit_transform(X), index=space_subject.index)
    emb_transformed["label"] = y
else:
    emb_transformed = pd.DataFrame(X, index=space_subject.index)
    emb_transformed = emb_transformed.rename(columns={"0": 0, "1": 1})
    emb_transformed["label"] = y

alpha = 0.7

fig, ax = plt.subplots(figsize=(7, 7))
scatter = ax.scatter(
    emb_transformed[0],
    emb_transformed[1],
    c=emb_transformed["label"].astype("category"),
    cmap="jet",
    alpha=alpha,
)

legend1 = ax.legend(*scatter.legend_elements(), title="Classes", loc='center left', bbox_to_anchor=(1, 0.5))
ax.add_artist(legend1)

emb_transformed.columns = ['x', 'y','label']
label_name = np.unique(emb_transformed["label"])
label_name = ['AD/PD', 'Balcony', 'Bathroom', 'Bedroom', 'Dress room', 'Entrance', 'Evacuation room', 'Hallway', 'Kitchen', 'Living room', 'Storage']
le = LabelEncoder()
le.fit(label_name)
le.inverse_transform(list(emb_transformed["label"].values))
emb_transformed["label"] = le.inverse_transform(list(emb_transformed["label"].values))

for name, group in emb_transformed.groupby('label'):
    ax.scatter(group.x, group.y, label = name, cmap="jet")
    ax.legend(loc='center left', bbox_to_anchor=(1,0.5))

ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
plt.title(
    "{} visualization of GCN 1-layer for space dataset".format(transform.__name__)
)
plt.show()