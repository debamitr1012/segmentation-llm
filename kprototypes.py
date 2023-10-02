import pandas as pd # dataframe manipulation
import numpy as np # linear algebra
# data visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import shap
# sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import PowerTransformer, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples, accuracy_score, classification_report
from pyod.models.ecod import ECOD
from yellowbrick.cluster import KElbowVisualizer
import lightgbm as lgb
import prince
from kmodes.kprototypes import KPrototypes
from plotnine import *
import plotnine
df = pd.read_csv("C:/Users/91983/Downloads/Clustering-with-LLM-main/train.csv", sep = ";")
df = df.iloc[:, 0:8]
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
pipe = Pipeline([('ordinal', OrdinalEncoder()), ('scaler', PowerTransformer())])
pipe_fit = pipe.fit(df)
data = pd.DataFrame(pipe_fit.transform(df), columns = df.columns)
data
from pyod.models.ecod import ECOD
# https://github.com/yzhao062/pyod
clf = ECOD()
clf.fit(data)
outliers = clf.predict(data)
df["outliers"] = outliers
df_no_outliers = df[df["outliers"] == 0]
df_no_outliers = df_no_outliers.drop(["outliers"], axis = 1)
df_no_outliers
"""Dataset normalized without onehot preprocessing. Get the numeric features to modify their scale

"""
pipe = Pipeline([('scaler', PowerTransformer())])
df_aux = pd.DataFrame(pipe_fit.fit_transform(df_no_outliers[["age", "balance"]] ), columns = ["age", "balance"])
df_no_outliers_norm = df_no_outliers.copy()
# Replace age and balance columns by preprocessed values
df_no_outliers_norm = df_no_outliers_norm.drop(["age", "balance"], axis = 1)
df_no_outliers_norm["age"] = df_aux["age"].values
df_no_outliers_norm["balance"] = df_aux["balance"].values
df_no_outliers_norm
df_no_outliers
"""DataSet with onehot preprocessing AND normalized"""
data["outliers"] = outliers
data_no_outliers = data[data["outliers"] == 0]
data_no_outliers = data_no_outliers.drop(["outliers"], axis = 1)
"""Optimal number of cluster (be careful, it takes a long time)
We get the index of categorical columns
"""
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorical_columns = df_no_outliers_norm.select_dtypes(exclude=numerics).columns
print(categorical_columns)
categorical_columns_index = [df_no_outliers_norm.columns.get_loc(col) for col in categorical_columns]
# # Choose optimal K using Elbow method
# cost = []
# range_ = range(2, 15)
# for cluster in range_:

#         kprototype = KPrototypes(n_jobs = -1, n_clusters = cluster, init = 'Huang', random_state = 0)
#         kprototype.fit_predict(df_no_outliers, categorical = categorical_columns_index)
#         cost.append(kprototype.cost_)
#         print('Cluster initiation: {}'.format(cluster))
# # Converting the results into a dataframe and plotting them
# df_cost = pd.DataFrame({'Cluster':range_, 'Cost':cost})
# # Data viz
# plotnine.options.figure_size = (8, 4.8)
# (
#     ggplot(data = df_cost)+
#     geom_line(aes(x = 'Cluster',
#                   y = 'Cost'))+
#     geom_point(aes(x = 'Cluster',
#                    y = 'Cost'))+
#     geom_label(aes(x = 'Cluster',
#                    y = 'Cost',
#                    label = 'Cluster'),
#                size = 10,
#                nudge_y = 1000) +
#     labs(title = 'Optimal number of cluster with Elbow Method')+
#     xlab('Number of Clusters k')+
#     ylab('Cost')+
#     theme_minimal()
# )
# https://github.com/MaxHalford/prince
df_no_outliers_norm
cluster_num = 5
kprototype = KPrototypes(n_jobs = -1, n_clusters = cluster_num, init = 'Huang', random_state = 0)
kprototype.fit(df_no_outliers_norm, categorical = categorical_columns_index)
clusters = kprototype.predict(df_no_outliers_norm , categorical = categorical_columns_index)
clusters
# Cluster centorid
print(kprototype.cluster_centroids_)
# Check the iteration of the clusters created
print(kprototype.n_iter_)
# Check the cost of the clusters created
print(kprototype.cost_)
from prince import MCA
def get_MCA_3d(df, predict):
    mca = MCA(n_components =3, n_iter = 100, random_state = 101)
    mca_3d_df = mca.fit_transform(df)
    mca_3d_df.columns = ["comp1", "comp2", "comp3"]
    mca_3d_df["cluster"] = predict
    return mca, mca_3d_df
def get_MCA_2d(df, predict):
    mca = MCA(n_components =2, n_iter = 100, random_state = 101)
    mca_2d_df = mca.fit_transform(df)
    mca_2d_df.columns = ["comp1", "comp2"]
    mca_2d_df["cluster"] = predict
    return mca, mca_2d_df
mca_3d, mca_3d_df = get_MCA_3d(df_no_outliers_norm, clusters)
mca_3d_df
mca_3d.eigenvalues_summary
def get_pca_2d(df, predict):
    pca_2d_object = prince.PCA(
    n_components=2,
    n_iter=3,
    rescale_with_mean=True,
    rescale_with_std=True,
    copy=True,
    check_input=True,
    engine='sklearn',
    random_state=42
    )
    pca_2d_object.fit(df)
    df_pca_2d = pca_2d_object.transform(df)
    df_pca_2d.columns = ["comp1", "comp2"]
    df_pca_2d["cluster"] = predict
    return pca_2d_object, df_pca_2d
def get_pca_3d(df, predict):
    pca_3d_object = prince.PCA(
    n_components=3,
    n_iter=3,
    rescale_with_mean=True,
    rescale_with_std=True,
    copy=True,
    check_input=True,
    engine='sklearn',
    random_state=42
    )
    pca_3d_object.fit(df)
    df_pca_3d = pca_3d_object.transform(df)
    df_pca_3d.columns = ["comp1", "comp2", "comp3"]
    df_pca_3d["cluster"] = predict
    return pca_3d_object, df_pca_3d
def plot_pca_3d(df, title = "PCA Space", opacity=0.8, width_line = 0.1):
    df = df.astype({"cluster": "object"})
    df = df.sort_values("cluster")
    fig = px.scatter_3d(df,
                        x='comp1',
                        y='comp2',
                        z='comp3',
                        color='cluster',
                        template="plotly",
                        # symbol = "cluster",
                        color_discrete_sequence=px.colors.qualitative.Vivid,
                        title=title).update_traces(
                            # mode = 'markers',
                            marker={
                                "size": 4,
                                "opacity": opacity,
                                # "symbol" : "diamond",
                                "line": {
                                    "width": width_line,
                                    "color": "black",
                                }
                            }
                        ).update_layout(
                                width = 1000,
                                height = 800,
                                autosize = False,
                                showlegend = True,
                                legend=dict(title_font_family="Times New Roman",
                                            font=dict(size= 20)),
                                scene = dict(xaxis=dict(title = 'comp1', titlefont_color = 'black'),
                                            yaxis=dict(title = 'comp2', titlefont_color = 'black'),
                                            zaxis=dict(title = 'comp3', titlefont_color = 'black')),
                                font = dict(family = "Gilroy", color  = 'black', size = 15))
    fig.show()
def plot_pca_2d(df, title = "PCA Space", opacity=0.8, width_line = 0.1):
    df = df.astype({"cluster": "object"})
    df = df.sort_values("cluster")
    fig = px.scatter(df,
                        x='comp1',
                        y='comp2',
                        color='cluster',
                        template="plotly",
                        # symbol = "cluster",
                        color_discrete_sequence=px.colors.qualitative.Vivid,
                        title=title).update_traces(
                            # mode = 'markers',
                            marker={
                                "size": 8,
                                "opacity": opacity,
                                # "symbol" : "diamond",
                                "line": {
                                    "width": width_line,
                                    "color": "black",
                                }
                            }
                        ).update_layout(
                                width = 800,
                                height = 700,
                                autosize = False,
                                showlegend = True,
                                legend=dict(title_font_family="Times New Roman",
                                            font=dict(size= 20)),
                                scene = dict(xaxis=dict(title = 'comp1', titlefont_color = 'black'),
                                            yaxis=dict(title = 'comp2', titlefont_color = 'black'),
                                            ),
                                font = dict(family = "Gilroy", color  = 'black', size = 15))
    fig.show()
plot_pca_3d(mca_3d_df, title="MCA Space", opacity=1, width_line = 0.1)
mca_2d, mca_2d_df = get_MCA_2d(df_no_outliers_norm, clusters)
plot_pca_2d(mca_3d_df, title="MCA Space", opacity=1, width_line = 0.5)
pca_3d_object, df_pca_3d = get_pca_3d(data_no_outliers, clusters)
plot_pca_3d(df_pca_3d, title = "PCA Space", opacity=1, width_line = 0.1)
print("The variability is :", pca_3d_object.eigenvalues_summary)
pca_2d_object, df_pca_2d = get_pca_2d(data_no_outliers, clusters)
plot_pca_2d(df_pca_2d, title = "PCA Space", opacity=1, width_line = 0.5)
import lightgbm as lgb
import shap
clf_km = lgb.LGBMClassifier(colsample_by_tree=0.8)
for col in ["job", "marital", "education", "housing", "loan", "default"]:
    df_no_outliers_norm[col] = df_no_outliers_norm[col].astype('category')
clf_km.fit(X = df_no_outliers_norm , y = clusters, feature_name = "auto", categorical_feature = "auto")
# clf_km.fit(X = df_prueba, y = predict_embedding, feature_name='auto', categorical_feature = 'auto')
#SHAP values
explainer_km = shap.TreeExplainer(clf_km)
shap_values_km = explainer_km.shap_values(df_no_outliers_norm)
shap.summary_plot(shap_values_km, df_no_outliers_norm, plot_type="bar", plot_size=(15, 10))
df_no_outliers = df[df.outliers == 0]
df_no_outliers["cluster"] = clusters
df_no_outliers.groupby('cluster').agg(
    {
        'age':'mean',
        'balance': 'mean',
        'job': lambda x: x.value_counts().index[0],
        'marital': lambda x: x.value_counts().index[0],
        'education': lambda x: x.value_counts().index[0],
        'housing': lambda x: x.value_counts().index[0],
        'loan': lambda x: x.value_counts().index[0],
        'default': lambda x: x.value_counts().index[0],
    }
).sort_values("balance").reset_index()