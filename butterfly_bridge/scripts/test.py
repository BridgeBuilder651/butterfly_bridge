import librosa
import tqdm
import numpy as np

from sklearn import metrics
from sklearn.cluster import DBSCAN

from butterfly_bridge.clustering.denstream import DenStream
from butterfly_bridge.visualization.plotting import plt, pg, pv, pvq

waveforms = np.load('./data/waveforms.npy')
features = np.load('./data/features.npy')
labels = np.load('./data/labels.npy')

#%%

epsilon = 200
lambd = 0.00001
beta = 0.6
mu = 2
min_samples = 1
# label_metrics_list = [metrics.homogeneity_score, metrics.completeness_score]

ds = DenStream(
    epsilon=epsilon,
    beta=beta,
    mu=mu,
    lambd=lambd,
    min_samples=min_samples,
    # label_metrics_list=label_metrics_list
)
ds.set_clustering_model(
    DBSCAN(
            eps=epsilon * 1,
            min_samples=min_samples,
            metric="euclidean",
            algorithm="auto",
            n_jobs=-1,
    )
)


ds_labels = []
for i in tqdm.tqdm(range(len(features))):
    cluster, is_outlier = ds.partial_fit(features[[i]], time=i, label=i)
    cluster_labels = ds.request_clustering()
print(f'accepted={len(ds.p_micro_clusters)}, outlier={len(ds.o_micro_clusters)}')

cluster_labels = ds.request_clustering()
print(f'number of clusters={len(np.unique(cluster_labels))}')

#%%

ds_labels = np.full(len(features), fill_value=-1, dtype=int)

for label, cluster in zip(cluster_labels, ds.p_micro_clusters):
    ds_labels[cluster.time_array.reshape(-1)] = label

from umap import UMAP
um = UMAP(n_components=3, n_neighbors=3)
Y = um.fit_transform(features)

from butterfly_bridge.visualization.plotting import plot_embedding_3d

plot_embedding_3d(Y, ds_labels, cmap='hsv')


from sklearn.decomposition import PCA
pca = PCA(n_components=3)
Y = pca.fit_transform(features)

plot_embedding_3d(Y, ds_labels, cmap='hsv')


librosa.pyin()




#%%



#%%

ds.metrics_results




