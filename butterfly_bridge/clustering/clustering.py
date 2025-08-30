import librosa
import tqdm
import numpy as np

from sklearn import metrics
from sklearn.cluster import DBSCAN

from butterfly_bridge.clustering.denstream import DenStream
from butterfly_bridge.visualization.plotting import plt, pg, pv, pvq

from butterfly_bridge.streaming.buffer import ArrayBuffer

class Clustering:
    def __init__(self,epsilon=200, lambd=0.00001, beta=0.6, mu=2, min_samples=1):
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
        self.stream = ds
        self.n_samples = 0

        # self.samples = ArrayBuffer(sample_shape=)

    def add_sample_to_clustering(self, features, verbose: bool = False):
        ds = self.stream
        n = self.n_samples


        cluster, is_outlier = ds.partial_fit(features[np.newaxis, :], time=self.n)
        cluster_labels = ds.request_clustering()

        if verbose:
            print(f'accepted={len(ds.p_micro_clusters)}, outlier={len(ds.o_micro_clusters)} '
                  f'number of clusters={len(np.unique(cluster_labels))}')

        label = cluster_labels[np.where([cluster is c for c in ds.p_micro_clusters])[0]]

        self.n_samples += 1

        return label
