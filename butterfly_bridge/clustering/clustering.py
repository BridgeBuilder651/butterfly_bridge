import librosa
import tqdm
import numpy as np

from sklearn import metrics
from sklearn.cluster import DBSCAN

from butterfly_bridge.clustering.denstream import DenStream
from butterfly_bridge.clustering.denstream.micro_cluster import MicroCluster


class Clustering:
    def __init__(self, epsilon=200, lambd=0.00001, beta=0.6, mu=2, min_samples=1):
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
        self.n = 0

    def append_transform(self, features, verbose: bool = False):
        ds = self.stream

        cluster, is_outlier = ds.partial_fit(features[np.newaxis, :], time=self.n)

        self.n += 1

        if verbose:
            print(f'================ {self.n} ==================')
            print(f'accepted={len(ds.p_micro_clusters)}, '
                  f'outlier={len(ds.o_micro_clusters)} '
                  f'{is_outlier=}')

        if is_outlier:
            return cluster.features_array[-1]
        else:
            cluster_labels = ds.request_clustering()
            index = self.cluster_to_index(cluster)
            label = cluster_labels[index]

            if verbose:
                print(f'number of clusters={len(np.unique(cluster_labels))}')

            return self.mean_feature([c for l, c in zip(cluster_labels, ds.p_micro_clusters) if l == label])

    def cluster_to_index(self, cluster):
        indices = [i for i, c in enumerate(self.stream.p_micro_clusters) if c == cluster]
        if len(indices) != 1:
            raise KeyError
        return indices[0]

    @classmethod
    def mean_feature(cls, clusters):
        features = np.concatenate([cluster.features_array for cluster in clusters], axis=0)
        weights = np.concatenate([[cluster.weight] * len(cluster.features_array) for cluster in clusters], axis=0)
        mean_feature = np.sum(weights * features , axis=0) / np.sum(weights)
        return mean_feature
