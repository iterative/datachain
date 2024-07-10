import random

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def select_diverse_elements(embeddings: list, num_clusters: int, top_d: int):
    diverse_elements = []

    # Prepare the embeddings (convert list(tuples) to list(list))
    embeddings = [res[0] for res in embeddings]

    # Step 1: Cluster the embeddings
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Step 2: Calculate distances from centroids
    distances = cdist(embeddings, centroids, "euclidean")

    # Step 3: Select cluster cnetroids
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster_distances = distances[cluster_indices, i]

        # Get the index of the centroid (closest to the cluster center)
        centroid_index = cluster_indices[np.argmin(cluster_distances)]
        diverse_elements.append(centroid_index)

        # Select other elements randomly from the cluster, excluding the centroid
        other_indices = list(set(cluster_indices) - {centroid_index})
        random.shuffle(other_indices)
        number_to_select = top_d - 1  # Since we already have the centroid

        if number_to_select > len(other_indices):
            diverse_elements.extend(other_indices)
        else:
            diverse_elements.extend(other_indices[:number_to_select])

    return diverse_elements
