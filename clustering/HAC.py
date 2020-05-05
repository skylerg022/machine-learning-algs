import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

class HACClustering(BaseEstimator, ClusterMixin):

    def __init__(self, k=3, link_type='single'):
        """
        Args:
            k (int): the number of clusters at which to stop the cluster-grouping algorithm
            link_type (str): either 'single' or 'complete'; single-link determines the closeness of two clusters
                using the distance of the closest two points, both in a different cluster; complete-link uses the 
                distance of the furthest two points between two clusters
        """
        
        self.link_type = link_type
        self.k = k
        
    def fit(self, X):
        """
        Args:
            X: A 2D numpy array with the training data
        Returns:
            self: this allows function chaining, e.g. model.fit(X).save_clusters('clusters.txt')
        """
        
        self.out = []
        clusters = [[i] for i in range(X.shape[0])]
        while len(clusters) > self.k: # Combine clusters while there are more than k clusters
            best_clusters = ()
            best_dist = np.inf
            
            # Determine closest two clusters
            for c1 in range(len(clusters)-1):
                for c2 in range(c1+1, len(clusters)):
                    furthest_dist = -1
                    cluster1 = clusters[c1]
                    cluster2 = clusters[c2]
                    for i in cluster1:
                        for j in cluster2:
                            dist = np.round(np.sum( (X[i]-X[j])**2 )**0.5, 6)
                            if dist < best_dist and self.link_type == 'single':
                                best_dist = dist
                                best_clusters = (c1, c2)
                            elif dist > furthest_dist and self.link_type == 'complete':
                                furthest_dist = dist
                    if furthest_dist < best_dist and self.link_type == 'complete':
                        best_dist = furthest_dist
                        best_clusters = (c1, c2)
            
            c1, c2 = best_clusters
            clusters[c1] = np.concatenate((clusters[c1], clusters[c2]))
            clusters = np.delete(clusters, c2)
        
        # Calculate report variables for save_clusters() function
        centroids = []
        clusters_sse = []
        for c in range(self.k):
            # Calculate new centroids
            points = np.array(clusters[c])
            cent = np.mean(X[points], axis=0)
            
            # Calculate SSE for cluster
            sse = 0
            for p in points:
                point = X[p]
                sse += np.sum( (point-cent)**2 )
            centroids.append(cent)
            clusters_sse.append(sse)

        self.centroids = centroids
        self.clusters = clusters
        self.clusters_sse = clusters_sse
        return self
    
    def save_clusters(self, filename):
        """
        Args:
            filename: file path to save summary statistics
        Returns:
            Saves algorithm statistics to the path provided
        """
        
        f = open(filename, 'w+')
        f.write('Clusters: {:d}\n'.format(self.k))
        f.write('Total SSE: {:.4f}\n\n'.format(sum(self.clusters_sse)))
        for c in range(self.k):
            f.write('Centroid {:d} Location: '.format(c))
            f.write(np.array2string(self.centroids[c],precision=4,separator=',')+'\n')
            f.write('Observations in Cluster: {:d}\n'.format(len(self.clusters[c])))
            f.write('Cluster SSE: {:.4f}\n\n'.format(self.clusters_sse[c]))
        f.close()
        
        return
