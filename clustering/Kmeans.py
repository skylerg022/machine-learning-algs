import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
import random

class KMEANSClustering(BaseEstimator, ClusterMixin):

    def __init__(self,k=3,debug=False):
        """
        Args:
            k (int): the number of clusters at which to stop the cluster-grouping algorithm
            debug (bool): if debug is true, use the first k instances as the initial centroids otherwise choose 
                random points as the initial centroids.
        """
        
        self.k = k
        self.debug = debug

    def fit(self, X):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X: A 2D numpy array with the training data
        Returns:
            self: this allows function chaining, e.g. model.fit(X).save_clusters('clusters.txt')
        """
        
        point_indices = range(X.shape[0])
        if self.debug:
            centroids = X[:self.k]
        else:
            cent = random.sample(point_indices, k=self.k)
            centroids = X[cent]
        centroids_moved = True
        
        while centroids_moved:
            centroids_moved = False
            clusters = [[] for i in range(self.k)]
            distances = [[] for i in range(self.k)]
            
            # Determine which centroid each point is closest to
            for p in point_indices:
                point = X[p]
                best_dist = np.inf
                best_cent = -1
                for c in range(self.k):
                    dist = np.sum( (point-centroids[c])**2 )**.5
                    if dist < best_dist:
                        best_dist = dist
                        best_cent = c
                clusters[best_cent].append(p) # Add point into closest cluster
                distances[best_cent].append(best_dist)# Add calculated distance into closest cluster
            
            new_centroids = []
            clusters_sse = []
            for c in range(self.k):
                # Calculate cluster SSE
                sse = np.sum( np.array(distances[c])**2 )
                clusters_sse.append(sse)
                
                # Calculate new centroids
                points = clusters[c]
                new_cent = np.mean(X[points], axis=0)
                new_centroids.append(new_cent)
            
            new_centroids = np.array(new_centroids)
            if (new_centroids != centroids).any():
                centroids_moved = True
            centroids = new_centroids
        
        # Save final iteration variables for reporting
        self.report = (centroids, clusters, clusters_sse)
        return self
    
    def save_clusters(self,filename):
        """
        Args:
            filename: file path to save summary statistics
        Returns:
            Saves algorithm statistics to the path provided
        """
        
        centroids, clusters, clusters_sse = self.report
        
        f = open(filename, 'w+')
        f.write('Clusters: {:d}\n'.format(self.k))
        f.write('Total SSE: {:.4f}\n\n'.format(sum(clusters_sse)))
        for c in range(self.k):
            f.write('Centroid {:d} Location: '.format(c))
            f.write(np.array2string(centroids[c],precision=4,separator=',')+'\n')
            f.write('Observations in Cluster: {:d}\n'.format(len(clusters[c])))
            f.write('Cluster SSE: {:.4f}\n\n'.format(clusters_sse[c]))
        f.close()
        
        return
