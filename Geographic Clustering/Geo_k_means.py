### Geographic K Means Clustering
### Develop a K-means clustering with distance function based on Haversine instead of geometric distance

import numpy as np

#create and define haversine distance between 2 coordinates

def haversine(x, y, radius = 6371):
    lat1, long1 = x
    lat2, long2 = y
    
    lat1, long1, lat2, long2 = map(np.radians, [lat1, long1, lat2, long2])
    
    dlat = lat2-lat1
    dlong = long2-long1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlong/2)**2
    
    c = 2*np.arcsin(np.sqrt(a))
    
    distance = radius*c
    
    return distance

#create class in clustering

class GKMeans:
    
    #create function to generate a set of initial centroids
        
    def rand_assign(self, latlong, centers):
            
        #randomly assign each coordinate to a cluster
        init_group = np.random.randint(1, centers+1, len(latlong))
            
        #create empty array to house centers
        cents = np.empty([centers, 2])
        
        #calculate centers for each 
        
        for i in range(1, centers):
            avg = np.average(latlong, axis = 0, weights=(init_group==i))
            cents[i-1,] = avg
            
        return cents
    
    #creates function for the update step - updates to what new centroids should be
    
    def update(self, latlong, group_ids, centers):
        
        updated_centers = centers
        
        #updates centers by calculating conditional averages based on cluster assignment of each coordinate
        
        for i in range(1, len(centers)):
            avg = np.average(latlong, axis = 0, weights=(group_ids==i))
            updated_centers[i-1,] = avg
            
        return updated_centers
    
    #creates the function for the assignment step based on given centroids - assigns coordinates to new centroids
    
    def assign(self, latlong, centers):
        
        #create variable to house group assignments and distances
        
        group_ids = np.arange(len(latlong))
        distance = np.arange(len(centers))
        centroids = centers
        
        #assignments nested loop
        
        for i in range(0, len(latlong)):
            
            #calculate distance against each centroid using haversine formula
            for dist in range(0, len(centroids)):
                distance[dist] = haversine(latlong[i,],centroids[dist,])
            
            #assign coordinate to the index of the centroid with minimum distance
            group_ids[i] = np.argmin(distance)+1
        
        return group_ids
    
    
    
    def __init__(self, latlong, centers, init = None, n = 100, radius = 6371):
        self.latlong = latlong
        self.groups = centers
        
        #randomly assign initalization if none were provided
        
        if init == None:
            self.init = self.rand_assign(self.latlong,self.groups)
        else:
            self.init = init
        
        self.radius = radius
        self.iters = n
        
        self.clusters = self.assign(self.latlong, self.init)
        self.centroids = self.update(self.latlong, self.clusters, self.init)
        
        #generate clusters based on classification
        i = 0
        
        for i in range(0, n):
                
            self.clusters = self.assign(self.latlong, self.centroids)
            self.centroids = self.update(self.latlong, self.clusters, self.centroids)
                
            i += 1
        
            

#test script/demonstration
        
if __name__ == '__main__':
    
    
    #randomly generate coordinates
    longlat = np.random.uniform(-179,179,[100,2])

    groups = 6
    
    kmeans = GKMeans(latlong = longlat, centers = groups, n = 300)
    
    print(kmeans.clusters)
    
    
    