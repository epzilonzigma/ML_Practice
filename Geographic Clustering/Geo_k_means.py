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
    
    
    
    #create function for random initialization parameters
        
    def rand_assign(self, latlong, centers):
            
        #randomly assign each coordinate to a cluster
        init_group = np.random.randint(1, centers, len(latlong))
            
        #create empty array to house centers
        cents = np.empty([centers, 2])
            
        for i in range(1, centers):
            avg = np.average(latlong, axis = 0, weights=(init_group==i))
            cents[i-1,] = avg
            
        return cents
    
    
        
    

#test script/demonstration
        
if __name__ == '__main__':
    
    
    #randomly generate coordinates
    longlat = np.random.uniform(-179,179,100)

    groups = 6
    