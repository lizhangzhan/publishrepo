import numpy
import math
import random
import collections

class Kmean:
    def init_centers(self):
        pass
    def do_cluster(self, mintoler, numiter):
        pass

class Point:
    def __init__(self, data):
        self.data=data
        self.cluster=None
        self.dist=1e5

    def compute_euclidean_distance(self, center):
        if len(self.data) != len(center):
            print "mismatch dimension!"
            return
        return math.sqrt(sum(numpy.power((numpy.subtract(self.data, center)), 2)))
    def set_cluster(self, c, d):
        if self.cluster:
            self.cluster.remove_point(self)

        self.cluster = c
        self.cluster.add_point(self)
        self.dist = d
        
    def get_dist(self):
        return self.dist

    def set_dist(self, d):
        self.dist = d 
class Cluster:
    def __init__(self, center):
        self.center = center
        self.points = collections.defaultdict(lambda:0)
    def add_point(self, p):
        self.points[p] = 1
    def remove_point(self, p):
        if self.points[p]:
            del self.points[p]
    def update_center(self):
        if (len(self.points) == 0):
            return
        sumvect = [0] * len(self.center)
        for p in self.points.keys():
            sumvect = numpy.add(sumvect, p.data);
        self.center = numpy.divide(sumvect, len(self.points))
            
class Kmeanplusplus(Kmean):
    def __init__(self, ds, numdata, numCluster):
        self.dataset = ds
        self.numPoint = numdata 
        self.numCluster = numCluster
        self.points= []
        self.clusters = []

        for i in range(self.numPoint):
            point = Point(ds[i])
            self.points.append(point)

    def init_centers(self):
        # random to get the first init center
        newclusterindex = int(random.random() * self.numPoint)
        token = [1] * self.numPoint
        distsqusum = 0.0
        k = 1 

        token[newclusterindex] = 0
        seed_cluster = Cluster(self.dataset[newclusterindex])
        self.clusters.append(seed_cluster)
        newcluster = seed_cluster
        for i in range(self.numPoint):
            point = self.points[i]
            dist = point.compute_euclidean_distance(newcluster.center)
            point.set_cluster(newcluster, dist)
            if (token[i]):
                distsqusum += math.pow(dist, 2)
            
        while (k < self.numCluster):
            # random to select a point as the center of a new cluster
            rdist = random.random() * distsqusum
            newclusterindex = -1
            tmp = 0
            for i in range(self.numPoint):
                tmp += self.points[i].get_dist() 
                if (tmp >= rdist):
                    newclusterindex = i
            
            newcluster = Cluster(self.dataset[newclusterindex])
            self.clusters.append(newcluster);
            k += 1
            distsqusum = 0.0
            #update the euclidean distance between the point and its newest center
            for i in range(self.numPoint):
                point = self.points[i]
                if (token[i]):
                    dist = point.compute_euclidean_distance(newcluster.center)
                    if (dist < point.get_dist()):
                        point.set_cluster(newcluster,dist)
                    distsqusum += math.pow(point.get_dist(), 2)

            token[newclusterindex] = 0
    def assign_point_cluster(self, point):
        mindist = point.get_dist()
        cluster = point.cluster
        change = 0
        for c in self.clusters:
            dist = point.compute_euclidean_distance(c.center)
            if (dist < mindist):
                cluster = c
                mindist = dist
                change = 1
        if change:
            point.set_cluster(cluster, mindist)
        return change

    def do_cluster(self, numiter):
        # Select the initial cluster center randomly
       self.init_centers();
    
       for i in range(numiter):
           change = 0
           # E step % update the center for each cluster
           for cluster in self.clusters:
               cluster.update_center()

          # M step % assign the point to the newest cluster
           for point in self.points:
               change += self.assign_point_cluster(point)
           
           if (change == 0):
               break
if __name__ == "__main__":
    dataset= []
    for i in range(5):
        dataset.append([random.gauss(5,1), random.gauss(6,1)])

    for i in range(5):
        dataset.append([random.gauss(10,1), random.gauss(12,1)])

    for i in range(5):
        dataset.append([random.gauss(1,1), random.gauss(1,1)])
    random.shuffle(dataset)
    kmean = Kmeanplusplus(dataset,15, 3)
    kmean.do_cluster(100)

    for i in range(15):
        print dataset[i]

    print "cluster1"
    for point in kmean.clusters[0].points:
        print point.data
    print "over"
