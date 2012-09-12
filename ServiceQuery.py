#!/usr/bin/python

import random
import collections
class DataNode:
    def __init__(self, data):
        self.data = sorted(data)
        self.size = len(set(self.data))
    
    def getMinMax(self):
        return (self.data[0], self.data[-1])

    def searchQuery(self, target):
        dataset= self.data
        r1 = len(filter(lambda x:x<target, dataset))
        r2 = len(filter(lambda x:x==target, dataset))
        if r2 != 0:
            return (r1, r1+r2, target)
        elif r1 > 0:
            return (r1, r1, dataset[r1-1])
        else:
            return (r1, r1, None)


class ServiceCenter:
    def __init__(self, n, m, l, h):
        self.N = n
        self.data = []
        self.datanodes = []
        'randomly get the initial data'
        for i in xrange(n):
            item = random.randrange(l, h)
            self.data.append(item)
            
        random.shuffle(self.data)

        span = n / m
        start = 0
        for j in xrange(m):
            if j != m - 1:
                node = DataNode(self.data[start:start+span])
                start = start + span
            else:
                node = DataNode(self.data[start:])
            self.datanodes.append(node)

    def getMinMax(self):
        minval = 1e6
        maxval = 0
        interval = [0, 0]
        for node in self.datanodes:
            (lmin, lmax) = node.getMinMax()
            if lmin < minval:
                minval = lmin
            if lmax > maxval:
                maxval = lmax
        return (minval, maxval)
    def getOrder(self, q, k):
        count = 0
        left = 0
        right = 0
        values = []
        for node in self.datanodes:
            l, r, lval = node.searchQuery(q)
            left += l
            right += r       
            values.append(lval)
        if k >= left and k <= right:
            return (0, max(values))
        elif k < left:
            return (-1, -1)
        elif k > right:
            return (1, -1)
        
        
    def doService(self, k):
        if k > self.N or k < 0:
            print "index out of bound!"
            return
        
        lo, hi = self.getMinMax()
        while lo < hi:
            mid = lo + (hi - lo) / 2
            flag, value = self.getOrder(mid, k)
            if flag == 0:
                return value
            elif flag < 0:
                hi = mid
            else:
                lo = mid + 1
        return None
    
    def getKthValue(self, k):
        values = sorted(self.data)
        return values[k]

if __name__ == "__main__":
    service = ServiceCenter(1001, 10, 0, 500)
    #print "expected value: %d" % service.getKthValue(500)
    #value = service.doService(500)
    for i in xrange(10000):
        value = service.doService(501) # the actual value
        if value != service.getKthValue(501):
            print "unpass"

    #print "actual value: %d" % value
               

        
