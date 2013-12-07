# !/usr/bin/python
import collections
"""
  Implement a pageranking algorithm in python.
  Reference:
  [1] pagerank wikipage, http://en.wikipedia.org/wiki/PageRank
  [2] A nice introduction to pageran algorithm, http://www.ams.org/samplings/feature-column/fcarc-pagerank
  
  Describe a directed graph with a represenation similar to adjacency list.
  With a difference, we do not directly store all tail nodes adjacient
  to a node instead of all head nodes adjacent to a node.
"""
class graph(object):
    """  
    In a directed graph, a node has the four attributes:
        nid - the node identifier
        inadj - all head nodes adjacent to this node
        outdegree - the number of tail nodes adjacient to this node
        score - the weight / score assigned to this node
    """
    class node(object):
        def __init__(self, nid):
            self.nid = nid
            self.inadj = []
            self.outdegree = 0
            self.score = 1.0
        
        # Add a node connecting to it
        def add_adj(self, node):
            self.inadj.append(node)

        # increment outdegree
        def incre_degree(self):
            self.outdegree = self.outdegree + 1
        
        def __iter__(self):
            return iter(self.inadj)

        def __str__(self):
            return "(" + str(self.nid) + ") \t "+ str(self.score)

    def __init__(self):
        self.name2id = collections.defaultdict()
        self.id2name = []
        self.nodes = []

    def get_node(self, name):
        if name not in self.name2id:
            self.name2id[name] = len(self.name2id)
            self.id2name.append(name)
            self.nodes.append(graph.node(self.name2id[name]));
        
        return self.nodes[self.name2id[name]]
    
    def get_name(self, node):
        return self.id2name[node.nid]
    
    # add a edge to the graph
    # Parameters:
    #   v1 - the name of start node
    #   v2 - the name of end node  
    def add_edge(self, v1, v2):
        node1 = self.get_node(v1)
        node2 = self.get_node(v2)

        node1.incre_degree()
        node2.add_adj(node1)
    
    # sort all nodes ordered by their scores
    def rank(self):
        self.nodes.sort(key = lambda node: node.score, reverse = True)

    def __len__(self):
        return len(self.nodes)
 
    def __iter__(self):
        return iter(self.nodes)
    
    def __str__(self):
        gstr = ""
        gstr += "Nodes ranking in the graph\n"
        for node in self:
            gstr += (self.get_name(node) + str(node) + '\n')
        return gstr
"""
        gstr = "Adjacency-list representation: \n"
        for node in self:
            gstr += (str(node.nid) + ":")
            for adj in node:
                gstr += "-> " + str(adj.nid)
            gstr += '\n'
"""        

class pagerank(object):
    def __init__(self, G, alpha):
        self.G = G
        self.V = len(G)
        self.alpha = alpha
    
    def rank(self, maxiter):
        for i in range(maxiter):
            scores = []
            # S1 - compute each node score
            for node in self.G:
                score = 0
                for adj in node:
                    score += (adj.score / adj.outdegree)
                # keep matrix primitive and irreducible
                scores.append(self.alpha * score + (1 - self.alpha) / self.V)
            # S2 - update node score
            scores.reverse()
            for node in self.G:
                node.score = scores.pop()
        # normalize 
        norm = reduce(lambda x, y: x + y, map(lambda n: n.score, self.G))
        for node in self.G:
            node.score /= norm
        # sort nodes
        self.G.rank()

# dangling node in graph
def testToy1():
    print "testing dangling node"
    G = graph()
    G.add_edge("node1", "node2")
    p = pagerank(G, .85)
    p.rank(10)
    print str(G)
    

# the matrix primitive and irreducible
def testToy2():
    print "testing matrix's primitive and irreducible attributes"
    G = graph()
    G.add_edge("node1", "node2")
    G.add_edge("node2", "node3")
    G.add_edge("node3", "node1")
    p = pagerank(G, .85)
    p.rank(10)
    print str(G)

# the matrix primitive and irreducible
def testToy3():
    print "testing matrix's primitive and irreducible attributes"
    G = graph()
    G.add_edge("node1", "node2")
    G.add_edge("node2", "node3")
    G.add_edge("node3", "node1")
    G.add_edge("node4", "node3")
    G.add_edge("node5", "node4")
    p = pagerank(G, .85)
    p.rank(10)
    print str(G)

def main():
    testToy1()
    testToy2()
    testToy3()

if __name__ == "__main__":
    main()
