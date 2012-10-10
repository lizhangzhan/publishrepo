#! /usr/bin/python
import random
"""
Question: Select the m values that close to the key x from BST
Analysis: time complexity = O(m * lg(n)) space complexity = O(m)
Note:
"eahc node in BST contains attributes left. right, parent that
point to nodes conresponding to the left child, right child,
and its parent repectively." 
Reference:
Thomas H. Cormen, Introduction to the algorithm, the third edition, PP.286

"""
class Node:
    left, right, parent, data = None, None, None, 0

    def __init__(self, data):
        #initializes the data members
        self.left = None
        self.right = None
        self.parent = None
        self.data = data

class BSTree:
    def __init__(self):
        #initializes the root member
        self.root =None
    def add_node(self, value):
        # create a new node and returns it
        return Node(value)

    def insert_node(self, value):
        y = None
        x = self.root
        while x != None:
            y = x
            if x.data < value:
                x = x.right
            else:
                x = x.left
        node = self.add_node(value)
        node.parent = y

        if y == None:
            # The Tree is empty
            self.root = node
        else:
            if value < y.data:
                y.left = node
            else:
                y.right = node
    def search(self, x):
        if self.root == None or x == self.root.data:
            return self.root
        p = self.root
        while p != None:
            if x == p.data:
                return p
            elif x < p.data:
                p = p.left
            else:
                p = p.right
        return None

    def minimum(self, x):
        while x.left != None:
            x = x.left
        return x
    def maximum(self, x):
        while x.right != None:
            x = x.right
        return x
    def successor(self, x):
        if x.right != None:
            return self.minimum(x.right)
        y = x.parent
        while y != None and y.right == x:
            x = y
            y = y.parent
        return y
    def predecessor(self, x):
        if x.left != None:
            return self.maximum(x.left)
        y = x.parent
        while y != None and y.left == x:
            x = y
            y = y.parent
        return y
    """
    Find the predecessors and the successors, select the m nearest values
    predecessor for x is the biggest value from values that are smaller than x
    successor for x is the smallest value from values that are bigger than x
    """
    def findNeighbors(self, x, m):
        count = 0
        results = []
        rev = self.search(x)
        
        suc = self.successor(rev)
        pred = self.predecessor(rev)
        while count < m:
            if (suc.data - x) < (x - pred.data):
                results.append(suc.data)
                suc = self.successor(suc)
            else:
                results.append(pred.data)
                pred = self.predecessor(pred)
            count += 1
        return results

def equal(actual, expected):
    
if __name__ == "__main__":
    tree = BSTree()
    datas = [2, 252, 401, 398, 330, 344, 397, 339, 323]
    random.shuffle(datas)
    for item in datas:
        tree.insert_node(item)

    results = tree.findNeighbors(344, 2)
