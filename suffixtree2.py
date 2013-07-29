# !/usr/bin/env python
import collections
import os

class Node(object):
    def __init__(self, nid, leaf=True):
        self.id = nid
        self.suffix_link = None
        self.leaf = leaf
    def repr(self):
        return "Node(%d)" % self.id
    def explain(self):
        _str =  ""
        if self.leaf:
            _str = "Node[shape = doublecircle] %d;\n" % self.id
        else:
            _str = "Node[shape = circle] %d;\n" % self.id
            if self.suffix_link != None:
                _str += "%d->%d[style = dotted];\n" % (self.id, self.suffix_link.id)
            
        return _str

class Edge(object):
    def __init__(self, start_char, end_char, start_node, end_node):
        self.start_char = start_char
        self.end_char = end_char
        self.start_node = start_node
        self.end_node = end_node
    
    def update_start(self, start_char, start_node):
        self.start_char = start_char
        self.start_node = start_node
        
    def update_end(self, end_char, end_node):
        self.end_char = end_char
        self.end_node = end_node

    def length(self):
        return (self.end_char - self.start_char)

    def explain(self, seq):
        return "%d->%d[label = \"%s\"]" % (self.start_node.id, self.end_node.id, seq[self.start_char:self.end_char])
    
    def repr(self):
        return "Edge(%d, %d, %d, %d)" % (self.start_char, self.end_char, self.start_node.id, self.end_node.id)

class ActivePoint(object):
    def __init__(self, active_node, active_edge, active_length):
        self.node = active_node
        self.edge =active_edge
        self.length = active_length
    def repr(self):
        return "ActivePoint(%s, %s, %d)" % (self.node.repr(), self.edge.repr(), self.length)
        
class SuffixTree(object):
    def __init__(self, seq):
        self.seq = seq
        self.elements = set(seq)
        self.size = len(self.seq)
        self.root = Node(0, None)
        self.node_list = [self.root]
        self.edge_dict = collections.defaultdict()
        self.active_point = ActivePoint(self.root, None, 0)
        self.parent_node = None
        self.suffix_link = None
        self.reminder = 1 # the number of suffixes we had to actively insert at the end of each step
        self.rule = 0 # the default rule

    def build_suffixTree(self):
        for i in range(len(self.seq)):
            self.rule = 0
            self.insert_suffixes(i)

    def insert_suffixes(self, start):
        self.parent_node = self.active_point.node
        if self.active_point.edge == None:
            if (self.active_point.node.id, self.seq[start]) in self.edge_dict:
                active_edge = self.edge_dict[(self.active_point.node.id, self.seq[start])]
                self.active_point.edge = active_edge
                self.active_point.length += 1
                self.reminder += 1
                if self.active_point.length == self.active_point.edge.length():
                    self.active_point.node = self.active_point.edge.end_node
                    self.active_point.edge = None
                    self.active_point.length = 0
                    self.parent_node = self.active_point.node
                return
        else:
            # If 'ab' is in the tree, every suffix of it must be in the tree.
            if self.seq[self.active_point.edge.start_char + self.active_point.length] == self.seq[start]:
                self.active_point.length += 1
                self.reminder += 1
                if self.active_point.length == self.active_point.edge.length():
                    self.active_point.node = self.active_point.edge.end_node
                    self.active_point.edge = None
                    self.active_point.length = 0
                    self.parent_node = self.active_point.node
                return
            else:
                # print "Splitting"
                new_node = Node(len(self.node_list), False)
                new_edge = Edge(self.active_point.edge.start_char + self.active_point.length, self.active_point.edge.end_char, new_node, self.active_point.edge.end_node)
                # print new_edge.explain(self.seq)
                self.node_list.append(new_node)
                self.edge_dict[(new_node.id, self.seq[self.active_point.edge.start_char + self.active_point.length])] = new_edge
                self.active_point.edge.update_end(self.active_point.edge.start_char + self.active_point.length, new_node)
                # print self.active_point.edge.explain(self.seq)
                self.parent_node = new_node
                # Apply Rule 1 when active_point.node is root
                if self.active_point.node == self.root:
                    self.rule = 1
                    self.active_point.length -= 1
                    self.reminder -= 1
                else: # Apply Rule 2 when active_point.node is not root
                    print self.active_point.node.id
                    if self.active_point.node.suffix_link != None:
                        self.active_point.node = self.active_point.node.suffix_link
                    else:
                        self.active_point.node = self.root
                    self.reminder -= 1
                    self.rule = 2
                if self.suffix_link != None:
                    self.suffix_link.suffix_link = new_node
                    # print "suffix_link %d --> %d" % (self.suffix_link.id, new_node.id)
                self.suffix_link = new_node

        new_node = Node(len(self.node_list))
        new_edge = Edge(start, self.size, self.parent_node, new_node)
        self.node_list.append(new_node)
        self.edge_dict[(self.parent_node.id, self.seq[start])] = new_edge
        # print new_edge.explain(self.seq)
        if self.rule != 0:
            if self.reminder > 1:
                if self.rule == 1:
                    self.active_point.edge = self.edge_dict[(self.active_point.node.id, self.seq[start - self.reminder + 1])]
                    self.insert_suffixes(start)
                else:
                    self.active_point.edge = self.edge_dict[(self.active_point.node.id, self.seq[self.active_point.edge.start_char])]
                    self.insert_suffixes(start)
            else:
                self.rule = 0
                self.active_point.node= self.root
                self.active_point.edge = None
                self.insert_suffixes(start)
                self.suffix_link = None # reset suffix_link 
                assert self.active_point.length == 0
                assert self.active_point.node.id == 0
                assert self.reminder == 1

                

    def draw_graph(self, fn):
        fp = open(fn, 'wb')
        fp.write("digraph G {\n")
        for node in sorted(self.node_list):
            fp.write(node.explain())
        
        for edge in sorted(self.edge_dict.values()):
            fp.write(edge.explain(self.seq))
            fp.write('\n')
        fp.write('}')
        fp.close()
        cmd = "dot -Tpng %s > %s.png" % (fn, fn)
        os.system(cmd)

if __name__ == "__main__":
    suffixtree = SuffixTree("abcabxabcdabce")
    #suffixtree = SuffixTree("banana$")
    suffixtree.build_suffixTree()
    suffixtree.draw_graph("naive_suffix_tree")
        
        
        
