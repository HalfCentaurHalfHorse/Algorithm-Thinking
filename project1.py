"""
Code for Project 1
"""
# general imports
import urllib.request, urllib.parse, urllib.error
import matplotlib.pyplot as plt
import random

# Set timeout for CodeSkulptor if necessary
#import codeskulptor
#codeskulptor.set_timeout(20)


###################################
# Code for loading citation graph, note that we are dealing with directed graph

CITATION_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_phys-cite.txt"

def load_graph(graph_url):
    """
    Function that loads a graph given the URL
    for a text representation of the graph

    Returns a dictionary that models a graph
    """
    graph_file = urllib.request.urlopen(graph_url)
    dic = {}
    for line in graph_file:
        lst = line.decode().split()
        lst = [int(i) for i in lst]

        #the first item is the node, all the rest are the neighbors
        dic[lst[0]] = set(lst[1:])

    print(len(dic))
    return dic

def compute_in_degrees(digraph):
    """
    return a dictionary the keys of which being
    the nodes and the values of which being the
    in-degree of the nodes
    """
    dic = {}
    for key in digraph:
        dic[key] = 0
    for value in digraph.values():
        for item in value:
            if item not in dic.keys():
                dic[item] = 0
            dic[item] += 1
    return dic

def in_degree_distribution(digraph, normalized = False):
    """
    return a dictionary, the keys of which being
    the in-degrees and the values of which being the
    number of nodes with that in-degree
    """
    in_degree_dic = compute_in_degrees(digraph)
    dis_dic = {}
    for value in in_degree_dic.values():
        if value not in dis_dic:
            dis_dic[value] = 0
        dis_dic[value] += 1

    #normalize the values, that is, we have values such that
    #their sum would be 1
    if normalized == True:
        count = dis_dic.values()
        total = sum(count)
        for key in dis_dic:
            dis_dic[key] = dis_dic[key] / total
    return dis_dic

def ER(num, prob):
    """
    directed graph
    return a distionary, the key of which is the nodes
    and the value of which is a set that contains the out-edges
    picked randomly from all of the nodes based on probability prob
    """
    dic = {}
    for i in range(num):
        dic[i] = set([])
    for key in dic:
        for key2 in dic:
            if key2 == key: continue
            random_num = random.random()
            if random_num > prob:
                dic[key].add(key2)
    return dic

def make_complete_graph(num_nodes):
    """
    return a dictionary that represents the
    complete directed graph of num_nodes number
    of nodes
    """
    dic = {}
    if num_nodes <= 0:
        return dic
    for num in range(num_nodes):
        lst = [i for i in range(num_nodes)]
        lst.remove(num)
        dic[num] = set(lst)
    return dic

class DPATrial:
    """
    Simple class to encapsulate optimized trials for DPA algorithm

    Maintains a list of node numbers with multiple instances of each number.
    The number of instances of each node number are
    in the same proportion as the desired probabilities

    Uses random.choice() to select a node number from this list for each trial.
    """

    def __init__(self, num_nodes):
        """
        Initialize a DPATrial object corresponding to a
        complete graph with num_nodes nodes

        Note the initial list of node numbers has num_nodes copies of
        each node number
        """
        self._num_nodes = num_nodes
        self._node_numbers = [node for node in range(num_nodes) for dummy_idx in range(num_nodes)]


    def run_trial(self, num_nodes):
        """
        Conduct num_node trials using by applying random.choice()
        to the list of node numbers

        Updates the list of node numbers so that the number of instances of
        each node number is in the same ratio as the desired probabilities

        Returns:
        the set of nodes chosen as neighbors
        """

        # compute the neighbors for the newly-created node
        new_node_neighbors = set()
        for dummy_idx in range(num_nodes):
            new_node_neighbors.add(random.choice(self._node_numbers))

        # update the list of node numbers so that each node number
        # appears in the correct ratio
        self._node_numbers.append(self._num_nodes)
        self._node_numbers.extend(list(new_node_neighbors))

        #update the number of nodes
        self._num_nodes += 1
        return new_node_neighbors

def DPA(n, m):
    """
    directed graph
    return a dictionary, the keys of which being
    the nodes and the values of which being the
    out-edges generated randomly by DPATrial
    """
    dic = make_complete_graph(m)
    DPA_trial = DPATrial(m)
    for i in range(m, n):
        neighbors = DPA_trial.run_trial(m)
        dic[i] = neighbors
    return dic

citation_graph = load_graph(CITATION_URL)
distribution = in_degree_distribution(citation_graph, normalized = True)

names = distribution.keys()
values = distribution.values()
plt.plot(names, values, 'b.')
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.ylabel("Frequency (log scale)")
plt.xlabel("In-degree (log scale) ")
plt.title("Citation Graph (n=27770)")
plt.show()
