"""
Code for Project 2
"""

import timeit
import matplotlib.pyplot as plt
import random

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
def UPA(n, m):
    """
    undirected graph
    return a dictionary, the keys of which being
    the nodes and the values of which being the
    out-edges generated randomly by DPATrial
    n nodes, m the average # of edges for a given node
    """
    if type(m) is int:
        dic = make_complete_graph(m)
        UPA_trial = UPATrial(m)
        for i in range(m, n):
            #since the average edges for a graph is 2.5,
            #this optimize the simulation
            neighbors = UPA_trial.run_trial(m)
            dic[i] = neighbors
            #change the neigbors' neighbors
            for item in neighbors:
                dic[item].add(i)
        return dic
    elif type(m) is float:
        #find the int and float part of the value m
        #match just the tenth digit, ignore all the rest
        m_int = math.floor(m)
        m_float = int(re.findall('(?<=[.])\d',str(m))[0])
        #make the list so that the expected value of the choice is m
        choose_list = [(m_int+1) for i in range(m_float)] + [(m_int) for i in range(10-m_float)]

        dic = make_complete_graph(m_int)
        UPA_trial = UPATrial(m_int)
        for i in range(m_int, n):
            k = random.choice(choose_list)
            neighbors = UPA_trial.run_trial(k)
            dic[i] = neighbors
            for item in neighbors:
                dic[item].add(i)
        return dic
class UPATrial:
    """
    undirected graph

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
        self._node_numbers += [self._num_nodes for i in range(len(new_node_neighbors))]
        self._node_numbers.extend(list(new_node_neighbors))

        #update the number of nodes
        self._num_nodes += 1
        return new_node_neighbors

def copy_graph(graph):
    """
    Make a copy of a graph
    """
    new_graph = {}
    for node in graph:
        new_graph[node] = set(graph[node])
    return new_graph
def delete_node(ugraph, node):
    """
    Delete a node from an undirected graph
    """
    neighbors = ugraph[node]
    ugraph.pop(node)
    for neighbor in neighbors:
        ugraph[neighbor].remove(node)
def targeted_order(ugraph):
    """
    Compute a targeted attack order consisting
    of nodes of maximal degree

    Returns:
    A list of nodes
    """
    # copy the graph
    new_graph = copy_graph(ugraph)

    order = []
    while len(new_graph) > 0:
        max_degree = -1
        for node in new_graph:
            if len(new_graph[node]) > max_degree:
                max_degree = len(new_graph[node])
                max_degree_node = node

        neighbors = new_graph[max_degree_node]
        new_graph.pop(max_degree_node)
        for neighbor in neighbors:
            new_graph[neighbor].remove(max_degree_node)

        order.append(max_degree_node)
    return order

def find_degree(graph, node):
    """
    if the graph is undirected, but the edges aren't
    correct, use this function
    """
    count = 0
    for item in graph[node]:
        if item in graph:
            count += 1
    return count
def fast_targeted_order(ugraph):
    """
    return a list of nodes, each of
    which have the highest number of nodes
    among the remaining nodes
    """
    # copy the graph
    graph = copy_graph(ugraph)

    #create a list of sets such that kth set
    #contains the nodes that have degree k
    n = len(graph)
    degree_set_lst = [set([]) for i in range(n)]
    for key in graph.keys():
        degree = len(graph[key])
        degree_set_lst[degree].add(key)

    #initialize the node list
    final_node_lst = []

    #for all the nodes in the sets of degree_set_lst,
    #update its neighbors' degree in degree_set_lst;
    #put it in the node list and remove it from graph
    for i in range(n-1,-1,-1):
        while len(degree_set_lst[i]) != 0:
            current_node = list(degree_set_lst[i])[0]
            degree_set_lst[i].remove(current_node)
            for neighbor in graph[current_node]:

                #the neighbors' neighbors might already be removed,
                #so address this issue
                if neighbor not in graph:
                    continue
                #can't think of a better way
                degree = find_degree(graph, neighbor)

                #change the neighbors position in the degree_set_lst
                degree_set_lst[degree].remove(neighbor)
                degree_set_lst[degree-1].add(neighbor)

            final_node_lst.append(current_node)
            graph.pop(current_node)
    return final_node_lst

trial = list(range(10,1000,10))
time_lst_FTO = []
time_lst_TO = []

#using UPA to test the running time of the two algorithms
for n in trial:
    upa = UPA(n, 5)
    start1 = timeit.default_timer()
    fast_targeted_order(upa)
    stop1 = timeit.default_timer() - start1
    time_lst_FTO.append(stop1)

    start2 = timeit.default_timer()
    targeted_order(upa)
    stop2 = timeit.default_timer() - start2
    time_lst_TO.append(stop2)

plt.plot(trial, time_lst_FTO, label = 'fast_targeted_order')
plt.plot(trial, time_lst_TO, label = 'targeted_order')
plt.title('targeted_order & fast_targeted_order Running Time Comparison')
plt.xlabel('# of Nodes (m=5)')
plt.ylabel('Running Time (seconds)')
plt.legend()
plt.show()
