import logging
import time
import numpy as np
import joblib
import pickle
import os
from multiprocessing import Pool

import graph_utils


class Graph:
    def __init__(self, gt_g, is_directed, p, q, workers=2, alias_nodes_file=None, alias_edges_file=None,
                 neighbors_dict_file=None):
        self.G = gt_g
        self.is_directed = is_directed
        self.p = p
        self.divp = 1.0 / self.p
        self.q = q
        self.divq = 1.0 / self.q
        self.workers = workers
        self.alias_nodes_file = 'data/alias_nodes.pickle' if alias_nodes_file is None else alias_nodes_file
        self.alias_edges_file = 'data/alias_edges.pickle' if alias_edges_file is None else alias_edges_file
        self.neighbors_dict_file = 'data/neighbors_dict.pickle' if neighbors_dict_file is None else neighbors_dict_file
        # self.alias_edges_file_temp = 'alias_edges_{}.pickle'
        self.alias_nodes = None
        self.alias_edges = None
        self.neighbors_dict = None
        if os.path.isfile(self.alias_nodes_file):
            self.alias_nodes = joblib.load(self.alias_nodes_file)
        if os.path.isfile(self.alias_edges_file):
            self.alias_edges = joblib.load(self.alias_edges_file)
        if os.path.isfile(self.neighbors_dict_file):
            with open(self.neighbors_dict_file, 'rb') as f:
                self.neighbors_dict = pickle.load(f)

    def node2vec_walk(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        """

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = self.neighbors_dict[cur]
            if len(cur_nbrs) == 0:  # reached a dead end - finish the walk
                break
            degree = self.alias_nodes[cur]
            # if len(walk) == 1:
            #     alias_item = self.alias_nodes[cur]
            # else:
            #     prev = walk[-2]
            #     alias_item = self.alias_edges[(prev, cur)]
            # next_v = cur_nbrs[alias_method.alias_draw(alias_item[0], alias_item[1])]

            # draw the next neighbor (uniformly)
            next_neighbor_i = int(np.floor(np.random.rand() * degree))

            next_v = cur_nbrs[next_neighbor_i]
            walk.append(next_v)

        return walk

    def simulate_walks_iteration(self, walk_iter, permuted_nodes, walk_length):
        logging.info('Walk iteration #{}'.format(walk_iter + 1))
        walks = [self.node2vec_walk(walk_length=walk_length, start_node=node) for node in permuted_nodes]
        with open('data/walk_iter_{}.pickle'.format(walk_iter + 1), 'wb') as f:
            pickle.dump(walks, f, protocol=4)
        return walks

    def simulate_walks(self, num_walks, walk_length):
        """
        Repeatedly simulate random walks from each node.
        """

        nodes = list(self.G.get_vertices())
        logging.info('Simulating {} walks.'.format(num_walks))
        # multiprocessing
        # pool = Pool(processes=4)
        # start = time.time()
        # parameters = [(walk_iter, np.random.permutation(nodes), walk_length) for walk_iter in range(num_walks)]
        # end = time.time()
        # logging.info('Building the parameters took {}. Now running the simulation...'.format(end - start))
        # walks_result = list(pool.starmap(self.simulate_walks_iteration, parameters))
        # pool.close()
        # pool.join()
        # # flatten the result (each process returned a list of walks)
        # walks = [val for sublist in walks_result for val in sublist]

        # multiprocessing over chunks (process x walks concurrently)
        # walks = []
        # chunk_size = 2
        # pool = Pool(processes=4)
        # for chunk in graph_utils.chunks(range(num_walks), chunk_size):
        #     start = time.time()
        #     parameters = [(walk_iter, np.random.permutation(nodes), walk_length) for walk_iter in chunk]
        #     end = time.time()
        #     logging.info('Building the parameters (for a chunk of {}) took {}. Now running the simulation...'
        #                  .format(chunk_size, end - start))
        #     walks_result = list(pool.starmap(self.simulate_walks_iteration, parameters))
        #     pool.close()
        #     pool.join()
        #     # flatten the result (each process returned a list of walks)
        #     walks.extend([val for sublist in walks_result for val in sublist])

        # sequential
        walks = []
        for walk_iter in range(num_walks):
            start = time.time()
            permuted_nodes = np.random.permutation(nodes)
            end = time.time()
            logging.info('nodes permutation took {}. Now running the simulation...'.format(end - start))
            walks.extend(self.simulate_walks_iteration(walk_iter, permuted_nodes, walk_length))

        return walks

    def preprocess_transition_probs(self):
        """
        Calculates transition probabilities for nodes and edges, if necessary 
        """
        if self.alias_nodes is None:
            self.preprocess_node_transition_probs()
        if self.neighbors_dict is None:
            self.neighbors_dict = graph_utils.get_all_neighbors(self.G)
            with open(self.neighbors_dict_file, 'wb') as f:
                pickle.dump(self.neighbors_dict, f)
                # if self.alias_edges is None:
                #     self.preprocess_edge_transition_probs()

    def preprocess_node_transition_probs(self):
        """
        Preprocessing of node transition probabilities for guiding the random walks.
        """
        logging.info('starting to preprocess node transitions')
        start = time.time()
        self.alias_nodes = {int(node): node.out_degree() for node in self.G.vertices()}
        end = time.time()
        logging.info('preprocessing node transitions took {}'.format(end - start))
        joblib.dump(self.alias_nodes, self.alias_nodes_file)


"""
    def preprocess_edge_transition_probs(self):
        '''
Preprocessing of edge transition probabilities for guiding the random walks.
'''
logging.info('starting to preprocess edge transitions')
# remember that for an undirected graph, process_edge() may need to be applied also on the opposite edge
start = time.time()
pool = ThreadPool(self.workers)
edges = self.G.get_edges()
end = time.time()
logging.info('getting all edges took {}'.format(end - start))
# interval = 100 * 10 ^ 6
interval = int(len(edges) / 40)

# def grouper(n, iterable, fillvalue=None):
#    args = [iter(iterable)] * n
#    return itertools.zip_longest(*args, fillvalue=fillvalue)

i = 1
while i < len(edges):
    edges_chunk = edges[(i - 1) * interval: i * interval]
    start = time.time()
    alias_edges = pool.map(self.get_alias_edge_worker, edges_chunk)
    end = time.time()
    logging.info('preprocessing edge transitions #{} took {}'.format(i, end - start))
    alias_edges = dict(alias_edges)
    joblib.dump(alias_edges, self.alias_edges_file_temp.format(i))
    i += 1

    logging.info('done calculating all the edges aliases. will now merge them.')
self.alias_edges = {}
for j in range(i):
    start = time.time()
    alias_edges_chunk = joblib.load(self.alias_edges_file_temp.format(j + 1))
    self.alias_edges.update(alias_edges_chunk)
    end = time.time()
    print('merged chunk {} in {}'.format(j + 1, end - start))

def get_alias_edge_worker(self, edge):
return (edge[0], edge[1]), self.get_alias_edge(edge[0], edge[1])

def get_alias_edge(self, src, dst):
'''
Get the alias edge setup lists for a given edge.
"how do we continue the walk after this edge?"
src, dst are node indices
'''

unnormalized_probs = []
for dst_nbr in self.G.get_out_neighbours(dst):
    if dst_nbr == src:
        unnormalized_probs.append(self.divp)
    elif src in self.G.get_out_neighbours(dst_nbr):  # if there's an edge dst->src
        unnormalized_probs.append(1.0)
    else:
        unnormalized_probs.append(self.divq)
norm_const = sum(unnormalized_probs)
normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

return alias_setup(normalized_probs)
"""
