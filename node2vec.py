import logging
import numpy as np
import joblib
import pickle
import os
from multiprocessing.dummy import Pool as ThreadPool
import itertools
import alias_method

import time


def process_node(node_id, out_degree):
    nbrs_count = out_degree
    normalized_probs = nbrs_count * [1.0 / nbrs_count] if nbrs_count > 0 else []
    return node_id, alias_method.alias_setup(normalized_probs)


class Graph:
    def __init__(self, gt_g, is_directed, p, q, workers=2, alias_nodes_file=None, alias_edges_file=None):
        self.G = gt_g
        self.is_directed = is_directed
        self.p = p
        self.divp = 1.0 / self.p
        self.q = q
        self.divq = 1.0 / self.q
        self.workers = workers
        self.alias_nodes_file = 'alias_nodes.pickle' if alias_nodes_file is None else alias_nodes_file
        self.alias_edges_file = 'alias_edges.pickle' if alias_edges_file is None else alias_edges_file
        self.alias_edges_file_temp = 'alias_edges_{}.pickle'
        self.alias_nodes = None
        self.alias_edges = None
        if os.path.isfile(self.alias_nodes_file):
            self.alias_nodes = joblib.load(self.alias_nodes_file)
        if os.path.isfile(self.alias_edges_file):
            self.alias_edges = joblib.load(self.alias_edges_file)

    def node2vec_walk(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        """

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = self.G.get_out_neighbours(cur)
            if len(cur_nbrs) == 0:  # reached a dead end - finish the walk
                break
            if len(walk) == 1:
                alias_item = self.alias_nodes[cur]
            else:
                prev = walk[-2]
                alias_item = self.alias_edges[(prev, cur)]
            next_v = cur_nbrs[alias_method.alias_draw(alias_item[0], alias_item[1])]
            walk.append(next_v)

        return walk

    def simulate_walks(self, num_walks, walk_length):
        """
        Repeatedly simulate random walks from each node.
        """

        walks = []
        nodes = list(self.G.get_vertices())
        print('Walk iteration:')
        # TODO: parallelize
        for walk_iter in range(num_walks):
            logging.info(str(walk_iter + 1), '/', str(num_walks))
            np.random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def preprocess_transition_probs(self):
        """
        Calculates transition probabilities for nodes and edges, if necessary 
        """
        if self.alias_nodes is None:
            self.preprocess_node_transition_probs()
        if self.alias_edges is None:
            self.preprocess_edge_transition_probs()

    def preprocess_node_transition_probs(self):
        """
        Preprocessing of node transition probabilities for guiding the random walks.
        """
        logging.info('starting to preprocess node transitions')
        start = time.time()
        alias_nodes = joblib.Parallel(n_jobs=self.workers)(
            joblib.delayed(process_node)(int(node), node.out_degree()) for node in self.G.vertices())
        alias_nodes = dict(alias_nodes)
        self.alias_nodes = dict(alias_nodes)
        end = time.time()
        logging.info('preprocessing node transitions took {}'.format(end - start))
        joblib.dump(self.alias_nodes, self.alias_nodes_file)

    def preprocess_edge_transition_probs(self):
        """
        Preprocessing of edge transition probabilities for guiding the random walks.
        """
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
