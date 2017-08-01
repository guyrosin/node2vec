import logging
import pickle
import time


def save_neighbors(G, file_name):
    logging.info('starting to preprocess neighbors')
    start = time.time()
    neighbors_dict = {node: G.get_out_neighbours(node) for node in G.get_vertices()}
    end = time.time()
    logging.info('preprocessing neighbors took {}'.format(end - start))
    with open(file_name, 'wb') as f:
        pickle.dump(neighbors_dict, f)
    return neighbors_dict
