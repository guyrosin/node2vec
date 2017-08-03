import logging
import time


def get_all_neighbors(G):
    logging.info('starting to preprocess neighbors')
    start = time.time()
    neighbors_dict = {node: G.get_out_neighbours(node) for node in G.get_vertices()}
    end = time.time()
    logging.info('preprocessing neighbors took {}'.format(end - start))
    return neighbors_dict


def chunks(l, n):
    """ Yield successive n-sized chunks from l"""
    for i in range(0, len(l), n):
        yield l[i:i + n]
