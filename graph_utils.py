import logging
import pickle
import time

import os


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


def convert_walks_numpy_to_txt(walks_dir):
    walks_files = filter(lambda file: file.endswith('.pickle'), os.listdir(walks_dir))
    for file in walks_files:
        start = time.time()
        with open(os.path.join(walks_dir, file), 'rb') as f:
            walks = pickle.load(f)
        end = time.time()
        logging.info('loading walks from {} took {}'.format(file, end - start))
        start = time.time()
        walks = [','.join(map(str, walk)) for walk in walks]  # convert each node id to a string
        end = time.time()
        logging.info('converting walks to strings took {}'.format(end - start))
        with open(os.path.join(walks_dir, file + '.txt'), 'w') as f:
            f.write('\n'.join(walks))
        logging.info('done with walks file: {}'.format(file))
