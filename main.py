import argparse
import graph_tool
import time
import pickle
import os
import logging

import node2vec
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO
                    # , filename='log.log', filemode='w'
                    )


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='data/wiki_filtered_2_pruned.gt',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='data/wiki.emb',
                        help='Embeddings path')

    parser.add_argument('--walks-dir', nargs='?', default=None,
                        help='Directory that contains walks files')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=20,  # was 80
                        help='Length of walk per source. Default is 60.')

    parser.add_argument('--num-walks', type=int, default=4,  # was 10
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=5,  # was 10
                        help='Context size for optimization. Default is 5.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--min-count', type=int, default=5,  # wasn't here originally (w2v's default is 5)
                        help='Minimum count for word2vec. Default is 5.')

    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers. Default is 4.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is directed.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=True)

    return parser.parse_args()


def read_graph(graph_file_path):
    '''
    Reads the input network in graph_tool, as a directed graph
    '''
    start = time.time()
    g = graph_tool.load_graph(graph_file_path)
    end = time.time()
    logging.info('Loaded {}, which contains: {} vertices, {} edges. It took {}'.format(graph_file_path,
                                                                                       g.num_vertices(), g.num_edges(),
                                                                                       end - start))
    return g


def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    start = time.time()
    walks = [list(map(str, walk)) for walk in walks]  # convert each node id to a string
    end = time.time()
    logging.info('converting the walks to strings took {}'.format(end - start))
    start = time.time()
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=args.min_count, sg=1,
                     workers=args.workers, iter=args.iter)
    end = time.time()
    logging.info('building the w2v model took {}'.format(end - start))
    model.wv.save_word2vec_format(args.output)


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    # walks_file_path = 'data/walks.pickle'
    if not all(os.path.isfile('{}/walk_iter_{}.pickle'.format(args.walks_dir, walk_iter + 1))
               for walk_iter in range(args.num_walks)):
        logging.info('loading the graph')
        gt_g = read_graph(args.input)
        G = node2vec.Graph(gt_g, args.directed, args.p, args.q, workers=args.workers)
        G.preprocess_transition_probs()
        start = time.time()
        walks = G.simulate_walks(args.num_walks, args.walk_length)
        end = time.time()
        logging.info('simulating walks took {}'.format(end - start))
        # with open(walks_file_path, 'wb') as f:
        #     pickle.dump(walks, f, protocol=4)
        # logging.info('saved the walks to a file.')
        # with open(walks_file_path, 'wb') as file_obj:
        # pickle.dump(walks, file_obj)
    else:
        logging.info('loading walks files')
        start = time.time()
        walks = []
        for walk_iter in range(args.num_walks):
            walks_file_path = '{}/walk_iter_{}.pickle'.format(args.walks_dir, walk_iter + 1)
            with open(walks_file_path, 'rb') as f:
                walks.extend(pickle.load(f))
            logging.info('done with walks file #{}'.format(walk_iter + 1))
        # with open(walks_file_path, 'rb') as file_obj:
        #     walks = pickle.load(file_obj)
        end = time.time()
        logging.info('loading walks file took {}'.format(end - start))
    learn_embeddings(walks)


if __name__ == "__main__":
    args = parse_args()
    args.walks_dir = 'walks_walksnum{}_walklength{}'.format(args.num_walks, args.walk_length) \
        if args.walks_dir is None else args.walks_dir
    logging.info('running node2vec with params: ')
    for k, v in sorted(vars(args).items()):
        logging.info('{}: {}'.format(k, v))
    main(args)
