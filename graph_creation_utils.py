import numpy as np
import csv
import logging
import json
import pickle
import math

import graph_tool
import graph_tool.generation
import graph_tool.draw
import graph_tool.topology
import py2neo
import time
import unidecode
import pandas
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO
                    # , filename='log.log', filemode='w'
                    )


def mark_events_neo4j(neo4j):
    # load the events titles
    events = []
    with open('wiki_events2.json', 'r') as file:
        data = json.load(file)
        for title in data:
            if title:
                events.append(title)

    for event in events:
        if event is None:
            continue
        result = neo4j.data("MATCH (p:Page) WHERE p.title = {title} "
                            "SET p :Event "
                            "RETURN p.title AS title",
                            {"title": event})
        if len(result) == 1:
            logging.info("{}  {}".format(len(result), unidecode.unidecode(result[0]['title'])))
        else:
            logging.error('!!!!!!!!')
            print("{}: {}".format(event, len(result)))


def mark_events_csv(csv_name):
    # load the events
    new_rows = []
    with open('wiki_events.json', 'r', encoding='utf8') as events_file:
        events = json.load(events_file)
    # read all nodes
    with open(csv_name, 'r', encoding='utf8') as file:
        csv_in = csv.reader(file)
        for node_row in csv_in:
            if node_row[1] in events:
                node_row.append('True')
            new_rows.append(node_row)
    # write all (new) rows
    with open(csv_name, 'w', encoding='utf8', newline='') as file:
        out = csv.writer(file)
        for node_row in new_rows:
            out.writerow(node_row)


def neo4j_remove_titles(neo4j):
    start = time.time()
    neo4j.run("MATCH (p:Page) REMOVE p.title")
    end = time.time()
    print('neo4j query took {}'.format(end - start))


def neo4j_to_nodes_dictionary(neo4j):
    start = time.time()
    cursor = neo4j.run("MATCH (p:Page) RETURN ID(p) as pid, p.title AS title")
    end = time.time()
    print('neo4j query took {}'.format(end - start))
    dict = {}
    start = time.time()
    while cursor.forward():
        dict[cursor.current()['pid']] = cursor.current()['title']
    end = time.time()
    print('building the dictionary took {}'.format(end - start))
    with open('mycsvfile.csv', 'w', newline='', encoding='utf8') as f:
        w = csv.writer(f)
        w.writerows(dict.items())


def create_gt_from_edgelist(edgelist_file):
    g = graph_tool.Graph()
    df = pandas.read_csv(edgelist_file)
    edge_list = df.values
    g.add_edge_list(edge_list)
    # with open(edgelist_file, 'r', encoding='utf8') as nodes_file:
    #     csv_in = csv.reader(nodes_file)
    #     for edge_row in csv_in:
    #         node1 = int(edge_row[0])
    #         node2 = int(edge_row[1])
    #         largest_node = max(node1, node2)
    #         if largest_node >= g.num_vertices():  # add the new/missing vertices
    #             g.add_vertex(largest_node - g.num_vertices() + 1)
    #         g.add_edge(node1, node2)
    g.save('wiki_unfilled.gt')
    return g


def fill_gt(g, nodes_list):
    """
    fills a given graph with titles and node type from an external csv file
    :param g: a graph of node IDs only
    """
    title_prop = g.new_vertex_property('string')
    event_prop = g.new_vertex_property('bool')
    # load the nodes
    start = time.time()
    with open(nodes_list, 'r', encoding='utf8') as nodes_file:
        csv_in = csv.reader(nodes_file)
        for node_row in csv_in:
            node_id = int(node_row[0])
            title = node_row[1]
            title_prop[g.vertex(node_id, add_missing=True)] = title
            if len(node_row) > 2 and node_row[2] == 'True':  # for each event, add a property
                event_prop[g.vertex(node_id)] = True
    end = time.time()
    print('adding the properties took {}'.format(end - start))
    # internalize both properties
    g.vertex_properties['title'] = title_prop
    g.vertex_properties['event'] = event_prop
    # save the updated graph
    start = time.time()
    g.save('wiki.gt')
    end = time.time()
    print('saving the graph took {}'.format(end - start))


def create_filtered_graph(g, max_dist, save=True):
    """
    create a filtered view of g, with vertices that are close to an event node
    :param g: a Graph
    :param max_dist: the max distance to an event node
    :return: the filtered graph
    """
    event_i = 0
    # create a distance map of each vertex to its closest event vertex
    best_dist_map = np.full(g.num_vertices(), g.num_vertices())
    for v in g.vertices():
        if not g.vp.event[v]:
            continue
        if event_i % 1000 == 0:
            logging.info('starting event #{}'.format(event_i))
        # add v's neighborhood to the map
        dist_map = graph_tool.topology.shortest_distance(g, source=v, max_dist=max_dist)
        best_dist_map = np.minimum(dist_map.a, best_dist_map)
        event_i += 1
    logging.info('done processing nodes, will create the filtered view and save')
    # create a filtered view of the graph, with only the vertices that are close to events
    new_g = graph_tool.GraphView(g, vfilt=best_dist_map <= max_dist)
    # new_g.purge_vertices()
    new_g = graph_tool.Graph(new_g, prune=True)
    # print(new_g.num_vertices())
    if save:
        new_g.save('wiki_filtered_{}.gt'.format(max_dist))
        logging.info('saved the filtered graph, which contains {} vertices, {} edges'.format(new_g.num_vertices(),
                                                                                             new_g.num_edges()))
    return new_g


def copy_neighbors(g2, g, v, title_prop, dist):
    """

    :param g2: new graph
    :param v: index of a vertex to copy
    :param dist: if dist>0, will continue copying recursively
    """
    logging.info('before: contains {} vertices, {} edges'.format(g2.num_vertices(),
                                                                 g2.num_edges()))
    logging.info('{} has {} neighbors'.format(g.vp.title[v], v.out_degree()))
    for u in v.out_neighbours():
        g2.add_edge(v, u)
        title_prop[u] = g.vp.title[u]
        if dist > 0:
            copy_neighbors(g2, g, u, title_prop, dist=dist - 1)
    logging.info('after: contains {} vertices, {} edges'.format(g2.num_vertices(),
                                                                g2.num_edges()))


def create_filtered_graph_manually(g, max_dist, save=True, events_limit=math.inf):
    """
    create a filtered view of g, with vertices that are close to an event node
    :param g: a Graph
    :param max_dist: the max distance to an event node
    :return: the filtered graph
    """
    event_i = 0
    g2 = graph_tool.Graph()

    title_prop = g2.new_vertex_property('string')
    # event_prop = g2.new_vertex_property('bool')
    for v in g.vertices():
        if event_i > events_limit:
            break
        if not g.vp.event[v]:
            continue
        if event_i % 10 == 0:
            logging.info('starting event #{}'.format(event_i))
        copy_neighbors(g2, g, v, title_prop, max_dist - 1)
        # title_prop[v] = g.vp.title[v]
        event_i += 1
    # internalize both properties
    # g2.vertex_properties['title'] = title_prop
    # g2.vertex_properties['event'] = event_prop
    logging.info('done processing nodes, will save')
    if save:
        g2.save('wiki_filtered_{}_manual_limit{}.gt'.format(max_dist, events_limit))
        logging.info('saved the filtered graph, which contains {} vertices, {} edges'.format(g2.num_vertices(),
                                                                                             g2.num_edges()))
    return g2


def test_filter_graph():
    def sample_k(max):
        accept = False
        while not accept:
            k = np.random.randint(1, max + 1)
            accept = np.random.random() < 1.0 / k
        return k

    g = graph_tool.generation.random_graph(10, lambda: sample_k(3), directed=False)
    print('{} vertices, {} edges'.format(g.num_vertices(), g.num_edges()))
    graph_tool.draw.graph_draw(g, vertex_text=g.vertex_index)
    event_prop = g.new_vertex_property('bool')
    # for v in g.vertices():
    #     event_prop[v] = False
    event_prop[g.vertex(0)] = True
    event_prop[g.vertex(1)] = True
    # internalize both properties
    g.vertex_properties['event'] = event_prop
    # create a filtered view of the graph, with only the vertices that are close to events
    g = create_filtered_graph(g, 1, save=False)
    graph_tool.draw.graph_draw(g, vertex_text=g.vertex_index)
    # g = graph_tool.load_graph('wiki_filtered_1.gt')
    # graph_tool.draw.graph_draw(g, vertex_text=g.vertex_index)


def create_graph(edgelist_file, nodes_file):
    start = time.time()
    g = create_gt_from_edgelist(edgelist_file)
    end = time.time()
    print('creating the graph took {}'.format(end - start))
    fill_gt(g, nodes_file)
    return g


def remove_blank_lines(csv_name):
    new_rows = []
    # read all nodes
    with open(csv_name, 'r', encoding='utf8') as file:
        csv_in = csv.reader(file)
        for node_row in csv_in:
            if node_row:
                new_rows.append(node_row)
    # write all (new) rows
    with open(csv_name, 'w', encoding='utf8', newline='') as file:
        out = csv.writer(file)
        for node_row in new_rows:
            out.writerow(node_row)


def export_edgelist(neo4j):
    start = time.time()
    cursor = neo4j.run("MATCH (p1:Page)-[:Link]-(p2:Page) WHERE ID(p1)>=7000000 RETURN ID(p1) as id1, ID(p2) as id2")
    end = time.time()
    print('neo4j query took {}'.format(end - start))
    edges = []
    start = time.time()
    while cursor.forward():
        edges.append([cursor.current()['id1'], cursor.current()['id2']])
    end = time.time()
    print('building the dictionary took {}'.format(end - start))
    with open('edges_7m_to_end.csv', 'w', newline='', encoding='utf8') as f:
        w = csv.writer(f)
        w.writerows(edges)


def print_neighbors(g, v_name, nodes_dict):
    v = g.vertex(nodes_dict[v_name])
    for u in v.out_neighbours():
        print(g.vp.title[u])


def test(g):
    # if not os.path.exists('node_name_to_id.db'):
    node_name_to_id = {}
    # with open('nodes.csv', 'r', encoding='utf8') as file:
    #     csv_in = csv.reader(file)
    #     for node_row in csv_in:
    #         node_name_to_id[node_row[1]] = node_row[0]
    for v in g.vertices():
        node_name_to_id[g.vp.title[v]] = g.vertex_index[v]
    with open('node_name_to_id.db', 'wb') as f:
        pickle.dump(node_name_to_id, f)
    # else:
    #     with open('node_name_to_id.db', 'rb') as f:
    #         node_name_to_id = pickle.load(f)
    query = 'Khataba raid'
    while True:
        print('neighbors of {}:'.format(query))
        print_neighbors(g, query, node_name_to_id)
        # g2 = graph_tool.load_graph('wiki_filtered_2.gt')
        # print('wiki_filtered_2:')
        # print_neighbors(g2, query, node_name_to_id)
        # g3 = graph_tool.load_graph('wiki_filtered_2_pruned.gt')
        # print('wiki_filtered_2_pruned:')
        # print_neighbors(g3, query, node_name_to_id)
        query = input('enter a page title')
        # nodes_db.close()


def clean_node_names():
    def clean(s):
        return s.replace('"', '').replace(' ', '_').lower()

    node_id_to_name = {}
    node_name_to_id = {}
    with open('data/nodes.csv', 'r', encoding='utf8', newline='') as file:
        with open('data/nodes_clean.csv', 'w', encoding='utf8', newline='') as newfile:
            csv_in = csv.reader(file)
            writer = csv.writer(newfile)
            for node_row in csv_in:
                node_row[1] = clean(node_row[1])
                writer.writerow(node_row)
                node_id_to_name[node_row[0]] = node_row[1]
                node_name_to_id[node_row[1]] = node_row[0]
    with open('data/node_id_to_name_clean.db', 'wb') as f:
        pickle.dump(node_id_to_name, f)
    with open('data/node_name_to_id_clean.db', 'wb') as f:
        pickle.dump(node_name_to_id, f)


def w2v_model_inject_names(model_name):
    with open('data/node_id_to_name_clean.db', 'rb') as f:
        node_id_to_name = pickle.load(f)

    with open('data/{}.emb'.format(model_name), 'r', encoding='utf8', newline='') as file:
        with open('data/{}.model'.format(model_name), 'w', encoding='utf8', newline='') as new_file:
            reader = csv.reader(file, delimiter=' ')
            writer = csv.writer(new_file, delimiter=' ')
            writer.writerow(next(reader))
            vocab_len = 0
            invalid_count = 0
            for node_row in reader:
                if len(node_row) == 129 and len(node_row[0]) > 0:
                    try:
                        node_id = node_row[0]
                        node_row[0] = node_id_to_name[node_id]
                        writer.writerow(node_row)
                        vocab_len += 1
                    except:
                        logging.exception('exception in line: {}'.format(node_row))
                else:
                    # logging.error('invalid line: {}'.format(node_row))
                    invalid_count += 1
            print('number of invalid rows: {}'.format(invalid_count))
            print('vocabulary length: {}. Remember to edit the first line with it!'.format(vocab_len))
            # sed -i 's/%d+ /{vocab_len}' {filename} (vocab_len=6572500)


def write_graph_info():
    logging.info('loading graphs...')
    g = graph_tool.load_graph('wiki_FULL.gt')
    logging.info('wiki_FULL graph: {} vertices, {} edges'.format(g.num_vertices(), g.num_edges()))
    # g = graph_tool.load_graph('wiki_filtered_1.gt')
    # logging.info('filtered_1 graph: {} vertices, {} edges'.format(g.num_vertices(), g.num_edges()))
    g = graph_tool.load_graph('wiki_filtered_2.gt')
    logging.info('filtered_2 graph: {} vertices, {} edges'.format(g.num_vertices(), g.num_edges()))
    g = graph_tool.load_graph('wiki_filtered_3.gt')
    logging.info('filtered_3 graph: {} vertices, {} edges'.format(g.num_vertices(), g.num_edges()))


if __name__ == '__main__':
    # g = graph_tool.load_graph('wiki_filtered_2.gt')
    # new_g = graph_tool.Graph(g, prune=True)
    # logging.info('filtered_2 graph: {} vertices, {} edges'.format(new_g.num_vertices(), new_g.num_edges()))
    # new_g.save('wiki_filtered_{}_pruned.gt'.format(2))

    # neo4j = py2neo.Graph(user='neo4j', password='123')
    # mark_events_neo4j(neo4j)
    # neo4j_remove_titles(neo4j)
    # neo4j_to_nodes_dictionary(neo4j)

    # remove_blank_lines('nodes.csv')
    # mark_events_csv('nodes.csv')
    # export_edgelist(neo4j)
    # g = create_graph('edgeslist.csv', 'nodes.csv')
    # test_filter_graph()
    # g = graph_tool.load_graph('wiki_FULL.gt')
    # exit()

    # g = graph_tool.load_graph('wiki_filtered_2.gt')
    # logging.info('loaded the graph, with {} vertices and {} edges'.format(g.num_vertices(), g.num_edges()))
    # new_g = create_filtered_graph_manually(g, 1, events_limit=1)
    # new_g = create_filtered_graph(g, 2)
    # new_g = create_filtered_graph(g, 3)
    #
    # write_graph_info()

    # g = graph_tool.load_graph('wiki_filtered_2_pruned.gt')
    # logging.info('wiki_filtered_2_pruned contains: {} vertices, {} edges'.format(g.num_vertices(), g.num_edges()))
    # g1 = create_filtered_graph(g, 1)
    # test(g1)
    clean_node_names()
    w2v_model_inject_names('wiki_walksnum4_walklength30')
