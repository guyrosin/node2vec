import csv
import logging
import json

import graph_tool
import py2neo
import time
import unidecode
import pandas

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING,
                    filename='log.log', filemode='w'
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


if __name__ == '__main__':
    # neo4j = py2neo.Graph(user='neo4j', password='123')
    # mark_events_neo4j(neo4j)
    # neo4j_remove_titles(neo4j)
    # neo4j_to_nodes_dictionary(neo4j)

    # remove_blank_lines('nodes.csv')
    # mark_events_csv('nodes.csv')
    # export_edgelist(neo4j)
    g = create_graph('edgeslist.csv', 'nodes.csv')
    # g = graph_tool.load_graph('wiki.gt')
