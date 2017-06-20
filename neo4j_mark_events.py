import csv
import logging
import json
import py2neo
import time
import unidecode

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
    cursor = neo4j.run("MATCH (p1:Page)-[:Link]-(p2:Page) WHERE ID(p1)<5000000 RETURN ID(p1) as id1, ID(p2) as id2")
    end = time.time()
    print('neo4j query took {}'.format(end - start))
    edges = []
    start = time.time()
    while cursor.forward():
        edges.append([cursor.current()['id1'], cursor.current()['id2']])
    end = time.time()
    print('building the dictionary took {}'.format(end - start))
    with open('edges.csv', 'w', newline='', encoding='utf8') as f:
        w = csv.writer(f)
        w.writerows(edges)


if __name__ == '__main__':
    neo4j = py2neo.Graph(user='neo4j', password='123')
    # mark_events_neo4j(neo4j)

    # remove_blank_lines('nodes.csv')
    # mark_events_csv('nodes.csv')
    export_edgelist(neo4j)
