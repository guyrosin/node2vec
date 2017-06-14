import logging
import json
from py2neo import Graph
import unidecode

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                    filename='log.log', filemode='w'
                    )

# load the events titles
events = []
with open('wiki_events2.json', 'r') as file:
    data = json.load(file)
    for title in data:
        if title:
            events.append(title)

neo4j_mark_events = Graph(user='neo4j', password='123')

for event in events:
    if event is None:
        continue
    result = neo4j_mark_events.data("MATCH (p:Page) WHERE p.title = {title} "
                                    "SET p :Event "
                                    "RETURN p.title AS title",
                                    {"title": event})
    if len(result) == 1:
        logging.info("{}  {}".format(len(result), unidecode.unidecode(result[0]['title'])))
    else:
        logging.error('!!!!!!!!')
        print("{}: {}".format(event, len(result)))
