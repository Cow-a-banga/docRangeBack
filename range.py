import math
from typing import Dict

from igraph import Graph
from onto import Onto


def convert_onto_to_graph(onto: Onto):
    nodes = onto.nodes()
    id_converter = dict()
    for i in range(len(nodes)):
        node_id = nodes[i]["id"]
        id_converter[node_id] = i

    links = list(map(lambda x: [id_converter[x["source_node_id"]], id_converter[x["destination_node_id"]]] , onto.links()))

    g = Graph(n=len(nodes), edges=links, directed=True)
    return g, id_converter


def map_concepts(doc, beta, distance, graph: Graph, onto: Onto, converter: Dict):
    concepts = doc["concepts"]
    sentences = doc["sentences"]

    if len(concepts) == 0:
        return []

    pairs = []
    bi = dict()
    bj = dict()
    for i in range(len(concepts)):
        for j in range(len(concepts)):
            if i == j:
                continue

            concept1 = onto.get_nodes_by_name(concepts[i])[0]
            name1 = concept1["name"]
            concept2 = onto.get_nodes_by_name(concepts[j])[0]
            name2 = concept2["name"]
            concords = list(filter(lambda x: name1 in x["concepts"] and name2 in x["concepts"], sentences))
            E = len(concords)
            if name1 in bi:
                bi[name1].append(E)
            else:
                bi[name1] = [E]

            if name2 in bj:
                bj[name2].append(E)
            else:
                bj[name2] = [E]

            paths = graph.get_all_simple_paths(converter[concept1["id"]], converter[concept2["id"]])
            P = len(list(filter(lambda x: len(x) <= distance, paths)))
            pairs.append({
                "concept1": name1,
                "concept2": name2,
                "S": 0.0,
                "S_norm": 0.0,
                "P": P,
                "Ub": 0.0,
                "E": E,
                "bi": 0,
                "bj": 0,
            })

    bi_calc = dict()
    bj_calc = dict()

    for pair in pairs:
        if pair["concept1"] not in bi_calc:
            bi_calc[pair["concept1"]] = len(list(filter(lambda x: x >= beta, bi[pair["concept1"]])))
        pair["bi"] = bi_calc[pair["concept1"]]

        if pair["concept2"] not in bj_calc:
            bj_calc[pair["concept2"]] = len(list(filter(lambda x: x >= beta, bj[pair["concept2"]])))
        pair["bj"] = bj_calc[pair["concept2"]]
        pair["Ub"] = (pair["bi"] + pair["bj"]) / float(len(concepts)) if len(concepts) != 0 else 0

    for pair in pairs:
        s = pair["bi"] + pair["bj"]
        if s == 0:
            pair["S"] = 0
        else:
            pair["S"] = math.sqrt(pair["P"]) * 2 * pair["Ub"] * pair["E"] / s

    all_s = list(map(lambda x: x["S"], pairs))
    min_s = min(all_s)
    max_s = max(all_s)

    for pair in pairs:
        pair["S_norm"] = (pair["S"] - min_s) / max_s if max_s != 0 else 0

    return pairs


def map_doc(doc, beta, distance, graph, onto, converter):
    info = map_concepts(doc, beta, distance, graph, onto, converter)
    score = sum(list(map(lambda x: x["S_norm"], info))) / len(info) if len(info) != 0 else 0
    return {
        "name": doc["name"],
        "score": score,
        "info": info,
    }


def range_docs(beta, distance, scribed_documents, onto):
    graph, converter = convert_onto_to_graph(onto)
    return sorted(list(map(lambda d: map_doc(d, beta, distance, graph, onto, converter), scribed_documents)), key=lambda x: -x["score"])