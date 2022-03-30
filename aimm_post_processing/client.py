from tiled.client import from_uri
from tiled.query_registration import register
from dataclasses import dataclass

import collections
import json


@register(name="raw_mongo", overwrite=True)
@dataclass
class RawMongo:
    """
    Run a MongoDB query against a given collection.
    """

    query: str  # We cannot put a dict in a URL, so this a JSON str.

    def __init__(self, query):
        if isinstance(query, collections.abc.Mapping):
            query = json.dumps(query)
        self.query = query


##################


def search_sets_in_child(root, child_names, search_symbol=None, search_edge=None):
    """Walk down a branch of nodes to get a specified child node and use a search criteria

    :param parent_node: the parent node that the client will use to start going down using
        the specified branch in child_names.
    :type parent_node: tiled.client.node.Node
    :param node_name_list: list of subsequent child nodes. They must be sorted in the same way
        they were created in the tree
    :type node_name: list
    :param search_symbol: search criteria for element symbol
    :type node_name: str
    param search_symbol: search criteria for element edge
    :type node_name: str
    :return: the child node that was found in parent_node once the list of node names has
        been used completely.
    :rtype: a structure representing an element in the tree. If the element is the
        parent node or a node in the middle of the tree, it returns tiled.client.node.Node.
        If it reaches the last node of a branch, it returns a clietn structure that
        represents the type of data strcuture that it contains;
        e.g. tiled.client.dataframe.DataFrameClient, tiled.client.array.ArrayClient
    """

    if isinstance(child_names, list):
        child = tuple(child_names)
    child_node = root[child]

    if not search_symbol and not search_edge:
        return child_node
    if search_symbol:
        child_node = child_node.search(symbol(search_symbol))

    if search_edge:
        child_node = child_node.search(edge(search_edge))

    return child_node


def symbol(symbol):
    """Wrapper method to generate a RawMongo query to search for an element symbol

    :param symbol: query parameter used to search for an element symbol.
    :type symbol: str
    :return: if a match was found, it returns a client node that includes all
        the child nodes containing each individual dataset.
    :rtype: tiled.client.node.Node
    """

    query = {"metadata.element.symbol": symbol}
    return RawMongo(query)


def edge(edge):
    """Wrapper methof to generate a RawMongo query to search for an element edge

    :param symbol: the parent node that the client will use to start going down using
        the specified branch in child_names.
    :type symbol: str
    :return: if a match was found, it returns a client node that includes all
        the child nodes containing each individual dataset.
    :rtype: tiled.client.node.Node
    """
    query = {"metadata.element.edge": edge}
    return RawMongo(query)


if __name__ == "__main__":

    # Example Code
    client = from_uri("https://aimm.lbl.gov/api")
    child = ["NCM", "BM_NCMA"]
    result_nodes = search_sets_in_child(client, child, search_symbol="Ni", search_edge="L3")

    # TODO: Test multiple cases by passing a list with multiple paths.
    #       What is the best data structure to use as a container for the results?
    # children = [['NCM', 'BM_NCMA'],
    #             ['NCM', 'BM_NCM622']]
