from tiled.client import from_uri
from tiled.query_registration import register
from dataclasses import dataclass

import collections
import json


@register(name="raw_mongo", overwrite=True)
@dataclass
class RawMongo:
    """Run a MongoDB query against a given collection."""

    query: str  # We cannot put a dict in a URL, so this a JSON str.

    def __init__(self, query):
        if isinstance(query, collections.abc.Mapping):
            query = json.dumps(query)
        self.query = query


##################


# TODO: figure out a better way to document the return values for this function
def search_sets_in_child(
    root, child_names, search_symbol=None, search_edge=None
):
    """Walks down a branch of nodes to get a specified child node and use a
    search criteria.

    Parameters
    ----------
    root : tiled.client.node.Node
        The parent node that the client will use to start going down using
        the specified branch in child_names.
    child_names : list of tiled.client.node.Node
        List of subsequent child nodes. They must be sorted in the same way
        they were created in the tree.
    search_symbol : str, optional
        Search criteria for the element symbol (aka the element, e.g. "Cu").
        Default is None.
    search_edge : str, optional
        Search criteria for the spectroscopy edge (e.g. the K-edge, "K", or the
        L3 edge "L3"). Default is None.

    Returns
    -------
    tiled.client.node.Node
        A structure representing an element in the tree. If the element is the
        parent node or a node in the middle of the tree, it returns
        tiled.client.node.Node. If it reaches the last node of a branch,
        it returns a client structure that represents the type of data
        structure that it contains; e.g.
        tiled.client.dataframe.DataFrameClient or
        tiled.client.array.ArrayClient.
    """

    if isinstance(child_names, list):
        child_names = tuple(child_names)
    
    # Search subsequent child nodes along the tree
    child_node = root
    for child in child_names:
        child_node = child_node[child]

    if search_symbol is None and search_edge is None:
        return child_node

    if search_symbol is not None:
        child_node = child_node.search(symbol(search_symbol))

    if search_edge is not None:
        child_node = child_node.search(edge(search_edge))

    return child_node


def symbol(symbol):
    """Wrapper method to generate a RawMongo query to search for an element
    symbol

    Parameters
    ----------
    symbol : str
        Query parameter used to search for an element symbol.

    Returns
    -------
    tiled.client.node.Node
        If a match was found, it returns a client node that includes all the
        child nodes containing each individual dataset.
    """

    return RawMongo({"metadata.element.symbol": symbol})


def edge(edge):
    """Wrapper method to generate a RawMongo query to search for an element
    edge.

    Parameters
    ----------
    edge : str
        The parent node that the client will use to start going down using the
        specified branch in child_names.

    Returns
    -------
    tiled.client.node.Node
        If a match was found, it returns a client node that includes all the
        child nodes containing each individual dataset.
    """

    return RawMongo({"metadata.element.edge": edge})


if __name__ == "__main__":

    # Example Code
    client = from_uri("https://aimm.lbl.gov/api")
    child = ["NCM", "BM_NCMA"]
    result_nodes = search_sets_in_child(
        client, child, search_symbol="Ni", search_edge="L3"
    )

    # TODO: Test multiple cases by passing a list with multiple paths.
    #       What is the best data structure to use as a container for the results?
    # children = [['NCM', 'BM_NCMA'],
    #             ['NCM', 'BM_NCM622']]
