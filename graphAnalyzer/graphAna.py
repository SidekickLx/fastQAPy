import pydot
import numpy
import re


class CFG:

    graph_type = 'NORMAL'
    node_stmt_list = []
    adj_matrix = []
    inertion = []
    deletion = []
    mod_adj_matrix = []
    mod_node_stmt_list = []

    def __init__(self, filename):
        '''
        This funciton parse .dot file,
        generate adj Adjacency matrix and node stmt list.
        '''
        graphlst = pydot.graph_from_dot_file(filename)
        graph = graphlst[0]
        nodes = graph.get_nodes()
        edges = graph.get_edges()
        self.adj_matrix = numpy.zeros((len(nodes), len(nodes)))
        nodelist = []
        for node in nodes:
            self.node_stmt_list.append(
                re.search('{.*}', node.to_string()[12:-1]).group())
            nodelist.append(node.get_name())
        for edge in edges:
            self.adj_matrix[nodelist.index(edge.get_source()[:12]),
                            nodelist.index(edge.get_destination()[:12])] = 1


# Levenshtein distance
def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1],
                             distances_[-1])))
        distances = distances_
    return distances[-1]


def analysor(old_CFG, new_CFG):
    '''
    This function generate a diff map for two graphs.
    Use leven-shtein Distance Algorithm.
    '''
    diff_map = numpy.zeros((len(old_CFG.node_stmt_list),
                            len(new_CFG.node_stmt_list)))
    for i_A, node_A in enumerate(old_CFG.node_stmt_list):
        for i_B, node_B in enumerate(new_CFG.node_stmt_list):
            diff = levenshteinDistance(node_A, node_B)
            # We should use below after we solve the insertions and deletions.
            if(diff < len(node_A) and diff < len(node_B)):
                diff_map[i_A][i_B] = diff
            else:
                diff_map[i_A][i_B] = float("inf")
    # TODO deletion & insertion
    for idx, row in enumerate(diff_map):
        if not numpy.any(numpy.isfinite(row)):
            diff_map = numpy.delete(diff_map, idx, axis=0)
    for idx, col in enumerate(diff_map):
        if not numpy.any(numpy.isfinite(col)):
            diff_map = numpy.delete(diff_map, idx, axis=1)

    return diff_map


# if __name__ == "__main__":
