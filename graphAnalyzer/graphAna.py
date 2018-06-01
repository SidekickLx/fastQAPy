import sys
import pydot
import numpy
import re


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


def dot_file_parser(filename):
    graphlst = pydot.graph_from_dot_file(filename)
    graph = graphlst[0]
    nodes = graph.get_nodes()
    edges = graph.get_edges()
    adj_matrix = numpy.zeros((len(nodes), len(nodes)))
    nodelist = []
    node_stmt_list = []
    for node in nodes:
        node_stmt_list.append(
            re.search('{.*}', node.to_string()[12:-1]).group())
        nodelist.append(node.get_name())
    for edge in edges:
        adj_matrix[nodelist.index(edge.get_source()[:12]), nodelist.index(
            edge.get_destination()[:12])] = 1
    return adj_matrix, node_stmt_list


def main(argv=None):
    inputfile1 = ("C:\\Users\\lixue\\Project\\FastQAP\\fastQAPy\\graphAnalyzer"
                  "\\cfg.fib.dot")
    inputfile2 = ("C:\\Users\\lixue\\Project\\FastQAP\\fastQAPy\\graphAnalyzer"
                  "\\cfg.fib.dot")
    adj_matrix_A, node_stmt_list_A = dot_file_parser(inputfile1)
    adj_matrix_B, node_stmt_list_B = dot_file_parser(inputfile2)
    diff_map = numpy.zeros((len(node_stmt_list_A), len(node_stmt_list_B)))
    for i_A, node_A in enumerate(node_stmt_list_A):
        for i_B, node_B in enumerate(node_stmt_list_B):
            diff = levenshteinDistance(node_A, node_B)
            if(diff < len(node_A) and diff < len(node_B)):
                diff_map[i_A][i_B] = diff
            else:
                diff_map[i_A][i_B] = float("inf")
    print(diff_map)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
