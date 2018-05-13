import sys, getopt
import pydot
import pyparsing
import numpy


def  main(argv=None) :
    inputfile = ""
    try:
        opts, args = getopt.getopt(argv, "i:",["infile="])
    except getopt.GetoptError:
        print("Error: graphAna.py -i <inputfile>")
        print("\tor: graphAna.py --infile=<inputfile>")
        return 2
    for opt, arg in opts:
        if opt in ("-i", "--infile"):
            inputfile = arg
    graphlst = pydot.graph_from_dot_file(inputfile)
    graph = graphlst[0]
    nodes = graph.get_nodes()
    edges = graph.get_edges()
    adj_matrix = numpy.zeros((len(nodes), len(nodes)))
    nodelist = []
    for node in nodes:
        nodelist.append(node.get_name())
    for edge in edges :
        adj_matrix[nodelist.index(edge.get_source()[:12]), nodelist.index(edge.get_destination()[:12])] = 1
        adj_matrix[nodelist.index(edge.get_destination()[:12]), nodelist.index(edge.get_source()[:12])] = 1
    print(adj_matrix)

if __name__ == "__main__":
   sys.exit(main(sys.argv[1:]))




