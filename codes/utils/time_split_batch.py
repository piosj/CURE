from dgl.data.utils import load_graphs
import numpy as np

def split_train_graph(history_week, interval_hours, graph_path, plus=None):
    """
    Returns:
        Total_Graph: a global graph
        splitted: list of DGLGraphs
    """
    Total_Graph = load_graphs(graph_path)[0][0]
        
    splitted = []
    float_snapshot_num = history_week * 7 * 24 / interval_hours
    snapshots_num = int(float_snapshot_num)
    if plus is not None:
        snapshots_num += 1

    for i in range(snapshots_num):
        # Creating subgraphs by time_idx (preserve every nodes in global graph)
        graph_at_i = Total_Graph.edge_subgraph(np.where(Total_Graph.edata['time_idx'] == i)[0], preserve_nodes = True)
        graph_at_i.copy_from_parent()
        splitted.append(graph_at_i)

    return Total_Graph, splitted   # Total graph which has all user & news with all edges, subgraphs splitted by time_idx however have all nodes 