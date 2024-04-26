import numpy as np
import torch
from queue import Queue
from mmcv import Config


def k_adjacency(A, k, with_self=False, self_factor=1):
    # A is a 2D square array
    if isinstance(A, torch.Tensor):
        A = A.data.cpu().numpy()
    assert isinstance(A, np.ndarray)
    Iden = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return Iden
    Ak = np.minimum(np.linalg.matrix_power(A + Iden, k), 1) - np.minimum(np.linalg.matrix_power(A + Iden, k - 1), 1)
    if with_self:
        Ak += (self_factor * Iden)
    return Ak


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A, dim=0):
    # A is a 2D square array
    Dl = np.sum(A, dim)
    h, w = A.shape
    Dn = np.zeros((w, w))

    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)

    AD = np.dot(A, Dn)
    return AD


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.eye(num_node)

    for i, j in edge:
        A[i, j] = 1
        A[j, i] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [
        np.linalg.matrix_power(A, d) for d in range(max_hop + 1)
    ]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


class Graph:
    """The Graph to model the skeletons.

    Args:
        layout (str): must be one of the following candidates: 'openpose', 'nturgb+d', 'coco'. Default: 'coco'.
        mode (str): must be one of the following candidates: 'stgcn_spatial', 'spatial'. Default: 'spatial'.
        max_hop (int): the maximal distance between two connected nodes.
            Default: 1
    """

    def __init__(self,
                 layout='coco',
                 mode='spatial',
                 max_hop=1,
                 nx_node=1,
                 num_filter=3,
                 init_std=0.02,
                 init_off=0.04,
                 feature_connect=False):


        self.feature_connect = feature_connect
        self.max_hop = max_hop
        self.layout = layout
        self.part_A = []
        self.part_node = []
        self.mode = mode
        self.nx_node = nx_node
        self.num_filter = num_filter
        self.init_std = init_std
        self.init_off = init_off
        assert layout in ['openpose', 'nturgb+d', 'coco']

        self.get_layout(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.inward, max_hop)

        assert hasattr(self, mode), f'Do Not Exist This Mode: {mode}'
        if mode == 'part_random':
            self.A_of_DG = getattr(self, mode)()
            self.A_for_first_block = self.part_random(for_first_block=True)
        else:
            self.A = [getattr(self, mode)()]
            self.remain_node = [[i + 1 for i in range(self.A[0].shape[1])]]
            self.get_pooling_A(1, self.A, self.remain_node)
            self.get_pooling_A(2, self.A, self.remain_node)
            self.part_graph_larger_v3()
            # self.part_graph_larger_v3_k400()
        self.outward = None
        self.self_link = None
        self.center = None
        self.inward = None
        self.num_node = None

    def __str__(self):
        return self.A

    def get_layout(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self.inward = [
                (4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9),
                (9, 8), (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0),
                (14, 0), (17, 15), (16, 14)
            ]
            self.center = 1
        elif layout == 'nturgb+d':
            self.num_node = 25
            neighbor_base = [
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)
            ]
            # neighbor_base = [
            #     (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
            #     (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
            #     (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
            #     (20, 19), (22, 8), (23, 7), (24, 12), (25, 11)
            # ]
            self.inward = [(i - 1, j - 1) for (i, j) in neighbor_base]
            self.center = 21 - 1
        elif layout == 'coco':
            self.num_node = 17
            self.inward = [
                (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
                (1, 0), (3, 1), (2, 0), (4, 2)
            ]
            self.center = 0
        else:
            raise ValueError(f'Do Not Exist This Layout: {layout}')
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward

    def stgcn_spatial(self):
        adj = np.zeros((self.num_node, self.num_node))
        adj[self.hop_dis <= self.max_hop] = 1
        normalize_adj = normalize_digraph(adj)
        hop_dis = self.hop_dis
        center = self.center

        A = []
        for hop in range(self.max_hop + 1):
            a_close = np.zeros((self.num_node, self.num_node))
            a_further = np.zeros((self.num_node, self.num_node))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if hop_dis[j, i] == hop:
                        if hop_dis[j, center] >= hop_dis[i, center]:
                            a_close[j, i] = normalize_adj[j, i]
                        else:
                            a_further[j, i] = normalize_adj[j, i]
            A.append(a_close)
            if hop > 0:
                A.append(a_further)
        return np.stack(A)

    def spatial(self):
        Iden = edge2mat(self.self_link, self.num_node)
        In = normalize_digraph(edge2mat(self.inward, self.num_node))
        Out = normalize_digraph(edge2mat(self.outward, self.num_node))
        A = np.stack((Iden, In, Out))
        return A

    def binary_adj(self):
        A = edge2mat(self.inward + self.outward, self.num_node)
        return A[None]

    def random(self):
        num_node = self.num_node * self.nx_node
        num_node[0] = self.ge_out_nodes_num + self.part_in_nodes_num[0]
        return np.random.randn(self.num_filter, num_node, num_node) * self.init_std + self.init_off

    def part_random(self, for_first_block=False):
        num_node = [0, 0]
        A_random = []
        if not self.feature_connect or for_first_block:
            for i in range(2):
                num_node[i] = self.part_in_nodes_num[i]
                A_tmp = np.random.randn(self.num_filter, num_node[i], num_node[i]) * self.init_std + self.init_off
                A_random.append(A_tmp)
            return A_random
        else:
            for i in range(2):
                num_node[i] = self.ge_out_nodes_num + self.part_in_nodes_num[i]
                A_tmp = np.random.randn(self.num_filter, num_node[i], num_node[i]) * self.init_std + self.init_off
                A_random.append(A_tmp)
            return A_random


    """
    This function is primarily used for graph pooling,
    a process that downsamples the input graph to generate a smaller graph while preserving its key structural properties.
    """
    def get_pooling_A(self, pooling_time, A, remain_node_before):
        # Select the corresponding adjacency matrix
        A = A[pooling_time - 1]

        # Calculate the degree of each node
        degree = np.zeros(A.shape[1])
        for i in range(0, A.shape[1]):
            degree[i] = np.count_nonzero(A[1][i])+np.count_nonzero(A[2][i])

        # Initialize a queue for breadth-first search and a visited array
        q = Queue(A.shape[1])
        visited = np.zeros(A.shape[1])
        q.put(0)
        visited[0] = 1

        # Initialize the list of remaining nodes
        remain_node = [1]
        remain_node_before = remain_node_before[pooling_time - 1]
        times = 0

        # Perform breadth-first search to find the nodes that remain after pooling
        while not q.empty():
            times += 1
            for i in range(q.qsize()):
                current_node = q.get()
                for j in range(0, A.shape[1]):
                    if (A[1][current_node][j] > 0 or A[2][current_node][j] > 0) and visited[j] == 0:
                        visited[j] = 1
                        q.put(j)
                        if degree[j] != 2 or current_node+1 not in remain_node:
                            remain_node.append(remain_node_before[j])
        remain_node.sort()

        # Construct the new adjacency matrix
        self_link = [(i, i) for i in range(len(remain_node))]
        new_A_Iden = edge2mat(self_link, len(remain_node))
        new_A_inward = np.zeros((len(remain_node), len(remain_node)))

        for i in range(len(remain_node)):
            old_i = remain_node_before.index(remain_node[i])
            for j in range(len(remain_node)):
                old_j = remain_node_before.index(remain_node[j])
                if A[1][old_i][old_j] > 0:
                    new_A_inward[i][j] = 1
                else:
                    for k in range(len(remain_node_before)):
                        if remain_node_before[k] not in remain_node and A[1][k][old_j] > 0 and A[1][old_i][k] > 0:
                            new_A_inward[i][j] = 1

        # Transpose the matrix to get the outward adjacency matrix and normalize the matrices
        new_A_outward = new_A_inward.transpose()
        new_A_inward = normalize_digraph(new_A_inward)
        new_A_outward = normalize_digraph(new_A_outward)

        # Stack the identity, inward, and outward adjacency matrices to form the new adjacency matrix
        new_A = np.stack((new_A_Iden, new_A_inward, new_A_outward))

        # Update the list of remaining nodes and the adjacency matrix
        self.remain_node.append(remain_node)
        self.A.append(new_A)


    def part_graph_larger_v3(self):
        part_list = []
        body_leftarm = [
            (4, 3), (2, 21), (1, 2), (3, 21), (5, 21), (9, 21),
            (23, 8), (22, 8), (8, 7), (7, 6), (6, 5),
        ]
        part_list.append(body_leftarm)
        body_rightarm = [
            (4, 3), (2, 21), (1, 2), (3, 21), (5, 21), (9, 21),
            (24, 12), (25, 12), (12, 11), (11, 10), (10, 9),
        ]
        part_list.append(body_rightarm)
        leftarm_rightarm = [
            (23, 8), (22, 8), (8, 7), (7, 6), (6, 5),
            (24, 12), (25, 12), (12, 11), (11, 10), (10, 9),
        ]
        part_list.append(leftarm_rightarm)
        rightleg_rightarm = [
            (20, 19), (19, 18), (18, 17), (17, 1),
            (24, 12), (25, 12), (12, 11), (11, 10), (10, 9),
        ]
        part_list.append(rightleg_rightarm)
        leftleg_leftarm = [
            (16, 15), (15, 14), (14, 13), (13, 1),
            (23, 8), (22, 8), (8, 7), (7, 6), (6, 5),
        ]
        part_list.append(leftleg_leftarm)
        leftleg_rightarm = [
            (16, 15), (15, 14), (14, 13), (13, 1),
            (24, 12), (25, 12), (12, 11), (11, 10), (10, 9),
        ]
        part_list.append(leftleg_rightarm)

        rightleg_leftarm = [
            (20, 19), (19, 18), (18, 17), (17, 1),
            (23, 8), (22, 8), (8, 7), (7, 6), (6, 5),
        ]
        part_list.append(rightleg_leftarm)

        rightleg_leftleg = [
            (20, 19), (19, 18), (18, 17), (17, 1),
            (16, 15), (15, 14), (14, 13), (13, 1),
        ]
        part_list.append(rightleg_leftleg)

        part_node = []
        for list in part_list:
            node = []
            for x, y in list:
                if x - 1 not in node:
                    node.append(x - 1)
                if y - 1 not in node:
                    node.append(y - 1)
            node.sort()
            part_node.append(node)

        self.part_node.append(part_node)
        adjmatrix_list = []
        max_size = max([len(part) for part in part_node])
        for ind, part in enumerate(part_node):
            self_link = [(j, j) for j in range(len(part))]
            pad_s = ((0, max_size - len(part)), (0, max_size - len(part)))
            part_Iden = edge2mat(self_link, len(part))
            part_Iden = np.pad(part_Iden, pad_width=pad_s, mode='constant', constant_values=0)
            part_inward = np.zeros((max_size, max_size))
            for node_s, node_e in part_list[ind]:
                part_inward[part.index(node_e - 1)][part.index(node_s - 1)] = 1
            part_outward = part_inward.transpose()
            part_inward = normalize_digraph(part_inward)
            part_outward = normalize_digraph(part_outward)
            part_A = np.stack((part_Iden, part_inward, part_outward))
            adjmatrix_list.append(part_A)
        self.part_A.append(adjmatrix_list)

    def part_graph_larger_v3_k400(self):
        part_list = []
        body_leftarm = [
            (1, 0), (3, 1), (2, 0), (4, 2),  (6, 0),  (5, 0),
            (9, 7), (7, 5),
        ]
        part_list.append(body_leftarm)
        body_rightarm = [
            (1, 0), (3, 1), (2, 0), (4, 2),  (6, 0),  (5, 0),
            (10, 8), (8, 6),
        ]
        part_list.append(body_rightarm)
        leftarm_rightarm = [
            (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
        ]
        part_list.append(leftarm_rightarm)
        rightleg_rightarm = [
            (10, 8), (8, 6),  (6, 0),
            (16, 14), (14, 12), (12, 6), (6, 0),
        ]
        part_list.append(rightleg_rightarm)
        leftleg_leftarm = [
            (9, 7), (7, 5), (5, 0),
            (15, 13), (13, 11), (11, 5), (5, 0),
        ]
        part_list.append(leftleg_leftarm)
        leftleg_rightarm = [
            (10, 8), (8, 6),  (6, 0),
            (15, 13), (13, 11), (11, 5), (5, 0),
        ]
        part_list.append(leftleg_rightarm)

        rightleg_leftarm = [
            (9, 7), (7, 5),(5, 0),
            (16, 14), (14, 12), (12, 6), (6, 0),
        ]
        part_list.append(rightleg_leftarm)

        rightleg_leftleg = [
            (16, 14), (14, 12), (12, 6), (6, 0),
            (15, 13), (13, 11), (11, 5), (5, 0),
        ]
        part_list.append(rightleg_leftleg)

        part_node = []
        for list in part_list:
            node = []
            for x, y in list:
                if x  not in node:
                    node.append(x)
                if y not in node:
                    node.append(y)
            node.sort()
            part_node.append(node)

        self.part_node.append(part_node)
        adjmatrix_list = []
        max_size = max([len(part) for part in part_node])
        for ind, part in enumerate(part_node):
            self_link = [(j, j) for j in range(len(part))]
            pad_s = ((0, max_size - len(part)), (0, max_size - len(part)))
            part_Iden = edge2mat(self_link, len(part))
            part_Iden = np.pad(part_Iden, pad_width=pad_s, mode='constant', constant_values=0)
            part_inward = np.zeros((max_size, max_size))
            for node_s, node_e in part_list[ind]:
                part_inward[part.index(node_e)][part.index(node_s)] = 1
            part_outward = part_inward.transpose()
            part_inward = normalize_digraph(part_inward)
            part_outward = normalize_digraph(part_outward)
            part_A = np.stack((part_Iden, part_inward, part_outward))
            adjmatrix_list.append(part_A)
        self.part_A.append(adjmatrix_list)



