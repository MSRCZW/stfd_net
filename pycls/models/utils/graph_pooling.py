import numpy as np
import torch
from queue import Queue

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
                 max_hop=1):

        self.max_hop = max_hop
        self.layout = layout
        self.mode = mode
        self.remain_node = [[i+1 for i in range(25)]]
        assert layout in ['openpose', 'nturgb+d', 'coco']
        self.get_layout(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.inward, max_hop)
        assert hasattr(self, mode), f'Do Not Exist This Mode: {mode}'
        self.A = [getattr(self, mode)()]
        self.get_pooling_A(1, self.A, self.remain_node)
        self.get_pooling_A(2, self.A, self.remain_node)

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

    def get_pooling_A(self,pooling_time,A,remain_node_before):
        A = A[pooling_time-1]
        degree = np.zeros(A.shape[1])
        for i in range(0, A.shape[1]):
            sum = 0
            for j in range(0, A.shape[1]):
                sum = sum + np.ceil(A[1][i][j]) + np.ceil(A[2][i][j])
            degree[i]=sum

        q = Queue(A.shape[1])
        visited = np.zeros(A.shape[1])
        q.put(0)
        visited[0]=1
        remain_node = [1]
        remain_node_before = remain_node_before[pooling_time-1]
        times = 0
        while (q.empty() != True):
            times += 1
            for i in range(q.qsize()):
                current_node=q.get()
                if degree[current_node] > 2:
                    flag = 1
                else:
                    flag = 0
                for j in range(0, A.shape[1]):
                    if (A[1][current_node][j] > 0 or A[2][current_node][j] > 0) and visited[j] == 0:
                        visited[j] = 1
                        q.put(j)
                        if degree[j] != 2 or (times % 2 == 0 and flag != 1):
                            remain_node.append(remain_node_before[j])
        remain_node.sort()
        self_link = [(i, i) for i in range(len(remain_node))]
        new_A_Iden = edge2mat(self_link, len(remain_node))
        new_A_inward = np.zeros((len(remain_node),len(remain_node)))

        for i in range(len(remain_node)):
            old_i = remain_node_before.index(remain_node[i])
            for j in  range(len(remain_node)):
                old_j = remain_node_before.index(remain_node[j])
                if A[1][old_i][old_j] > 0:
                    new_A_inward[i][j] = 1
                else:
                    for k in range(len(remain_node_before)):
                        if remain_node_before[k] not in remain_node and A[1][k][old_j] > 0 and A[1][old_i][k] > 0:
                            new_A_inward[i][j] = 1
        new_A_outward = new_A_inward.transpose()
        new_A_inward = normalize_digraph(new_A_inward)
        new_A_outward = normalize_digraph(new_A_outward)
        new_A = np.stack((new_A_Iden,new_A_inward,new_A_outward))
        self.remain_node.append(remain_node)
        self.A.append(new_A)

