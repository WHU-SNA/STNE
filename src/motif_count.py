from collections import defaultdict

class Motif(object):
    def __init__(self, edges):
        self.pos_out_edge_list, self.neg_out_edge_list, self.pos_in_edge_list, self.neg_in_edge_list = \
            self.init_edge_list(edges)

    def init_edge_list(self, edges):
        pos_out_edge_list = defaultdict(list)
        neg_out_edge_list = defaultdict(list)
        pos_in_edge_list = defaultdict(list)
        neg_in_edge_list = defaultdict(list)
        for edge in edges:
            x, y, z = edge
            if z > 0:
                pos_out_edge_list[x].append(y)
                pos_in_edge_list[y].append(x)
            elif z < 0:
                neg_out_edge_list[x].append(y)
                neg_in_edge_list[y].append(x)
        return pos_out_edge_list, neg_out_edge_list, pos_in_edge_list, neg_in_edge_list

    def motif_vector(self, u, v):
        t1 = len(set(self.pos_out_edge_list[u]).intersection(set(self.pos_out_edge_list[v])))
        t2 = len(set(self.pos_out_edge_list[u]).intersection(set(self.pos_in_edge_list[v])))
        t3 = len(set(self.pos_out_edge_list[u]).intersection(set(self.neg_out_edge_list[v])))
        t4 = len(set(self.pos_out_edge_list[u]).intersection(set(self.neg_in_edge_list[v])))
        t5 = len(set(self.pos_in_edge_list[u]).intersection(set(self.pos_out_edge_list[v])))
        t6 = len(set(self.pos_in_edge_list[u]).intersection(set(self.pos_in_edge_list[v])))
        t7 = len(set(self.pos_in_edge_list[u]).intersection(set(self.neg_out_edge_list[v])))
        t8 = len(set(self.pos_in_edge_list[u]).intersection(set(self.neg_in_edge_list[v])))
        t9 = len(set(self.neg_out_edge_list[u]).intersection(set(self.pos_out_edge_list[v])))
        t10 = len(set(self.neg_out_edge_list[u]).intersection(set(self.pos_in_edge_list[v])))
        t11 = len(set(self.neg_out_edge_list[u]).intersection(set(self.neg_out_edge_list[v])))
        t12 = len(set(self.neg_out_edge_list[u]).intersection(set(self.neg_in_edge_list[v])))
        t13 = len(set(self.neg_in_edge_list[u]).intersection(set(self.pos_out_edge_list[v])))
        t14 = len(set(self.neg_in_edge_list[u]).intersection(set(self.pos_in_edge_list[v])))
        t15 = len(set(self.neg_in_edge_list[u]).intersection(set(self.neg_out_edge_list[v])))
        t16 = len(set(self.neg_in_edge_list[u]).intersection(set(self.neg_in_edge_list[v])))

        return [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16]
