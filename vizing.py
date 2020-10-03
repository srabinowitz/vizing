# vizing.py
# store, edge color, and display undirected graphs
# (c) Samuel Rabinowitz, 2020

import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

class colored_graph:
    
    # initialization functions
    
    # we can initialize a complete graph, give an adjacency matrix, or neither
    
    def __init__(self, complete = None, matrix = None, verbose = False):
        self.__v_list = []
        self.__col_list = []
        self.__colors = []
        self.verbose = verbose
        self.__fig = plt.Figure()
        if(complete):
            self.__make_complete(complete)
        elif(matrix):
            self.__import_matrix(np.array(matrix))
        pass
    
    # make a complete graph of degree k
    
    def __make_complete(self, k):
        for i in range(k):
            v_i = []
            c_i = []
            for j in range(k):
                if(i != j):
                    v_i.append(j)
                    c_i.append(-1)
            self.__v_list.append(v_i)
            self.__col_list.append(c_i)
            
    # import an adjacency matrix
    # note: we must convert the matrix into an adjaceny list
    # note: this function is untested
    
    def __import_matrix(self, matrix):
        assert(matrix.shape[0] == matrix.shape[1])
        for i in range(matrix.shape[0]):
            edges_i = []
            colors_i = []
            for j in range(matrix.shape[1]):
                if(matrix[i][j] > 0):
                    edges_i.append(j)
                    colors_i.append(-1)
            self.__v_list.append(edges_i)
            self.__col_list.append(colors_i)
    
    # graph utility functions
    
    def add_vertex(self):
        self.__v_list.append([])
        self.__col_list.append([])
        return len(self.__v_list) - 1

    # check to make sure a and b are not already connected
    # the graph should be undirected, but it gives
    # me peace of mind to check twice here
    # note: we cannot use sets here because we must
    # have __col_list aligned with __v_list

    def connect_chk(self, a, b):
        if not (a in self.__v_list[b]):
            self.__v_list[b].append(a)
            self.__col_list[b].append(-1)
        if not (b in self.__v_list[a]):
            self.__v_list[a].append(b)
            self.__col_list[a].append(-1)
    
    # this will fail down the line if a and b are already connected

    def connect(self, a, b):
        self.__v_list[b].append(a)
        self.__col_list[b].append(-1)
        self.__v_list[a].append(b)
        self.__col_list[a].append(-1)
        
    # drawing functions
    
    # when we draw the graph, we arrange the vertices into a circle
        
    def __get_circle_coordinates(self):
        n_v = len(self.__v_list)
        return np.asarray([[math.cos(i * 2 * math.pi / n_v),
            math.sin(i * 2 * math.pi / n_v)] for i in range(n_v)]).T
    
    # maps color numbers [0, k - 1] to matplotlib color
    # edges that are not already colored (color -1) should be black
    
    def __get_color(self, e_color):
        if(e_color < 0):
            return "k"
        else:
            return self.__colors[e_color]
        
    # we typically operate on a subgraph of G, H, so this makes
    # edge that are part of H solid, or dashed otherwise
    # n is the number of vertices in H, and k is the number of edges
    
    def __get_style(self, e_color, n, k, source, dest):
        if(e_color < 0 or e_color >= k or source >= n or dest >= n):
            return "--"
        else:
            return "-"
    
    # draw colored edges between vertices
    
    def __draw_lines(self, coords, n, k):
        n_v = len(self.__v_list)
        for i in range(n_v):
            for j in range(len(self.__v_list[i])):
                dest = self.__v_list[i][j]
                e_color = self.__col_list[i][j]
                plt.plot([coords[0, i], coords[0, dest]], [coords[1, i],
                    coords[1, dest]], self.__get_style(e_color, n, k, i, dest),
                    color = self.__get_color(e_color))
                
    # draw graph with edge colors
    # note: edges with unassigned colors or colors >= k are dashed
    # note: vertices >= n are drawn with x's instead of o's
    # highlight highlights vertex n - 1 red

    def __draw_partial_graph(self, n, k, highlight = True):
        coords = self.__get_circle_coordinates()
        self.__draw_lines(coords, n, k)
        plt.plot(coords[0][:n - 1], coords[1][:n - 1], "ko")
        high_style = "ko"
        if highlight:
            high_style = "ro"
        plt.plot([coords[0][n - 1]], [coords[1][n - 1]], high_style)
        plt.plot(coords[0][n:], coords[1][n:], "kx")
        for i in range(coords.shape[1]):
            plt.annotate(str(i), coords[:,i])
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.axis("scaled")
        plt.axis("off")
        plt.show()
        
    # draw full graph
    # note: this function has very limited usage
    # since I have not fully integrated it with
    # matplotlib
    
    def draw_graph(self):
        self.__draw_partial_graph(len(self.__v_list), self.__find_max_degree()
            + 1, highlight = False)
        
    # add a text message to graph
    
    def __draw_msg(self, msg):
        plt.annotate(msg, (0, -0.75))
                
    # Coloring Algorithm
    
    # delta
        
    def __find_max_degree(self):
        return max([len(self.__v_list[i]) for i in range(len(self.__v_list))])
    
    # get a list of edge colors adjacent to u excluding
    # edges to vertices > n - 2 or edges of color > k - 1
    
    def __get_adj_cols(self, u, n, k):
        n_adj = len(self.__v_list[u])
        return [self.__col_list[u][i] for i in range(n_adj)
        if self.__col_list[u][i] < k and self.__v_list[u][i] < n - 1]
    
    # edge colors not present at u in subgraph
    
    def __missing_colors(self, u, n, k):
        adj_cols = self.__get_adj_cols(u, n, k)
        return [col for col in range(k) if col not in adj_cols]
    
    # note: The algorithm calls for us to generate dummy vertices
    # so that each neighbor of v has degree k - 1, expect one vertex
    # of degree k. The way I handle this here is that instead of
    # actually generating dummy vertices, I just trim the list of
    # edge colors that do not hit vertex u, which gives the same
    # result. I account for this later in __follow_component and
    # __follow_component_swap
    
    def __missing_colors_w_dummies(self, u, n, k, deg):
        expected_size = k - deg + 1
        return self.__missing_colors(u, n, k)[:expected_size]
    
    # x[i] is a list of vertices that
    # neighbor v but lack color i
    # note: we do not include neighbors of
    # v where we have already colored that edge
    # a color out of range(k)
    
    def __gen_x(self, v, n, k):
        x = [[] for i in range(k)]
        sub = 0
        u_n = len(self.__v_list[v])
        valid_neighbors = 0
        for ind in range(u_n):
            i = self.__v_list[v][ind]
            if(self.__col_list[v][ind] < k and i < n):
                valid_neighbors = valid_neighbors + 1
                missing = self.__missing_colors_w_dummies(i, n, k, k - sub)
                for j in missing:
                    x[j].append(i)
                sub = 1
        if(valid_neighbors == 0):
            return None
        else:
            return x
    
    # given a path of edge colors cmin and cmax
    # that start at u, find where the path ends
    # note: the first edge on the path must be
    # color cmin
    
    def __follow_component(self, n, u, cmin, cmax):
        if cmin in self.__col_list[u]:
            i = self.__col_list[u].index(cmin)
            v = self.__v_list[u][i]
            if v < n:
                return self.__follow_component(n, v, cmax, cmin)
            else:
                return u
        
    # given a path of edge colors cmin and cmax
    # that start at u, find where the path ends,
    # and switch the edge colors cmin and cmax
    # along the path
    # note: the first edge on the path must be
    # color cmin
    
    def __follow_component_swap(self, n, u, cmin, cmax):
        if cmin in self.__col_list[u]:
            i = self.__col_list[u].index(cmin)
            v = self.__v_list[u][i]
            if v < n:
                self.__col_list[u][i] = cmax
                end_v = self.__follow_component_swap(n, v, cmax, cmin)
                self.__col_list[v][self.__col_list[v].index(cmin)] = cmax
                return end_v
            else:
                return u
        
    # given len(x[cmax]) > len(x[cmin]) + 2, perform a series
    # of color swaps that will transfer that will transfer either
    # one or two vertices from x[cmax] to x[cmin]
    
    def __swap_col(self, n, x, cmin, cmax):
        # there is a vertex in Xcmax but not Xcmin
        us = [u for u in x[cmax] if u not in x[cmin]]
        v = us[0]
        for u in us:
            v = u
            if self.__follow_component(n, u, cmin, cmax) not in x[cmin]:
                break
        x[cmax].remove(v)
        x[cmin].append(v)
        v = self.__follow_component_swap(n, v, cmin, cmax)
        if v in x[cmax]:
            x[cmax].remove(v)
            x[cmin].append(v)
            
    # for all colored edges in our graph, swap color a with b
    # only for vertices < n
            
    def __swap_all_col(self, n, a, b):
        # n_v = len(self.__v_list)
        for i in range(n):
            deg_i = len(self.__v_list[i])
            for j in range(deg_i):
                if(self.__v_list[i][j] < n):
                    if(self.__col_list[i][j] == a):
                        self.__col_list[i][j] = b
                    elif(self.__col_list[i][j] == b):
                        self.__col_list[i][j] = a
                        
    # reset all edges of color < k between the first n vertices
                        
    def __clear_coloring(self, n, k):
        for i in range(n):
            deg_i = len(self.__v_list[i])
            for j in range(deg_i):
                if(self.__v_list[i][j] < n and self.__col_list[i][j] < k):
                    self.__col_list[i][j] = -1
                                        
    # color the edge between a and b color c,
    # there must actually be an edge between
    # a and b to color, or this will fail
    
    def __color_edge(self, a, b, c):
        self.__col_list[a][self.__v_list[a].index(b)] = c
        self.__col_list[b][self.__v_list[b].index(a)] = c

    # color the edges of H, the subgraph of the first n vertices of G,
    # with k colors we ignore edges that already have colors assigned
    # that are >= k

    def __color_edges_n(self, n, k):
        if(n <= 1 or k < 1):
            return
        self.__color_edges_n(n - 1, k)
        v = n - 1
        x = self.__gen_x(v, n, k)
        if x:
            xlens = list(map(len, x))
            while 1 not in xlens:
                self.__swap_col(n, x, np.argmin(xlens), np.argmax(xlens))
                xlens = list(map(len, x))
            i = list(map(len, x)).index(1)
            u = x[i][0]
            if(i != k - 1):
                self.__swap_all_col(n, i, k - 1)
            self.__color_edge(u, v, k - 1)
            self.__clear_coloring(n, k - 1)
            self.__color_edges_n(n, k - 1)
        if(self.verbose):
            self.__draw_msg("n: {}, k: {}".format(n, k))
            self.__draw_partial_graph(n, k)
            
    # guaranteed no more than delta + 1 colors

    def color_edges(self):
        n_v = len(self.__v_list)
        delta = self.__find_max_degree()
        self.__colors = sns.color_palette(n_colors = delta + 1).as_hex()
        self.__color_edges_n(n_v, delta + 1)
    
    # makes sure no adjacent edges have the same color
    
    def check_correct(self):
        return not any([(len(es) != len(set(es))) for es in self.__col_list])