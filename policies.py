# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:19:56 2019

@author: Pranesh
"""

import numpy as np
import operator
import networkx as nx
from gamestate import GameState

EPSILON = 10e-6


class RandomPolicy():
    
    def move(self, state):
        
        moves = state.legal_moves()
        index = np.random.randint(len(moves))
        return moves[index]

class MCTSPolicy():

    def __init__(self, player):

        self.graph = nx.DiGraph()
        self.player = player
        self.num_sim = 0
        self.uct_c = np.sqrt(2)
        self.node_count = 0
        empty_board = GameState()
        self.graph.add_node(self.node_count, attr_dict={'w(s,a)': 0,'n(s,a)': 0,'uct(s,a)': 0,'expand': False,'state': empty_board})
        empty_board_node_id = self.node_count
        self.node_count += 1
        self.last_move = None
        if player is 'O':
            for successor in [empty_board.transition_function(*move) for move in empty_board.legal_moves()]:
                self.graph.add_node(self.node_count, attr_dict={'w(s,a)': 0,'n(s,a)': 0,'uct(s,a)': 0,'expand': False,'state': successor})
                self.graph.add_edge(empty_board_node_id, self.node_count)
                self.node_count += 1

    def reset_game(self):
        
        self.last_move = None
        
    def move(self, start_state, simulations= 25):

        root = None
        if self.last_move is not None:
            exists = False
            for child in self.graph.successors(self.last_move):
                if self.graph.node[child]['state'] == start_state:
                    if self.graph.has_edge(self.last_move, child):
                        exists = True
                        root = child
            if not exists:
                self.graph.add_node(self.node_count,attr_dict={'w(s,a)': 0, 'n(s,a)': 0,'uct(s,a)': 0,'expand': False,'state': start_state})
                self.graph.add_edge(self.last_move,self.node_count)
                root = self.node_count
                self.node_count += 1     
        else:      
            for node in self.graph.nodes():
                if self.graph.node[node]['state'] == start_state:
                    root = node
        for i in range(simulations):
            self.num_sim += 1
            selected_node = self.selection(root)
            if self.graph.node[selected_node]['state'].winner():
                break
            new_child_node = self.expansion(selected_node)
            reward = self.simulation(new_child_node)
            self.backpropagation(new_child_node, reward)
        move, resulting_node = self.best(root)
        self.last_move = resulting_node
        if self.graph.node[resulting_node]['state'].winner():
            self.last_move = None
        return move

    def best(self, root):

        children = self.graph.successors(root)
        mod_uct_val = {}
        for child_node in children:
            mod_uct_val[child_node] = self.uct(child_node)
        best_children = [key for key, val in mod_uct_val.iteritems() if val == max(mod_uct_val.values())]
        index = np.random.randint(len(best_children))
        best_child = best_children[index]
        action = self.graph.get_edge_data(root, best_child)['action']
        return action, best_child

    def selection(self, root):

        if root not in self.graph.nodes():
            self.graph.add_node(self.node_count,attr_dict={'w(s,a)': 0,'n(s,a)': 0,'uct(s,a)': 0,'expand': False,'state': root})
            self.node_count += 1
            return root
        elif not self.graph.node[root]['expand']:
             return root  
        else:
            children           = self.graph.successors(root)
            mod_uct_val     = {}
            for child_node in children:
                mod_uct_val[child_node] = self.mod_uct(state=child_node)
            best_child_node = max(mod_uct_val.items(), key=operator.itemgetter(1))[0]
            return self.selection(best_child_node)

    def expansion(self, node):
        
        children        = self.graph.successors(node)
        moves           = self.graph.node[node]['state'].legal_moves()
        unvisited       = []
        actions         = []
        for move in moves:
            child = self.graph.node[node]['state'].transition_function(*move)
            in_children = False
            for child_node in children:
                if self.graph.node[child_node]['state'] == child:
                    in_children = True
            if not in_children:
                unvisited.append(child)
                actions.append(move)
        if len(unvisited) > 0:
            index = np.random.randint(len(unvisited))
            child, move = unvisited[index], actions[index]
            self.graph.add_node(self.node_count,attr_dict={'w(s,a)': 0,'n(s,a)': 0,'uct(s,a)': 0, 'expand': False,'state': child})
            self.graph.add_edge(node, self.node_count, attr_dict={'action': move})
            child_node_id = self.node_count
            self.node_count += 1
        else:
            return node
        if len(children) + 1 == len(moves):
            self.graph.node[node]['expand'] = True

        return child_node_id

    def simulation(self, node):

        random_policy = RandomPolicy()
        current_state = self.graph.node[node]['state']
        while not current_state.winner():
            move = random_policy.move(current_state)
            current_state = current_state.transition_function(*move)

        if current_state.winner() == self.player:
            return 1
        else:
            return 0

    def backpropagation(self, last_visited, reward):

        current = last_visited
        while True:
            self.graph.node[current]['n(s,a)'] += 1
            self.graph.node[current]['w(s,a)'] += reward
            if self.graph.node[current]['state'] == GameState():
                break
            try:
                current = self.graph.predecessors(current)[0]
            except IndexError:
                break

    def uct(self, state):
        n = self.graph.node[state]['n(s,a)']  
        w = self.graph.node[state]['w(s,a)'] 
        t = self.num_sim
        c = self.uct_c
        epsilon = EPSILON
        exploitation_value = w / (n + epsilon)
        exploration_value = c * np.sqrt(np.log(t) / (n + epsilon))
        value = exploitation_value + exploration_value
        self.graph.node[state]['uct(s,a)'] = value
        return value