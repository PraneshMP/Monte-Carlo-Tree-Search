# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:19:56 2019

@author: Pranesh
"""

import numpy as np
import operator
import networkx as nx
from gamestate import GameState
from visualize import  visualize
import copy

EPSILON = 10e-6


class RandomPolicy():
    
    def move(self, state):
        
        moves = state.legal_moves()
        index = np.random.randint(len(moves))
        return moves[index]

class MCTSPolicy():

    def __init__(self, player):

        self.graph = nx.DiGraph()
        self.simigraph=nx.DiGraph()
        self.player = player
        self.num_sim = 0
        self.uct_c = np.sqrt(2)
        empty_board = GameState()
        self.graph.add_node(empty_board, attr_dict={'w(s,a)': 0,'n(s,a)': 0,'uct(s,a)': 0,'expand': False,'state': empty_board})
        self.last_move = None
        if player is 'O':
            for successor in [empty_board.transition_function(*move) for move in empty_board.legal_moves()]:
                self.graph.add_node(successor, attr_dict={'w(s,a)': 0,'n(s,a)': 0,'uct(s,a)': 0,'expand': False,'state': successor})
                self.graph.add_edge(empty_board, successor)

    def reset_game(self):
        
        self.last_move = None
        
    def move(self, start_state, simulations):

        root = None
        if self.last_move is not None:
            exists = False
            for child in self.graph.successors(self.last_move):
                if self.graph.node[child]['state'] == start_state:
                    if self.graph.has_edge(self.last_move, child):
                        exists = True
                        root = child
            if not exists:
                self.graph.add_node(start_state,attr_dict={'w(s,a)': 0, 'n(s,a)': 0,'uct(s,a)': 0,'expand': False,'state': start_state})
                self.graph.add_edge(self.last_move,start_state)   
        else:      
            for node in self.graph.nodes():
                #print(self.graph.node[node])
                if self.graph.node[node]['attr_dict']['state'] == start_state:
                    root = node
        for i in range(1):
            self.num_sim += 1
            selected_node = self.selection(root)
            if self.graph.node[selected_node]['attr_dict']['state'].winner():
                break
            #print('1')
            new_child_node = self.expansion(selected_node)
            #print('2')
            reward = self.simulation(new_child_node)
            #print([str(self.simigraph.nodes[i]) for i in self.simigraph.nodes()])
            #print('3')
            self.backpropagation(new_child_node, reward)
            #print([str(self.simigraph.nodes[i]) for i in self.simigraph.nodes()])
        move, resulting_node = self.best(root)
        self.last_move = resulting_node

        if self.graph.node[resulting_node]['attr_dict']['state'].winner():
            self.last_move = None
        visualize(self.graph)
        return move

    def best(self, root):

        children = self.graph.successors(root)
        mod_uct_val = {}
        for child_node in children:
            mod_uct_val[child_node] = self.mod_uct(child_node)
        best_children = [key for key, val in mod_uct_val.items() if val == max(mod_uct_val.values())]
        index = np.random.randint(len(best_children))
        best_child = best_children[index]
        action = self.graph.get_edge_data(root, best_child)['attr_dict']['action']
        return action, best_child

    def selection(self, root):

        if root not in self.graph.nodes():
            self.graph.add_node(root,attr_dict={'w(s,a)': 0,'n(s,a)': 0,'uct(s,a)': 0,'expand': False,'state': root})
            return root
        elif not self.graph.node[root]['attr_dict']['expand']:
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
        moves           = self.graph.node[node]['attr_dict']['state'].legal_moves()
        unvisited       = []
        actions         = []
        for move in moves:
            child = self.graph.node[node]['attr_dict']['state'].transition_function(*move)
            in_children = False
            for child_node in children:
                if self.graph.node[child_node]['attr_dict']['state'] == child:
                    in_children = True
            if not in_children:
                unvisited.append(child)
                actions.append(move)
        if len(unvisited) > 0:
            index = np.random.randint(len(unvisited))
            child, move = unvisited[index], actions[index]
            self.graph.add_node(child,attr_dict={'w(s,a)': 0,'n(s,a)': 0,'uct(s,a)': 0, 'expand': False,'state': child})
            self.graph.add_edge(node, child, attr_dict={'action': move})
        else:
            return node
        
        if len(list(children)) + 1 == len(moves):
            self.graph.node[node]['attr_dict']['expand'] = True

        return child

    def simulation(self, node):


        random_policy = RandomPolicy()
        current_state = self.graph.node[node]['attr_dict']['state']
        temp=copy.deepcopy(current_state)
        while not current_state.winner():
            move = random_policy.move(current_state)
            temp1=copy.deepcopy(current_state)
            current_state = current_state.transition_function(*move)
            if current_state not in self.simigraph.nodes():
                self.simigraph.add_node(current_state,attr_dict={ 'w_hat': 0,'nsim': 0,'state' : current_state})
                self.simigraph.add_edge(temp1,current_state, attr_dict={'action': move})
            #print(str(self.simigraph.nodes[current_state]))
        #print([i for i in self.simigraph.nodes()])
        empty=None
        for i in self.simigraph.nodes():
            #if self.simigraph.nodes[i] == {}:
                #empty=i
        #self.simigraph.remove_node(empty)
        if current_state.winner() == self.player:
            reward=1
            while True:
                if current_state == temp:
                    break
                self.simigraph.node[current_state]['attr_dict']['nsim'] += 1
                self.simigraph.node[current_state]['attr_dict']['w_hat'] += reward
                try:
                    current_state = list(self.simigraph.predecessors(current_state))[0]
                except IndexError:
                    break
            #print([str(self.simigraph.nodes[i]) for i in self.simigraph.nodes()])
            return 1
        else:
            reward=0
            while True:
                if current_state == temp:
                    break
                self.simigraph.node[current_state]['attr_dict']['nsim'] += 1
                self.simigraph.node[current_state]['attr_dict']['w_hat'] += reward
                try:
                    current_state = list(self.simigraph.predecessors(current_state))[0]
                except IndexError:
                    break
            return 0

    def backpropagation(self, last_visited, reward):

        current = last_visited
        while True:
            self.graph.node[current]['attr_dict']['n(s,a)'] += 1
            self.graph.node[current]['attr_dict']['w(s,a)'] += reward
            if self.graph.node[current]['attr_dict']['state'] == GameState():
                break
            try:
                current = list(self.graph.predecessors(current))[0]
            except IndexError:
                break

    def mod_uct(self, state):
        n = self.graph.node[state]['attr_dict']['n(s,a)']  
        w = self.graph.node[state]['attr_dict']['w(s,a)'] 
        t = self.num_sim
        c = self.uct_c
        nsim=1e-10
        w_hat=1e-10        
        if self.simigraph.node[state]!={}:
            #print(self.simigraph.node[state])
            nsim=self.simigraph.node[state]['attr_dict']['nsim']
            w_hat=self.simigraph.node[state]['attr_dict']['w_hat']
        
        epsilon = EPSILON
        beta=0.1
        Qplus_val = (1-beta)*(w / n) + beta * (w_hat/nsim)
        Q_val = c * np.sqrt(np.log(t) / (n + epsilon))
        Qplus_value = Q_val + Qplus_val
        self.graph.node[state]['attr_dict']['uct(s,a)'] = Qplus_value
        return Qplus_value
    
