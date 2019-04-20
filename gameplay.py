# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:19:35 2019

@author: Pranesh
"""

import networkx as nx
from gamestate import GameState
from policies import MCTSPolicy
G1=nx.DiGraph()

class StateNode(object):
    def __init__(self, board):
        self.state = board
        self.parent = None
        self.child = None

def play_game(player_policies):

    game = GameState()
    G = nx.DiGraph()
    G.add_node(str(game))
    root = str(game)
    current = root
    plies = 0
    for player_policy in player_policies:
         if type(player_policy) is MCTSPolicy:
            player_policy.reset_game()    
    while game.winner() is None:
        for player_policy in player_policies:
            plies += 1
            game.domove(*player_policy.move(game,1))
            previous = current
            G.add_node(str(game))
            current = str(game)
            G.add_edge(previous, current)
            print(current)
            break
            if game.winner() is not None:
                break
            move=list(input())
            move=[int(i) for i in move]
            game.domove(*move)
            previous = current
            G.add_node(str(game))
            current = str(game)
            G.add_edge(previous, current)
            if game.winner() is not None:
                break
    print('Game over. Winner is {}.'.format(game.winner()))
    return G, game.winner()

player_policies=[MCTSPolicy("X")]
G,R=play_game(player_policies)
#visualize(G)