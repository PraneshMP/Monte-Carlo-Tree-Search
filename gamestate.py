# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:17:53 2019

@author: Pranesh
"""
import numpy as np


class GameState(object):

    def __init__(self):
        
        self.player="O"
        self.shape=3
        self.board = np.zeros((self.shape,self.shape),dtype=str)

    def __key(self):
        return self.__str__()

    def __eq__(x, y):
        return x.__key() == y.__key()

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):

        output = ''
        for row in range(3):
            for col in range(3):
                contents = self.board[row][col]
                if col < 2:
                    output += '{}'.format(contents)
                else:
                    output += '{}\n'.format(contents)

        output = output.replace('', '-')
        return output

    def turn(self):
        
        return ("X","O")[self.player=="X"]

    def domove(self, move):

        self.board[move] = self.turn()

    def legal_moves(self):

        if self.winner() is not None:
            return []
        row,col=np.where(self.board=='')
        return  tuple(zip(row,col))

    def transition_function(self, move):

        self.domove(move)

    def winner(self):

        for player in ['X', 'O']:
            comb        =[player]*self.shape
            if (np.any(np.logical_or(np.all(self.board==comb,axis=0),np.all(self.board==comb,axis=1)))):
                return player
            if(np.logical_or(np.all(np.diag(self.board)==comb,axis=0),np.all(np.diag(np.fliplr(self.board))==comb,axis=0))):
                return player

        return len(np.argwhere(self.board==''))
            
        
    
        