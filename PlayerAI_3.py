#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 17:17:34 2018

@author: jinzhao
"""
from BaseAI_3 import BaseAI
from Grid_3   import Grid
from copy     import deepcopy
from random   import randint
import time
import numpy as np

timeLimit = 0.2
defaultProbability = 0.9
directionVectors = (UP_VEC, DOWN_VEC, LEFT_VEC, RIGHT_VEC) = ((-1, 0), (1, 0), (0, -1), (0, 1))
vecIndex = [UP, DOWN, LEFT, RIGHT] = range(4)
timeLimit = 0.2
allowance = 0.05
depth_limit = 3

class State2048:
    def __init__(self, grid, parent = None, action = None):
        self.grid = grid
        self.parent = parent
        self.action = action
        
        self.children_playerAI = []
        self.children_computerAI = []
        
        self.depth = self.calculate_depth()
        self.factor = 1
        
    def calculate_utility(self):
        reward = 0
        penalty = 0
        
        # Heuristic 1: reward
        for cell_value in self.grid.map:
            if cell_value == 0:
                reward += 1
        
        # Heuristic 2: reward
        weights = np.array([[6, 5, 4, 3], [5, 4, 3, 2], [4, 3, 2, 1], [3, 2, 1, 0]])
        heuristic = weights * np.exp(np.log2(np.array(self.grid.map)))
        #heuristic = weights * np.log2(np.array(self.grid.map))

        reward += sum(sum(heuristic))
        
        
        # Heuristic 3: penalty
        penalty = self.calculate_penalty_neighborsDifference()
        
        utility = reward - penalty
        
        return utility 
        
    def expand_playerAI(self):
        available_moves = self.grid.getAvailableMoves()
        for v in available_moves:
            grid_clone = self.grid.clone()
            grid_clone.move(v)
            new_state = State2048(grid_clone, self, v)
            
            self.children_playerAI.append(new_state)
        
    def expand_computerAI(self):
        available_cells = self.grid.getAvailableCells()        
        for cell in available_cells:   
            grid_clone = self.grid.clone()
            # Every time, add two children for one empty cell
            # So the length of children_computerAI must be a even number
            grid_clone.setCellValue(cell, 2)
            new_state_1 = State2048(grid_clone, self)
            new_state_1.set_factor(0.9)
            self.children_computerAI.append(new_state_1)

            grid_clone.setCellValue(cell, 4)
            new_state_2 = State2048(grid_clone, self)
            new_state_2.set_factor(0.1)
            self.children_computerAI.append(new_state_2)

    def generate_new_tile(self):
        if randint(0, 99) < 100 * defaultProbability:
            return 2
        else:
            return 4 
 
    # Heuristics       
    def calculate_penalty_neighborsDifference(self):
        penalty = 0
        filled_cells = self.get_filled_cells()
        for pos in filled_cells:
            # Neighbors: Up Down Left Right
            n1 = [pos[0], pos[1]-1]
            n2 = [pos[0], pos[1]+1]
            n3 = [pos[0]-1, pos[1]]
            n4 = [pos[0]+1, pos[1]]
            neighbors = [n1, n2, n3, n4]
            
            for n_pos in neighbors:
                if not self.grid.crossBound(n_pos):
                    penalty += abs(self.grid.map[pos[0]][pos[1]] - self.grid.map[n_pos[0]][n_pos[1]])
            
        return penalty
    
    def get_filled_cells(self):
        filled_cells = []

        for x in range(self.grid.size):
            for y in range(self.grid.size):
                if self.grid.map[x][y] != 0:
                    filled_cells.append([x, y])

        return filled_cells        
            
    def calculate_depth(self):    
        depth = 0
        cur = self
        while cur.parent is not None:
            cur = cur.parent
            depth += 1
        return depth
        
    def depth_check(self):
        if self.depth < depth_limit:
            return True
        else:
            return False
        
    def set_factor(self, val):
        self.factor = val
    
class PlayerAI(BaseAI):        
    def getMove(self, grid):
        initial_state = State2048(grid)
        child = self.decision(initial_state)
 
        action = child.action
        
        return action
    
    def decision(self, state_2048):
        # Baic
        #child, tmp = self.maximize(state_2048)
        
        # Alpha-Beta pruning
        child, tmp = self.maximize_prune(state_2048, float('-Inf'), float('Inf'))
        
        return child
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    # Basic expectiminimax
    # (Merge Chance step in Minimize step)
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    def maximize(self, state_2048):
        if not state_2048.grid.canMove() or not state_2048.depth_check():
            utility = state_2048.calculate_utility()
            return None, utility
        else:
            max_child, max_utility = None, float('-Inf')
            state_2048.expand_playerAI()
            for child in state_2048.children_playerAI:
                tmp, utility = self.minimize(child)
                
                if utility > max_utility:
                    max_child, max_utility = child, utility

            return max_child, max_utility
        
    def minimize(self, state_2048):
        if not state_2048.grid.canMove() or not state_2048.depth_check(): 
            utility = state_2048.calculate_utility()
            return None, utility
        else:
            min_child, min_utility = None, float('Inf')
            state_2048.expand_computerAI()
            utility = 0
            for child in state_2048.children_computerAI:
                # Expectiminimax
                if child.factor == 0.9:
                    tmp, u = self.maximize(child)
                    u *= child.factor # Odd index corresponds to a weight of 0.9
                    print(child.fator)
                    utility += u
                if child.factor == 0.1:
                    tmp, u = self.maximize(child)
                    u *= child.factor  # Even index corresponds to a weight of 0.1
                    utility += u
            
                    if utility < min_utility:
                        min_child, min_utility = child, utility

            return min_child, min_utility
        
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    # Alpha-Beta pruning
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    def maximize_prune(self, state_2048, alpha, beta):
        if not state_2048.grid.canMove() or not state_2048.depth_check():
            utility = state_2048.calculate_utility()
            return None, utility
        else:
            max_child, max_utility = None, float('-Inf')
            
            state_2048.expand_playerAI()
            for child in state_2048.children_playerAI:
                tmp, utility = self.minimize_prune(child, alpha, beta)
                
                if utility > max_utility:
                    max_child, max_utility = child, utility
                if max_utility >= beta:
                    break
                if max_utility > alpha:
                    alpha = max_utility

            return max_child, max_utility
        
    def minimize_prune(self, state_2048, alpha, beta):
        if not state_2048.grid.canMove() or not state_2048.depth_check():
            utility = state_2048.calculate_utility()
            return None, utility
        else:
            min_child, min_utility = None, float('Inf')
            
            state_2048.expand_computerAI()
            utility = 0
            for child in state_2048.children_computerAI:
                # Expectiminimax
                if child.factor == 0.9:
                    tmp, u = self.maximize_prune(child, alpha, beta)
                    u *= child.factor # Odd index corresponds to a weight of 0.9
                    utility += u
                if child.factor == 0.1:
                    tmp, u = self.maximize_prune(child, alpha, beta)
                    u *= child.factor  # Even index corresponds to a weight of 0.1
                    utility += u
                    
                    if utility < min_utility:
                        min_child, min_utility = child, utility
                    if min_utility <= alpha:
                        break
                    if min_utility < beta:
                        beta = min_utility

            return min_child, min_utility
        

if __name__ == '__main__':
    
    g = Grid()
    g.map[0][0] = 2
    g.map[0][1] = 4
    g.map[0][2] = 2
    g.map[0][3] = 4
    
    g.map[1][0] = 4
    g.map[1][1] = 2
    g.map[1][2] = 4
    g.map[1][3] = 2
    
    g.map[2][0] = 2
    g.map[2][1] = 4
    g.map[2][2] = 2
    g.map[2][3] = 4
    
    
    state = State2048(g)
    
    player_AI = PlayerAI()
    action = player_AI.getMove(g)
    #print(action)



