#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 16:36:48 2018

@author: jinzhao
"""

from Grid_3 import Grid

g = Grid()
g.map[0][0] = 2
g.map[1][0] = 2
g.map[3][0] = 4

empty = g.getAvailableCells()
print(empty)


def test_fun():
    return 1, 2

b = test_fun()[1]

print(b)

class test_class():
    def __init__(self):
        self.val = 100
        
    def get_self(self):
        
        return self
    
a = test_class()

b = a.get_self()
print(b.val)




import numpy as np

print(np.exp(np.log2(0)))


