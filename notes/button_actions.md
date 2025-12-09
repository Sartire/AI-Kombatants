
convert to discrete action space (?)

per manual

from frame testing, relevant buttons are 

[0, 1, 3, 4, 5, 6, 7, 8]

identify button mapping based on the genesis manual: https://www.manua.ls/sega/mortal-kombat-ii/manual?p=8

down: button 5
up: button 4

A: button 1
B: button 0
C: button 8

start: button 3
left: button 6
right: button 7



moves are: None, U, D, L, R + Start + [A or B or C] * [none, U, D, L, R]

6 + 15 = 21
Discrete(21) action space
