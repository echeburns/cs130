####################################################
# A super basic example of linear transformation
# featuring... wait for it.. dots.
#####################################################
import turtle
import math
import random
import time

# permutation - interchage x and y
A = [[0, 1], 
     [1, 0]]

I = [[1, 0], 
     [0, 1]]


xy = [[-4, -4, -4, -4, -4, -4, -4, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2,
  -2, -2, -2, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  1,
   1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,
   3,  3,  4,  4,  4,  4,  4,  4,  4],
 [-3, -2, -1,  0,  1,  2,  3, -3, -2, -1,  0,  1,  2,  3, -3, -2, -1,  0,
   1,  2,  3, -3, -2, -1,  0,  1,  2,  3, -3, -2, -1,  0,  1,  2,  3, -3,
  -2, -1,  0,  1,  2,  3, -3, -2, -1,  0,  1,  2,  3, -3, -2, -1,  0,  1,
   2,  3, -3, -2, -1,  0,  1,  2,  3]]

# adjust the distance between dots
for i in range(63):
    xy[0][i] = xy[0][i] * 60
    xy[1][i] = xy[1][i] * 60

print("hello")

# setup the window with a background colour
wn = turtle.Screen()
wn.bgcolor("black")

# assign a name to your turtle
pong = turtle.Turtle()
pong.speed('fastest')

# peform matrix multiplication
def matmult(a,b):
    zip_b = zip(*b)
    # uncomment next line if python 3 : 
    # zip_b = list(zip_b)
    return [[sum(ele_a*ele_b for ele_a, ele_b in zip(row_a, col_b)) 
             for col_b in zip_b] for row_a in a]

colors=['black','yellow','cyan','magenta','blue', 'pink', 'red', 'purple', 'green', 'orange', 'light blue']
def matdraw(a, c):
  for i in range(63):
    pong.setposition(a[0][i],a[1][i])
    pong.dot(6, colors[c])

T = A
# add intermediate transformation in 4 steps 
def trans() :
  n = 4 # factor in the distance between dots
  u = xy
  for i in range(n+1):
    matdraw(u,0)
    j = i/float(n)
    print(j)
    T = [[1-j, j], 
         [j, 1-j]]
    print(T)
    u = matmult(T, xy)
    print(u)
    matdraw(u,i+1)
    time.sleep(5) 
   

trans()


print("bye")
