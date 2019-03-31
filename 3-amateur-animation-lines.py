####################################################
# A super basic example of linear transformation
# featuring... wait for it.. lines.
#####################################################
import turtle
import math
import random
import time

# mapping
RefY=[[-1,0],
       [0,1]]
RefX=[[1,0],
      [0,-1]]
Shr=[[1,1],
     [0,1]]      
PoS=[[2,0],
     [0,2]]
NeS=[[0.5,0],
     [0,0.5]]
Proj=[[1,0],
      [0,0]]
Rot=[[math.cos(math.radians(90)),-math.sin(math.radians(90))],
     [math.sin(math.radians(90)),math.cos(math.radians(90))]]     
IdM=[[1,0],
     [0,1]]

# setup the window with a background colour
wn = turtle.Screen()
wn.bgcolor("black")

# assign a name to your turtle
lt = turtle.Turtle()
lt.speed(7)

# draw line
def drawLine (lt, x1, y1, x2, y2):
  lt.penup()
  lt.goto (x1, y1)
  lt.pendown()
  lt.goto (x2, y2)
  lt.penup()

def drawRec(lt, x, y, color):  
  lt.color(color)
  drawLine(lt,0,0,0,y)
  drawLine(lt,0,y,x,y)
  drawLine(lt,x,y,x,0)
  drawLine(lt,x,0,0,0)
  
# draw xy coordinates
def drawCoordinates(lt):
  lt.color("white")
  drawLine(lt,-200,0,200,0)
  drawLine(lt, 0,200,0,-200)

# generate vectors
def genVectors(xy,v, minv, maxv):
  del v[:]
  del xy[:]
  for i in range(1):
    xy = [random.randint(minv, maxv),random.randint(minv, maxv)]
    v.append(xy)

# draw linear transformation
def drawLT(X,Y,color):
  lt.color(color)
  result = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*Y)] for X_row in X]
  print(result)
  # starting from origin
  for i in result:
    drawLine(lt, 0,0, i[0],i[1])
    #drawRec(lt, 4, i[0],i[1],color)
 
xy = []
v = list([])

def drawTitle(lt, title):
  lt.color('white')
  style = ('Arial', 12)
  lt.penup()
  lt.fd(-200)
  lt.write(title,font=style)  
  lt.pendown()

# draw standard coord
drawCoordinates(lt)
drawTitle(lt,"Reflection on the x-axis")
# generate random vectors
genVectors(xy, v, 0, 100)
drawLT(v,IdM,"yellow")
drawLT(v,RefX,"blue")

# wait 2 seconds then erase
time.sleep(2)
lt.clear()

# draw standard coord
drawCoordinates(lt)
drawTitle(lt,"Reflection on y-axis")
# generate new set of random vectors
genVectors(xy, v, 0, 100)
drawLT(v,IdM,"yellow")
drawLT(v,RefY,"blue")

time.sleep(2)
lt.clear()

# draw standard coord
drawCoordinates(lt)
drawTitle(lt,"Upscaling")
# generate new set of vectors
genVectors(xy, v, -100, 100)
drawLT(v,IdM,"yellow")
drawLT(v,PoS,"blue")

time.sleep(2)
lt.clear()

drawCoordinates(lt)
drawTitle(lt,"Downscaling")
# generate new set of vectors
genVectors(xy, v, -100, 100)
drawLT(v,IdM,"yellow")
drawLT(v,NeS,"blue")

time.sleep(2)
lt.clear()

# shear
drawCoordinates(lt)
drawTitle(lt,"Shear")
# generate new set of vectors
genVectors(xy, v, -100, 100)
drawLT(v,IdM,"yellow")
drawLT(v,Shr,"blue")

time.sleep(2)
lt.clear()

# rotation
drawCoordinates(lt)
drawTitle(lt,"Rotation 90 degrees")
# generate new set of vectors
genVectors(xy, v, -200, 200)
drawLT(v,IdM,"yellow")
drawLT(v,Rot,"blue")
time.sleep(2)
lt.clear()

# draw standard coord
drawCoordinates(lt)
drawTitle(lt,"Projection")
# generate random vectors
genVectors(xy, v, -200, -200)
drawLT(v,IdM,"yellow")
drawLT(v,Proj,"blue")

# wait 2 seconds then erase
time.sleep(2)
drawTitle(lt,"The end.")

print('Done.')
