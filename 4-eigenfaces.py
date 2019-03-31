##############################################################################################################
# Resources:
# https://sandipanweb.wordpress.com/2018/01/06/eigenfaces-and-a-simple-face-detector-with-pca-svd-in-python/
# https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/
# https://medium.com/@MarynaL/eigenfaces-3675c94a7d
# https://towardsdatascience.com/eigenfaces-recovering-humans-from-ghosts-17606c328184
##############################################################################################################
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, imshow, title, savefig
from itertools import chain
import numpy as np
import glob
import imageio

print("hello")

# each folder only contains a single image 1.pgm
# each image is 112x92 pixels
# TODO: arrange the files so that the row index correspond to the
# number assigned to the image.
filenames = [img for img in glob.glob("attr/s*/*.pgm")]
images = [[] for i in range(40)]
for i in range(40):
  print(str(i),filenames[i])
  images[i] = list(chain.from_iterable(imageio.imread(filenames[i])))

# 40 images in a single matrix  
images = np.matrix(images)

print(type(images))
print(images.shape)
#print(images)

def method_2(images):
  # skipped part on centering, etc.
  m = len(images)
  d = 112 * 92
  X = np.reshape(images,(m,d))

  mean = np.mean(X, axis=0)
  centered_data = X-mean

  # perform SVD on the image matrix
  U, Sigma, VT = np.linalg.svd(centered_data)

  print("X:",  X.shape)
  # eigenvector columns of AA^T
  print("U:", U.shape)
  # eigenvalues
  print("Sigma:", Sigma.shape)
  # eigenvector rows of A^TA
  print("V^T:", VT.shape)

  #print(U)
  # inspect sigma to decide on selecting k-values
  print(np.diag(Sigma))
  #print(VT)

  # select 3 principal components
  model = PCA(n_components=3)
  pts = normalize(X)
  model.fit(pts)
  # calculate VTk, the same as extracting V^T for k components
  result = model.transform(pts)
  
  #print(result)
  #print("PCA:",result.shape)
  
  #fig, ax = plt.subplots(1,3)
  # reshaping the matrix to display an image
  for i in range(3):
    ord = model.components_[i]
    img = ord.reshape(112,92)
    #ax[i].imshow(img,cmap='gray')
    tmpfile=str(i)+"_.png"
    print(tmpfile)
    imageio.imwrite(tmpfile,img) 

  # each right singular vector has the same number of entries as
  # there are pixels in an image
  
  # project each of the original images onto the second and third eigenfaces
  # this process reduces the original 112,92 dimensions into two dimension 
  # num_components = 3 # Number of principal components
  #Y = np.matmul(X, VT[:num_components,:].T)
  # arr[:,[1,2]]
  # extract the second and third eigenfaces and project to original images

  # X original image (40,10304)
  print("X:",  X.shape)
  # VT^T (10304, 40) features x images
  print("VT^T:",  VT.T.shape)
  print("VT^T:",  VT.T)
  
  # first three columns
  #print("VT.T[:3,:]:", VT[:3,:].T.shape)
  #print(VT[:3,:].T)
  
  # second and third column
  print("VT.T[1:3,:]", VT[1:3,:].T.shape)
  #print(VT[1:3,:].T)
  Y = np.matmul(X, VT[1:3,:].T)
  print("Y:",  Y.shape)
  print(Y)
  return Y

def plot_Y(Y):
  #scatter plot x - column 0, y - column 1, shown with marker o
  #plt.plot(Y[:, 0], Y[:, 1], 'o', label = 'data')
  #create legend in case you have more than one series
  print(Y)

  # Plot each of the projected faces in a 2D coordinate system
  # and color them by gender
  # female - rows: 35, 34, 3, 6 (file: 8, 10, 32,35 )
  # male - the rest
  # copy female to a new array

  x = np.array(Y[:,0])
  y = np.array(Y[:,1])

  # all
  scat = plt.scatter(x, y, c='k')
  plt.show()
  plt.savefig('plot_all.png')
  scat.remove()

  #attempt at grouping using K-means clustering and graph the 
  Kmean = KMeans(n_clusters=2)
  Kmean.fit(Y)
  arr = Kmean.cluster_centers_
  print(arr)

  plt.scatter(x, y, s =50, c='b')
  c1=plt.scatter(arr[0,0], arr[0,1], s=200, c='c', marker='s')
  c2=plt.scatter(arr[1,0], arr[1,1], s=200, c='m', marker='s')
  plt.show()
  plt.savefig('plot_cluster.png')
  c1.remove()
  c2.remove()

  # TODO: index and file number should be the same when loaded as matrix
  for i in range(40):
    if (i == 35 or i == 34 or i == 3 or i == 6):
      plt.scatter(x[i], y[i], c='r', label="female" if i == 3 else "")
    else :
      plt.scatter(x[i], y[i], c='b', label="male" if i == 1 else "")
  
  scat=plt.legend()
  plt.show()
  plt.savefig('plot_by_gender.png')
  scat.remove()
  
  for i in range(40):
    scat=plt.scatter(x[i], y[i], c=np.random.rand(3,), label=str(i))

  lgd=plt.legend(bbox_to_anchor=(0, 1), loc='lower right', ncol=4)
  plt.show()
  plt.savefig('plot_by_index.png',bbox_extra_artists=(lgd,), bbox_inches='tight')
 
  print("bye")

  
Y = method_2(images)
plot_Y(Y)

# function unused
def method_1(images):
  # perform SVD on A, k=6
  # Principal Component Analysis
  model = PCA(n_components=6)
  pts = normalize(images)
  model.fit(pts)
  pts2 = model.transform(pts)

  fig, ax = plt.subplots(1,6)
  for i in range(6):
    ord = model.components_[i]
    img = ord.reshape(112,92)
    ax[i].imshow(img,cmap='gray')
    tmpfile=str(i)+"_.png"
    print(tmpfile)
    imageio.imwrite(tmpfile,img) 

  # reconstruct the faces
  #fig.savefig("fig.png") 
  #faces_pca = PCA(n_components=20)
  #faces_pca.fit(m)
      
  #components = faces_pca.transform(m)
  #projected = faces_pca.inverse_transform(components)
  #fig, axes = plt.subplots(20,1,figsize=(20,80))
  #for i, ax in enumerate(axes.flat):
  #  ax.imshow(projected[i].reshape(112,92),cmap="gray")
  #fig.savefig("fig.png") 
