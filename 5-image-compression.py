import matplotlib.pyplot as plt 
import scipy.linalg as sp
import numpy as np 
import imageio

print("hello")
def svd_compression(A, k):
  U, S, V = np.linalg.svd(A, full_matrices=False)
  print("U: ", U.shape)
  #print(U)
  print("S: ", S.shape)
  #print(S)
  print("V: ", V.shape)
  #print(V)
  # k reconstruction
  M = np.dot(U[:,:k],np.dot(np.diag(S[:k]),V[:k,:]))
  # full reconstruction
  #M = np.dot(U, np.dot(np.diag(S),V))
  return M

def qr_compression(A, k):
  Q, R = np.linalg.qr(A)
  # first k rows of R and first k columns of Q
  M = np.dot(Q[:,:k],R[:k])
  return M

def qrp_compression(A,k):
  #Q, R = np.linalg.qr(A)
  Q, R, P = sp.qr(A, pivoting=True)
  #Q, R = sp.qr(A)
  print("Q: ", Q.shape)
  #print(Q)
  print("R: ", R.shape)
  print(R)
  print("P ",  P.shape)
  print(P)
  
  #print(P)
  r_kl = R[k:]
  r_kr = R[:k]
  print("r_kl ", r_kl.shape)
  print("r_kr ", r_kr.shape)
  q1 = Q[:k,:]
  q2 = Q[:,k:]
  q3 = Q[:,:k]

  print("q1 ", q1.shape)
  print("q2 ", q2.shape)
  print("q3 ", q3.shape)
  #print("A[:,P]")
  #print(A[:,P])
  #print("diagonal P")
  #d = np.diag(P)
  #print(d)
  #M = np.dot(np.dot(Q,R),A[:,P])
  #M = np.dot(np.dot(Q,R),A[:,P])
  #M = np.dot(Q,R)
  #create an identity matrix
  rc = A.shape

  I = np.eye(rc[0])
  IP = I[:,P]
  print("IP ", IP)
  #M = np.dot(Q,R).dot(IP.T)
  # first k rows of R and first k columns of Q
  M = np.dot(Q[:,:k],R[:k]).dot(IP.T)
  
  #print("Diag P ")
  #print(np.diag(P))
  # first k rows of R and first k columns of Q
  # QRP
  #M = np.dot(Q[:,:k],R[:k])
  
  #M = np.dot(np.dot(Q[:,:k],R[:k]),A[:,P])
  #print("allclose: ", np.allclose(Q, R))
  #print("allclose: ", np.allclose(Q @ R, A[:,P]))
  #print("allclose: ", np.allclose(np.dot(Q, R), A[:,P]))
  #print("allclose: ", np.allclose(Q @ R, A[:,P]))
  #print("allclose: ", np.allclose(Q[:,:k] @ R[:k], A[:,P]))
  print("M ", M.shape)
  return M

## start
def my_main():
  img = imageio.imread("lena.pgm")
  img_shape = img.shape
  print("img:",img.shape)

  #A = np.matrix(img)
  A = img
  print(A)
  k = 100

  svd_img = svd_compression(A, k)
  qr_img = qr_compression(A, k)
  qrp_img = qrp_compression(A, k)

  print("SVD equal?", (svd_img.round(0).astype(int)==A).all())
  print(svd_img.round(0).astype(int))

  print("QR equal?", (qr_img.round(0).astype(int)==A).all())
  print(qr_img)
  print("k ", k)
  print("A:",A.shape)
  print("svd_img:",svd_img.shape)
  #print(svd_img)
  print("qr_img:",qr_img.shape)
  #print(qr_img)

  compression_ratio = 100.0* (k*(img_shape[0] + img_shape[1])+k)/(img_shape[0]*img_shape[1])
  print(compression_ratio)

  print("compute for approximation error for k= ", k)
  print("svd_apx: ", np.linalg.norm(A - svd_img)/np.linalg.norm(A))

  print("qr_apx: ", np.linalg.norm(A - qr_img)/np.linalg.norm(A))

  print("qrp_apx: ", np.linalg.norm(A - qrp_img)/np.linalg.norm(A))

"""
  p = plt.imshow(qr_img)
  outfile = "lena_k"+str(k)+"_qr.png"
  print(outfile)
  imageio.imwrite(outfile,qr_img)

  p = plt.imshow(qrp_img)
  outfile = "lena_k"+str(k)+"_qrp.png"
  print(outfile)
  imageio.imwrite(outfile,qrp_img)

  p = plt.imshow(svd_img)
  outfile = "lena_k"+str(k)+"_svd.png"
  print(outfile)
  imageio.imwrite(outfile,svd_img)
"""
my_main() 

def test_lang():
  A = np.array([[1, 2, 3], [
                 4, 5, 6], 
                [7, 8, 9]])

  I = np.array([[1, 0, 0], 
                [0, 1, 0],
                [0, 0, 1]])
  #arrange the diagonal in decreasing order
  # you'll get P

  # 3,1 2
  # 6,4,5
  # 9,7,8

  #0,1,0
  #0,0,1
  #1,0,0

  # transpose
  # 0,0,1
  # 1,0,0
  # 0,1,0

  # P is 2,0,1
  PI = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

  PX = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

  Q, R, P = sp.qr(A, pivoting=True)
  #Q, R = np.linalg.qr(A)
  Q2, R2 = sp.qr(A)
  print("A")
  print(A)
  print("Q ")
  print(Q)
  print("R ")
  print(R)
  print("P ")
  print(P)
  print("QR")
  print(np.dot(Q,R))

  print("ID ", I[:,P])

  print("A[:,P]")
  print(A[:,P])
  #P = [0,1,2]
  #P = [2,0,1]

  print("A[:,P]")
  print(A[:,P])

  print("A[:,P].T")
  print(A[:,P].T)
  print("P.T")
  print(P.T)

  QR = np.dot(Q.T,R[:,P])
  print("QR--X", QR)

  print("diag(P.T")
  print(np.diag(P.T))
  print("diag(P")
  print(np.diag(P).T)


  pt = np.array(P)[np.newaxis]
  print("pt.T")
  print(pt.T)

  print("QRP")
  print(np.dot(Q,R).dot(A))

  print("-----np.qr------")
  Q, R = np.linalg.qr(A)
  print("A")
  print(A)
  print("Q ")
  print(Q)
  print("R ")
  print(R)
  print("QRP")
  print(np.dot(Q,R))
  print("----sp.qr-------")
  Q, R = sp.qr(A)
  print("A")
  print(A)
  print("Q ")
  print(Q)
  print("R ")
  print(R)
  print("QRP")
  print(np.dot(Q,R))

# test_lang()
print("bye")
