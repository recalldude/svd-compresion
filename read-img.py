import numpy as np
import matplotlib.pyplot as plt

def openImage(imgPath):
    img = plt.imread(imgPath)
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    heigth = img.shape[0]
    width = img.shape[1]
    return (img,R,G,B,heigth,width)

def approx(A,k):
    U, S, VT = np.linalg.svd(A ,full_matrices=False)
    S = np.diag(S)    
    approx = U[:,:k] @ S[:k,:k] @ VT[:k,:]
    return approx



    # j = 0
    # for r in k:
    
    #     # plt.figure(j+1)
    #     # j += 1
    #     plt.imshow(approx)
    #     plt.axis('off')
    #     plt.title('r=' + str(r))
    #     plt.show()



img, R, G, B, h, w = openImage('demo_2.jpg')
approxR = approx(R, 50)
approxG = approx(G, 50)
approxB = approx(B, 50)
approxRGB = np.zeros((img.shape))
approxRGB[:,:,0] = np.copy(approxR)
approxRGB[:,:,1] = np.copy(approxG)
approxRGB[:,:,2] = np.copy(approxB)
plt.imshow(approxRGB)
plt.show()

# approxRGB[:,:,0] = approxR
# approxRGB[:,:,1] = approxG
# approxRGB[:,:,2] = approxB
# plt.imshow(approxRGB)
# plt.show()
# plt.imshow(R)
# plt.show()