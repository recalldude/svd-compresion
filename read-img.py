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
    # print('ok+1 :', S[k+1, k+1])
    return approx.astype(int)

def psnr(img, approxRGB, width, heigth):
    rMax = 255
    e = (np.linalg.norm(img-approxRGB)**2) / (width * heigth)
    return 10*np.log10((rMax**2)/e)
    
def makeAnApprox(imgPath, krange):
    fig = plt.figure()
    img, R, G, B, h, w = openImage(imgPath)
    j = 0
    x = np.array([0])
    y = np.array([0])
    long = len(krange) + 2
    for i in range(max(krange)+1):
        approxR = approx(R, i)
        approxG = approx(G, i)
        approxB = approx(B, i)
        y = np.append(y, getAvgPsnr(i, img, approxR, approxG, approxB, w, h))
        x= np.append(x, i)
        if i in krange: 
            j+=1
            approxRGB = np.zeros((img.shape), dtype=int)
            approxRGB[:,:,0] = np.copy(approxR)
            approxRGB[:,:,1] = np.copy(approxG)
            approxRGB[:,:,2] = np.copy(approxB)
            fig.add_subplot(1, long, j)
            plt.imshow(approxRGB)
            plt.axis("off")
            plt.title("k =" + str(i))
            print('psnr pour k=', i, " : ", psnr(img, approxRGB, w, h))
    fig.add_subplot(1,long,j+1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("originale")
    fig.add_subplot(1, long, j+2)
    plt.plot(x, y)
    plt.title('courbe moyenne psnr')
    return (img, approxRGB, w, h)


def getAvgPsnr(i, img, R, G, B, width, heigth):
    psnrR = psnr(img[:,:,0], R, width, heigth)
    psnrG = psnr(img[:,:,1], G, width, heigth)
    psnrB = psnr(img[:,:,2], B, width, heigth)
    avgPsnr = np.average((psnrR, psnrG, psnrB))
    return avgPsnr


# def makeAnApprox(imgPath, krange):
#     fig = plt.figure()
#     img, R, G, B, h, w = openImage(imgPath)
#     j = 0
#     long = len(krange) + 1
#     for i in krange:
#         j+=1
#         approxR = approx(R, i)
#         approxG = approx(G, i)
#         approxB = approx(B, i)
#         approxRGB = np.zeros((img.shape), dtype=int)
#         approxRGB[:,:,0] = np.copy(approxR)
#         approxRGB[:,:,1] = np.copy(approxG)
#         approxRGB[:,:,2] = np.copy(approxB)
#         fig.add_subplot(1, long, j)
#         plt.imshow(approxRGB)
#         plt.axis("off")
#         plt.title("k =" + str(i))
#         print('psnr pour k=', i, " : ", psnr(img, approxRGB, w, h))
#     fig.add_subplot(1,long,j+1)
#     plt.imshow(img)
#     plt.axis("off")
#     plt.title("originale")
#     return (img, approxRGB, w, h)





# def drawPsnr(krange):
#     x = np.array([0])
#     y = np.array([0])
# def PSNR(original, compressed):
#     mse = np.mean((original - compressed) ** 2)
#     if(mse == 0):  # MSE is zero means no noise is present in the signal .
#                   # Therefore PSNR have no importance.
#         return 100
#     max_pixel = 255.0
#     psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
#     return psnr

    # def psnr(img, approx):
    # j = 0
    # for r in k:
    
    #     # plt.figure(j+1)
    #     # j += 1
    #     plt.imshow(approx)
    #     plt.axis('off')
    #     plt.title('r=' + str(r))
    #     plt.show()



img, approxRGB, width, heigth = makeAnApprox('chat.jpg', (0,15,100))
print('width =', width)
print('heigth =', heigth)
plt.show()


