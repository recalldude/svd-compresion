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
    print('ok+1 :', S[k+1, k+1])
    return approx.astype(int)


def makeAnApprox(imgPath, krange):
    fig = plt.figure()
    img, R, G, B, h, w = openImage(imgPath)
    j = 0
    long = len(krange) + 1
    for i in krange:
        j+=1
        approxR = approx(R, i)
        approxG = approx(G, i)
        approxB = approx(B, i)
        approxRGB = np.zeros((img.shape), dtype=int)
        approxRGB[:,:,0] = np.copy(approxR)
        approxRGB[:,:,1] = np.copy(approxG)
        approxRGB[:,:,2] = np.copy(approxB)
        fig.add_subplot(1, long, j)
        plt.imshow(approxRGB)
        plt.axis("off")
        plt.title("k =" + str(i))
    fig.add_subplot(1,long,j+1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("originale")
    return (img, approxRGB)


    # def psnr(img, approx):
    # j = 0
    # for r in k:
    
    #     # plt.figure(j+1)
    #     # j += 1
    #     plt.imshow(approx)
    #     plt.axis('off')
    #     plt.title('r=' + str(r))
    #     plt.show()


k = (1, 2, 3)
print(len(k))
img, approxRGB = makeAnApprox('chat.jpg', (5, 20, 50, 100))

plt.show()


# approxRGB[:,:,0] = approxR
# approxRGB[:,:,1] = approxG
# approxRGB[:,:,2] = approxB
# plt.imshow(approxRGB)
# plt.show()
# plt.imshow(R)
# plt.show()