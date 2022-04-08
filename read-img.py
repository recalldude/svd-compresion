from select import select
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
    plt.xlabel('Valeur de k')
    plt.ylabel('PSNR moyenn')
    plt.title('courbe moyenne psnr')
    return (img, approxRGB, w, h)


def getAvgPsnr(i, img, R, G, B, width, heigth):
    psnrR = psnr(img[:,:,0], R, width, heigth)
    psnrG = psnr(img[:,:,1], G, width, heigth)
    psnrB = psnr(img[:,:,2], B, width, heigth)
    avgPsnr = np.average((psnrR, psnrG, psnrB))
    return avgPsnr

def menu():
    print('donnez un tuple de valeur k :ex (5, 20, 100)')
    k = eval(input())
    while not(type(k) is tuple):
        print('erreur : donnez un tuple , ex: (1,2,3)')
        k = eval(input())
    print('select a picture : ')
    print('1.cat')
    print('2.colors')
    print('3.hamster')
    print('4.parrots')
    print('5.import my image')
    select = eval(input())
    while not(select in (1,2,3,4)):
        print('please select a correct picture')
        print('select a picture : ')
        print('1.cat')
        print('2.colors')
        print('3.hamster')
        print('4.parrots')
        select = eval(input())
    if(select == 1):
        return k,'chat.jpg'
    elif(select == 2):
        return k,'couleurs.jpg'
    elif(select == 3):
        return k,'demo_2.jpg'
    elif(select == 4):
        return k,'inseparables.jpg'
    elif(select == 5):
        print('enter image path')
        imgPath = input()
        return k, imgPath

k,imgPath = menu()
img, approxRGB, width, heigth = makeAnApprox(imgPath, k)
print('width =', width)
print('heigth =', heigth)
plt.show()


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





