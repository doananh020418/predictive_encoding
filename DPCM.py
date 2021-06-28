# def OverflowX(x, High, Low):
#     if (x > High):
#         return High
#     elif (x < Low):
#         return Low
#     else:
#         return x
#
# def MSE(YOrigi, YRestr, height, width):
#     size = height * width
#     sum = 0
#     for i in range(size):
#         temp = (YOrigi[i] - YRestr[i]) * (YOrigi[i] - YRestr[i])
#         sum += temp
#     mean = sum / size
#     return mean
#
#
# def ErrorQuantity(X, bits):
#     X = X + 255
#     X = X / 2
#     X = np.floor(X / pow(2, 8 - bits))
#     X = X * pow(2, 8 - bits)
#     X = OverflowX(X, 255, 0)
#     return X
#
#
# def invErrorQuantity(X, bits):
#     X = X * 2
#     X = X - 255
#     return X
#
#
# def DPCM_Pixel(YOrigi, YError, YRestr, pre, j, bits):
#     if j == 0:
#         E_temp = 128 - YOrigi[pre + j]
#         E_temp = ErrorQuantity(E_temp, bits)
#         YError[pre + j] = E_temp
#         E_temp = invErrorQuantity(E_temp, bits)
#         R_temp = 128 - E_temp
#         R_temp = OverflowX(R_temp, 255, 0)
#         YRestr[pre + j] = R_temp
#     else:
#         E_temp = YOrigi[pre + j] - YRestr[pre + j - 1]
#         E_temp = ErrorQuantity(E_temp, bits)
#         YError[pre + j] = E_temp
#         E_temp = invErrorQuantity(E_temp, bits)
#         R_temp = E_temp + YRestr[pre + j - 1]
#         R_temp = OverflowX(R_temp, 255, 0)
#         YRestr[pre + j] = R_temp
#     return YError,YRestr
#
# def RDPCM_Pixel(YError, YRestr, pre, j, bits):
#     if j == 0:
#         E_temp = YError[pre + j]
#         E_temp = invErrorQuantity(E_temp, bits)
#         R_temp = 128 - E_temp
#         R_temp = OverflowX(R_temp, 255, 0)
#         YRestr[pre + j] = R_temp
#     else:
#         E_temp = YError[pre + j]
#         E_temp = invErrorQuantity(E_temp, bits)
#         R_temp = E_temp + YRestr[pre + j - 1]
#         R_temp = OverflowX(R_temp, 255, 0)
#         YRestr[pre + j] = R_temp
#     return YRestr
#
# def DPCM(YOrigi,height, width, bits):
#     YError = []
#     YRestr = []
#     for i in range(height):
#         for j in range(width):
#             e,r = DPCM_Pixel(YOrigi, YError, YRestr, i * width, j, bits)
#             YError.append(e)
#             YRestr.append(r)
#     return YError,YRestr
#
# def RDPCM(YError, height, width, bits):
#     YRestr = []
#     for i in range(height):
#         for j in range(width):
#             r = RDPCM_Pixel(YError, YRestr, i * width, j, bits)
#             YRestr.append(r)
#     return YRestr
#
# def PSNR(YOrigi, YRestr, height, width):
#     fmax = pow(2, 8) - 1
#     a = fmax * fmax
#     mean_se = MSE(YOrigi, YRestr, height, width)
#     peak_SNR = 10 * np.log10(a / mean_se)
#     return peak_SNR

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huffman import *
from zig_zag_scan import *

def showImg(img):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()


def MSE(img, new_img):
    return np.square(img - new_img).mean()


def runLength_decoding(stream):
    decoded = []
    for i in range(0, len(stream), 2):
        for j in range(int(stream[(i) + 1])):
            decoded.append(np.float32(stream[i]))
    return (decoded)


def runLength_encoding(message):
    encoded_message = []
    i = 0
    message = np.array(message).flatten()
    while (i <= len(message) - 1):
        count = 1
        ch = message[i]
        j = i
        while (j < len(message) - 1):
            if (message[j] == message[j + 1]):
                count = count + 1
                j = j + 1
            else:
                break
        encoded_message.append(ch)
        encoded_message.append((count))
        i = j + 1
    return encoded_message


def padding(img, block_size):
    new_img_width = np.ceil(img.shape[0] / block_size)
    new_img_height = np.ceil(img.shape[1] / block_size)
    new_img_shape = (new_img_width.astype(np.uint8) * block_size, new_img_height.astype(np.uint8) * block_size)
    mask = np.ones(new_img_shape).astype(np.uint8) * 255
    mask[:img.shape[0], :img.shape[1]] = img
    return mask


def QMatrix_generator(Q):
    Tb = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                   [12, 12, 14, 19, 26, 58, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77],
                   [24, 35, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]])
    if (Q < 50):
        S = 5000 / Q
    else:
        S = 200. - 2 * Q
    Ts = np.floor((S * Tb + 50) / 100)
    Ts[Ts == 0] = 1
    return Ts


def DCT(blocks):
    DCToutput = []
    for block in blocks:
        block = np.float32(block)
        DCToutput.append(cv2.dct(block))
    return DCToutput


def IDCT(blocks):
    IDCToutput = []
    for block in blocks:
        IDCToutput.append((cv2.idct(block)))
    return IDCToutput


def block_quantization(blocks, delta):
    q_matrix = QMatrix_generator(delta)

    for block in blocks:
        for i in range(block.shape[0]):
            for j in range(block.shape[0]):
                block[i, j] = np.around(block[i, j] / q_matrix[i, j])
    return blocks

def single_block_quantization(block, delta):
    q_matrix = QMatrix_generator(delta)
    for i in range(block.shape[0]):
        for j in range(block.shape[0]):
            block[i, j] = np.around(block[i, j] / q_matrix[i, j])
    return block
def get_img(blocks, img_shape):
    row = 0
    rowNcol = []
    width = img_shape[1]
    block_size = 8
    for j in range(int(width / block_size), len(blocks) + 1, int(width / block_size)):
        rowNcol.append(np.hstack(blocks[row:j]))
        row = j
    res = np.vstack(rowNcol)
    res[res>255]=255
    res[res<0]=0

    # showImg(res)
    return res


def split_into_blocks(img2, block):
    if (img2.shape[0] % block != 0 or img2.shape[1] % block != 0):
        img2 = padding(img2, block)

    width = img2.shape[1]
    height = img2.shape[0]
    currY = 0
    sliced = []
    for i in range(block, height + 1, block):
        currX = 0
        for j in range(block, width + 1, block):
            sliced.append(img2[currY:i, currX:j] - np.ones((8, 8)) * 128)
            currX = j
        currY = i
    return sliced


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


from scipy.stats import entropy


def resh(img):
    out = []
    block_size = 8
    for i in range(0, len(img) - block_size * block_size + 1, block_size * block_size):
        out.append(np.array(img[i:i + block_size * block_size]).reshape((block_size, block_size)))
    return out


def de_quantization(blocks, delta):
    q_matrix = QMatrix_generator(delta)
    de_quantized = []
    for block in blocks:
        de_quantized.append(np.multiply(block, q_matrix))
    return de_quantized

def single_block_de_quantization(block, delta):
    q_matrix = QMatrix_generator(delta)
    return np.multiply(block, q_matrix)


def OverflowX(block, High, Low):
    for x in block:
        if (x > High):
            x = High
        elif (x < Low):
            x =  Low
    return block

def dpcm(blocks, delta):
    err = []
    recontruct = []
    for i, block in enumerate(blocks):
        if i == 0:
            e = block
            e = single_block_quantization(e,delta)
            err.append(e)
            recontruct.append(block)
        else:
            tmp_err = single_block_quantization(block - recontruct[i - 1], delta)
            err.append(tmp_err)
            re = recontruct[i - 1] + single_block_de_quantization(tmp_err, delta)
            recontruct.append(re)

    return np.ceil(np.asarray(err).astype(int)/1)*1


def inv_dpcm(err, delta):
    reconstruct = []
    for i, e in enumerate(err):
        if i == 0:
            err = single_block_de_quantization(e,delta)
            reconstruct.append(err)
        else:
            re = single_block_de_quantization(e, delta) + reconstruct[i - 1]
            reconstruct.append(re)
    return reconstruct
def encoder(img, delta):
    block_size = 8
    blocks = split_into_blocks(img, block_size)
    dct = DCT(blocks)
    #q_dct = block_quantization(dct, delta)
    #print(dct)
    q_pred = dpcm(dct,delta)
    zz= []
    for block in q_pred:
        zz.append(zigzag(block))
    RL = runLength_encoding(np.array(zz))
    return RL

def decoder(bit_stream, delta):
    RLD = runLength_decoding(bit_stream)
    #out = resh(RLD)
    inv_zz = []
    block_size = 8
    for i in range(0, len(RLD) - block_size * block_size + 1, block_size * block_size):
        inv_zz.append(inverse_zigzag(np.array(RLD[i:i + block_size * block_size]), 8, 8))
    q_reconstruct = inv_dpcm(inv_zz, delta)
    #q_reconstruct = de_quantization(q_reconstruct,delta)
    out_idct = IDCT(q_reconstruct)
    #out_dq = de_quantization(out, delta)

    # get_img(out_idct)
    return out_idct

def Entropy(data):
    data = data.flatten()  # Chuyen ve 1 chieu

    pd_series = pd.Series(data)
    counts = pd_series.value_counts()
    data_entropy = entropy(counts)

    return data_entropy

block = 8
# file_name = 'data/uncompressed.bmp'
# img = cv2.imread(file_name, 0)
# img = padding(img, block)
# # cv2.imwrite("land2_gray_org.bmp",img)
# # cv2.imshow("origin",img)
# img_shape = img.shape
# print(img_shape)


import os

# file_size = os.path.getsize(file_name)
# Q = []
# psnr = []
# deltas = [90]
# ratio = []
# for delta in deltas:
#     bit_stream = encoder(img, delta)
#     HC = HuffmanCoding()
#     bit = HC.compress(bit_stream)
#
#     file1 = open("data/bit_stream" + str(delta) + ".txt", "wb")
#     file1.write(bit)
#     file1.close()
#
#     decoded_bit_stream = HC.decompress("data/bit_stream" + str(delta) + ".txt")
#
#     decoded_img = decoder(decoded_bit_stream, delta)
#     reconstructed_image = get_img(decoded_img, img_shape)
#     plt.imsave("data/reconstructed_image" + str(delta) + ".png", reconstructed_image,cmap = 'gray')
#     img2 = cv2.imread("data/reconstructed_image" + str(delta) + ".png", 0)
    # cv2.imshow('hjhj',img2)
    # cv2.waitKey(0)
#     Q.append(abs(Entropy(img) - Entropy(img2)))
#     psnr.append(PSNR(img, decoded_img))
#     file_size_2 = os.path.getsize("bit_stream" + str(delta) + ".txt")
#     ratio.append(file_size / file_size_2)
#
# print(Q)
# print(psnr)
# print(ratio)
# plt.figure("Rate-Distortion Optimization")
# plt.plot(ratio, psnr)
#
# plt.show()
def save_img(path,delta = 20):
    img = cv2.imread(path, 0)
    img = padding(img, block)
    bit_stream = encoder(img, delta)
    HC = HuffmanCoding()
    bit = HC.compress(bit_stream)

    file1 = open("data/bit_stream" + str(delta) + ".txt", "wb")
    file1.write(bit)
    file1.close()

    decoded_bit_stream = HC.decompress("data/bit_stream" + str(delta) + ".txt")

    decoded_img = decoder(decoded_bit_stream, delta)
    reconstructed_image = get_img(decoded_img, img.shape)
    plt.imsave("data/reconstructed_image" + str(delta) + ".png", reconstructed_image, cmap='gray')
    return os.path.join('./','data/reconstructed_image'+str(delta)+".png")