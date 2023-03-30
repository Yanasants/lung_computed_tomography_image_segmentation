# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 08:56:37 2022
@author: Labic
"""

# VERSÃO EDUADO E VITOR 
from keras import backend
import numpy as np

def dice_coef_EV(y_true, y_pred):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (2.*intersection+1)/(backend.sum(y_true_f)+backend.sum(y_pred_f)+1)

# VERSÃO UTILIZADA ATUALMENTE NA PESQUISA GIRINO
from sklearn.metrics import jaccard_score

def jaccard_coef(y_true, y_pred):
    
    # Recebe dois 2d-array de bool
    
    # array_true = asarray(y_true)
    # array_pred = asarray(y_pred)
    iou = jaccard_score(y_true.flatten(),y_pred.flatten())
    return iou


def dice_coef_girino(y_true, y_pred):
    
    # Recebe dois 2d-array de bool
    
    iou = jaccard_coef(y_true, y_pred)
    dice = (2*iou)/(iou+1)
    return dice


def calcula_metricas(Y_test, predicao):
    
    # Recebe predicao e Y_test (no padrao keras, do tipo (78, 256, 256, 1) )
    
    # Reshape e transforma em boolean
    Y_test = Y_test[:,:,:,0]
    Y_test_bool = Y_test > 0.5 #
    
    # Reshape e transforma em boolean
    predicao = predicao[:,:,:,0] #
    predicao_bool = predicao > 0.5 #
    
    #iou = np.zeros(predicao.shape[0])
    #dice = np.zeros(predicao.shape[0])
    
    iou_list = []
    dice_list = []
    
    for i in range(predicao.shape[0]):
        iou_list.append(jaccard_coef(Y_test_bool[i], predicao_bool[i]))
        dice_list.append(dice_coef_girino(Y_test_bool[i], predicao_bool[i]))
    
    return dice_list

# VERSÃO UTILIZADA NA SEGMENTAÇÃO DE PULMÃO
import keras.backend as K

def dice_coef_lung(y_true, y_pred): #Sørensen–Dice coefficient
    y_true_f = K.flatten(y_true) #transformando em unidimensional
    y_pred_f = K.flatten(y_pred) #transformando em unidimensional
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + \
        K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())    

# Dice similarity function
def online_dice(pred, true, k = 1):
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice
