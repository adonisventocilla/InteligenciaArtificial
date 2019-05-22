#Link del tutorial en YouTube: https://www.youtube.com/watch?v=Twoem98AOAc
#importamos los paquetes para analizar 
import cv2                              #Libreria openCV
import numpy as np                      #Libreria numpy(matrices...)
import math as m                        #Libreria math(matematica)
from matplotlib import pyplot as plt    #Libreria pyplot(graficación)

"""
PRE-PROCESAMIENTO
"""
for i in range(0,4):
    # Carga de imagenes
    directorio = "./regionPlacas/recorte (%d).jpg" %(i+1)
    a = cv2.imread(directorio)
    """
    AUN FALTA ARREGLAR ESTA PARTE... 
    """

    #Extraccion de capas de la imagen
    rgB = np.matrix(a[:,:,0])
    rGb = np.matrix(a[:,:,1])
    Rgb = np.matrix(a[:,:,2])

    #Definicion de combinaciones RGB para trabajar
    I = cv2.absdiff(rGb,rgB)    #Se coge la capa verde y se le resta la capa azul
    II = I
    cv2.imshow('Imagen en escala de grises',II*255)
    cv2.waitKey(0)

    #Binarizacion inicial de la imagen
    #Se recorre por filas y columnas cada pixel
    [fil,col] = I.shape
    for o in range(0,fil):
        for oo in range(0,col):
            if I[o,oo]<80:      #Para cada pixel que sea menor que 80 se convierte a 0
                I[o,oo] = 0
    for o in range(0,fil):
        for oo in range(0,col):
            if I[o,oo]>0:      #Para cada pixel que sea mayor que 0 se convierte a 1
                I[o,oo] = 1
    
    cv2.imshow('Imagen en blanco y negro',I*255)
    cv2.waitKey(0)

    #Transformaciones morfologicas
    if i==0:
        #Se crea street cuadrado : 'se' para closing y 'se2' para dilation
        se = np.ones((50,50), np.uint8)
        se2 = np.ones((10,10), np.uint8)

    closing = cv2.morphologyEx(I,cv2.MORPH_CLOSE,se)     #Cierre de la imagen
    dilation = cv2.dilate(closing,se2,1)                 #Dilatacion de la imagen
    cv2.imshow('Enmarcado del cuadro de placa',dilation*255)
    cv2.waitKey(0)

    #Para encontrar cajas como boundingBox
    #--1 Encontrar contornos
    #'S' es la imagen de salida
    #'contours' las coodenadas de los contornos (IMPORTANTE)
    #'hierarchy' las relaciones de jerarquias ante los contornos
    S, countours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #--2 Extraer contornos
    cnt = countours[:]            #Se extraen los contornos
    num = len(cnt)                #Se calcula numero de contornos
    #--3 Calcular rectangulo mas grande
    box = np.zeros((num,4))       #Para inicializar el box
    for j in range(0,num):
        box[j,:] = cv2.boundingRect(cnt[j])     #Creacion de cajas para regiones    boundingRect: sirve para extraer las regiones de interes d ela imagen
    L = np.zeros((num, 4))                      #Inicializacion de L
    Max = [0,0]                                 #Inicializacion Max
    for j in range(0,num):                     #Para calcular la maxima region
        L[j,:] = box[j]
        if L[j,2] > Max[1]:                     #Se cuenta sobre los indices
            Max = [j, L[j,2]]
    BOX = box[Max[0],:]

    #MASCARA
    #x = desde la esquina superior izquierda hasta la esquina superior derecha
    #y = desde la esquina superior izquierda hasta la esquina inferior izquierda
    b = a[int(BOX[1]):int(BOX[1]+BOX[3]),int(BOX[0]):int(BOX[0]+BOX[2]),:]

    cv2.imshow('Imagen de la placa',b)
    cv2.waitKey(0)

    #Carga de la imagen
    directorio2 = "./blanks/blank (%d).jpg" % (i+1)
    cv2.imwrite(directorio2,b)


