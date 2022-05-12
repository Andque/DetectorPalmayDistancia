import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import cvzone
import time
import autopy
from mediapipe import solutions as mp
import pyautogui
import keyboard
##########################
wCam, hCam = 1920, 1080

frameRxi = 400  # lado de la derecha
frameRxd = 400  # lado de la izquierda
frameRyi = 200    # lado superior
frameRyd = 200    #Lado inferior
smoothening = 2
velocidadt_real=0
#########################
index_list=[33, 359]
x4=[0, 0]
y4=[0, 0]
x4[1] = 5
mp_face_mesh = mp.face_mesh
############RELLENAR POR USUARIO VALOR EN CM##############################################################################################
dist_real_ojos=9
########################################################################################################################################
dist_real_ojos=dist_real_ojos-0.5 #debido a que la mano se localiza enfrente de la cara y asi se realiza una mejor aproximación de la velocidad
# Webcam
cap = cv2.VideoCapture(2)
cap.set(3, wCam)
cap.set(4, hCam)

# Hand Detector
detector = HandDetector(detectionCon=0.5, maxHands=1)
###PARAMETROS USADOS PARA DISTANCIA, medidos en vida real###
# Find Function
# x is the raw distance y is the value in cm
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C
########################
## parametros para ver fps#########
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
################
ax, ay=0, 0
atime = 0
distancia_pixel = 1
count_puno = 0
puno_abierto = 1
distanceCM=0
###########################
wScr, hScr = autopy.screen.size()
############################# Array donde poner los datos, el primer dato son las veces en las que se ha cerrado el puño
Datos=[[0,0,0]]
#el ultimo dato va a ser las veces que se ha cerrado el pucño
# Loop
with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

    while True:
        success, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hands = detector.findHands(img, draw=False)
        resultsface = face_mesh.process(img_rgb)
        if resultsface.multi_face_landmarks is not None:
            for face_landmarks in resultsface.multi_face_landmarks:
                    for index in index_list:
                        if index == 33 : a = 0
                        if index == 359 : a = 1
                        x4[a] = int(face_landmarks.landmark[index].x * wCam)
                        y4[a] = int(face_landmarks.landmark[index].y * hCam)
                        cv2.circle(img, (x4[a], y4[a]), 1, (0, 255, 255), 2)
                        #distancia
                        distancia_pixel = int(math.sqrt((y4[0] - y4[1]) ** 2 + (x4[0] - x4[1]) ** 2))
        # 11. Frame Rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        #print("Velocidad actualizacion",cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                         ( 255, 0, 0), 3)
        if hands:
            for hand in hands:
                lmList = hands[0]['lmList']
                x, y, w, h = hands[0]['bbox']
                x1, y1 = lmList[5][:2]
                x2, y2 = lmList[17][:2]
                distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
                A, B, C = coff
                distanceCM = A * distance ** 2 + B * distance + C

            # print(distanceCM, distance)

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 3)
                cvzone.putTextRect(img, f'{int(distanceCM)} cm', (x+5, y-10))
        #Empieza la parte del raton
            #Mirar si la palma esta abierta
                fingers= detector.fingersUp(hand)
                #print(fingers)
                ### AQUI VER LA VELOCIDAD
                #Si la palma esta abierta buscar el punto medio de esta y hacer que el raton de la pantalla lo siga
                #El punto medio
                xm = int(x+w/2)
                ym = int(y+(h/2))
                htime = time.time()
                hx = xm
                hy = ym
                At = htime-atime
                Ax = hx-ax
                Ay = hy-ay
                ax = hx
                ay = hy
                atime = htime
               #pasar a distancia real sabiendola relaion de cm a pixel usando los ojos
                velocidadx_real = int(abs((Ax*dist_real_ojos/distancia_pixel)/At))
                velocidady_real = int(abs((Ay*dist_real_ojos/distancia_pixel)/ At))
                velocidadt_real = int(math.sqrt(velocidady_real**2+velocidadx_real**2))
                #print("distancia x real", Ax*dist_real_ojos/distancia_pixel)
                #print("tiempo para velocidad", At)
                print("Velocidad total",velocidadt_real)

                cv2.rectangle(img, (xm, ym), (xm + 5, ym + 5), (0, 0, 0), 3)
                #convertir coordenadas a pantalla
                x3 = np.interp(x1, (frameRxi, wCam - frameRxd), (0, wScr))
                y3 = np.interp(y1, (frameRyi, hCam - frameRyd), (0, hScr))
                cv2.rectangle(img, (frameRxi, frameRyi), (wCam - frameRxd, hCam - frameRyd),
                              (255, 0, 255), 2)
                #hacerlas mas smooth
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                #Mover raton si no es lo de drag

                autopy.mouse.move(wScr - clocX, clocY)
                plocX, plocY = clocX, clocY
                #modo click si cierras el puño ( no contar el pulgar) y para mejor usabilidad si corazon abajo
                if fingers[1] == 1 and fingers[2]==0 and fingers[3] == 1 and fingers[4] == 1 :
                    cv2.circle(img, (xm, ym),
                               15, (0, 255, 0), cv2.FILLED)
                    pyautogui.click(interval=0.2)
                if fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0and fingers[4] == 0:
                    cv2.circle(img, (xm, ym),
                                15, (0, 255, 0), cv2.FILLED)
                    pyautogui.click(interval=0.2)
                    if puno_abierto==1:
                        puno_abierto=0
                        count_puno=count_puno+1
                        print("veces cerrado el puño", count_puno)
                        Datos[0][0]=count_puno
                    #modo click continuo si pones el indice y meñique arriba y el resto abajo y lo dejas presionado
                if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 1:
                    cv2.circle(img, (xm, ym),
                               15, (255, 50, 50), cv2.FILLED)
                    pyautogui.mouseDown(button='left')
                    #modo click continuo si bajas el indice y pulgar abajo y el resto arriba y lo sueltas
                if fingers[0] == 0 and fingers[1] == 0 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] ==1 :
                    cv2.circle(img, (xm, ym),
                               15, (255, 50, 50), cv2.FILLED)
                    pyautogui.mouseUp(button='left')
                #para poner los datos en un excel por si el fisio los quiere
                #para no tener faloso puños abiertos
                if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] ==1 :
                    puno_abierto=1
        cv2.imshow("Image", img)
        Datos.append([velocidadt_real, distanceCM, time.time()])
        cv2.waitKey(1)
        if  keyboard.is_pressed("q"):
            print("Aa")
            print(Datos)
            a = np.array(Datos)
            print(a)
            np.savetxt('data3D.csv', Datos, delimiter=",")
            break  # si presionas q sales y se guarda los datos en el cvs

#una vez que sales del programa
