import os
import cv2
import numpy as np


#importante que as faces coletadas venham de vídeos diferentes, com condições diferentes de iluminação, variações sutis no ângulo e etc
#quanto mais variada for sua base para cada pessoa, melhor será o resultado


if __name__ == "__main__":

    if os.path.isdir("./data") == False:
        os.mkdir("./data")
    
    video = "camila.mp4"
    diretorio = "./data/Camila"
    frontal_face_cascade = cv2.CascadeClassifier("./haarcascade/haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(video)
    cont_frames = 0
    total_amostras = 250
    
    try:
        os.mkdir(diretorio)
    except OSError:
        local = os.listdir(diretorio)
        local_arquivos = []
        for arq in local:
            if arq[-3::] == "png":
                local_arquivos.append(arq)
        total_amostras += len(local_arquivos)
        cont_frames += len(local_arquivos)
        
    while cont_frames < total_amostras:
        ret, img = cap.read()
        if ret == False:
            cap.release()
        else:
            #img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = frontal_face_cascade.detectMultiScale(img, 1.3, 5)
            if len(faces) == 0:
                continue
            for (x, y, w, h) in faces:
                coord_face = img[y:y+h, x:x+w]
            larg, alt, _ =  coord_face.shape
            if(larg * alt <= 20 * 20): #descarta faces muito pequenas
                continue
            coord_face = cv2.resize(coord_face, (50, 50)) #quanto maior for a largura e a altura, maior será o gasto de memória no treinamento 
            cv2.imwrite(diretorio + "\\" + str(cont_frames)+ ".png", coord_face)
            print(diretorio + "\\" + str(cont_frames)+ ".png")
            cont_frames += 1
    cap.release()
    print(" [*] Concluído!")
