# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 12:59:23 2019

@author: alx34
"""
from tkinter import *
import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

# Clase principal que contendrá todos los llamados
class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # Activa webcam
        self.vid = MyVideoCapture(self.video_source)

        # tomamos características de frame para construir panel
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack(side = LEFT)
        
        self.xroi = int(np.floor((self.vid.width/2)-128))
        self.yroi = int(np.floor((self.vid.height/2)-128))
        
        self.canvas2 = tkinter.Canvas(window, width = 512, height = 512)
        self.canvas2.pack()

        # Botón para tomar snapshots
        icon1=PIL.ImageTk.PhotoImage(file="snapshot.png")
        self.btn_snapshot=tkinter.Button(window, image=icon1, width=64, command=self.snapshot)
        self.btn_snapshot.pack(side = LEFT)
        
        icon2=PIL.ImageTk.PhotoImage(file="prediction.png")
        self.btn_snapshot=tkinter.Button(window, image=icon2, width=64, command=self.prediction)
        self.btn_snapshot.pack(side = LEFT)
        
        icon3=PIL.ImageTk.PhotoImage(file="save.png")
        self.btn_snapshot=tkinter.Button(window, image=icon3, width=64, command=self.saving)
        self.btn_snapshot.pack(side = LEFT)
        
        self.interpreter = Interpreter(model_path="saved_model/MobileNetV2_PlantVillage_Tomato_full.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        self.objects = ('Bacterial_spot', 'Early_blight', 'Healthy', 'Late_blight', 'Leaf_mold', 'Mosaic_virus','Septoria', 'Spider_mites', 'Target_spot', 'Yellow_leaf_curl_virus')
        
        self.font                   = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText = (10,30)
        self.fontScale              = 1
        self.fontColor              = (228,200,50)
        self.lineType               = 2
        
        # Después de ser llamado 1 ves, el método esperará un delay y repetirá el proceso
        self.delay = 1 # en milisegundos
        self.proc = 0
        self.update()

        self.window.mainloop()
    
    def snapshot(self):
        self.proc = 1
        
    def prediction(self):
        self.proc = 2
        
    def saving(self):
        self.proc = 3

    def update(self):
        # toma nuevo frame de webcam
        ret, frame = self.vid.get_frame()
        frame = cv2.rectangle(frame, (self.xroi,self.yroi), (self.xroi+224,self.yroi+224), (255, 0, 0),2)
        
        if ret:
            if self.proc == 1: # Caso snapshot
                self.ROI = frame[self.yroi:self.yroi+224,self.xroi:self.xroi+224,:]
                self.photo2 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.ROI))
                self.canvas2.create_image(0, 0, image = self.photo2, anchor = tkinter.NW)
                self.proc = 0
            elif self.proc == 2: #Caso prediction
#                self.ROI = cv2.cvtColor(self.ROI, cv2.COLOR_RGB2BGR)
                self.IMG = np.expand_dims(self.ROI, axis=0)
                self.IMG=np.float32(self.IMG)/255
                self.interpreter.set_tensor(self.input_details[0]['index'], self.IMG)
                self.interpreter.invoke()
                answer = self.interpreter.get_tensor(self.output_details[0]['index'])
                answer = np.array(answer).ravel()
                x = np.argmax(answer)
                self.pred = self.objects[x]
                if np.max(answer)>0.95: # Nivel de confianza para la deteccion [entre -1 y 1]
                    cv2.putText(self.ROI,self.objects[x], 
                    self.bottomLeftCornerOfText, 
                    self.font, 
                    self.fontScale,
                    self.fontColor,
                    self.lineType)
                else:
                    self.pred = 'None'
                    cv2.putText(self.ROI,'None', 
                    self.bottomLeftCornerOfText, 
                    self.font, 
                    self.fontScale,
                    self.fontColor,
                    self.lineType)
                self.photo2 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.ROI))
                self.canvas2.create_image(0, 0, image = self.photo2, anchor = tkinter.NW)
                self.proc = 0
            elif self.proc == 3:
                cv2.imwrite("frame-"+ self.pred + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(self.ROI, cv2.COLOR_RGB2BGR))
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                self.proc = 0

            else:
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            
            
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame)) 
        self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=1):
        # Se prueba fuente de video
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("No fue posible encontrar objeto de video", video_source)

        # Tomamos dimensiones de video
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Si existe captura de webcam, convierte de BGR a RGB
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Al cerrar la ventana debe desactivar la webcam
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Crea ventana y pasa los parámetros para su creación
App(tkinter.Tk(), "PlantVillage Tomato disease classifier")