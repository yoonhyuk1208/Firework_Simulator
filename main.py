import pygame as pg
import pygame.locals
import sys
import random
import time
import ctypes

from Firework import Class as FireClass

pg.init()

FPS = 60
FramePerSec = pg.time.Clock()

window_size = (1280, 720)
Start_of_FireworkZone = 46
Len_of_FireworkZone = 1188/32
G = 9.8

SimulateDisplay = pg.display.set_mode(window_size)
SimulateDisplay.fill((0,0,0))
pg.display.set_caption("Firework Simulator")

while True:

    pg.display.update()

    for event in pg.event.get():

        if event.type == pg.KEYDOWN:
            input_text = event.unicode

            if len(input_text) == 1:
                input_char = ctypes.c_char(input_text.encode('utf-8'))

                returnLocate = ctypes.CDLL("./Calculation/KeyLocation.so")

                result = returnLocate.return_locate(input_char)
                
                if result != -1:
                    Firework_SummonX = Start_of_FireworkZone + Len_of_FireworkZone*result
                    print(Firework_SummonX)

        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()
    
    SimulateDisplay.fill((0,0,0))
    
    #pg.draw.circle(SimulateDisplay,(255,255,255),(400,400),30)

    FramePerSec.tick(FPS)