import pyautogui
import time
import random

pyautogui.FAILSAFE = True

while True:
    x, y = pyautogui.position()
    pyautogui.moveTo(x + random.randint(-3, 3), y + random.randint(-3, 3))
    time.sleep(120)
