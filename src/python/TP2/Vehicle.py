import numpy as np

class Vehicle:

  def __init__(self, x, y, speed):
    self.x = x
    self.y = y
    self.hist_x = 0
    self.hist_y = 0
    self.updated = True
    self.inactive_counter = 0
    self.speed = speed
    self.img = np.array([])