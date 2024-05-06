import tkinter as tk
import os
from PIL import Image
import numpy as np
m = tk.Tk()

folder = "CIFAR10/data/"
images = {}
for image_name in os.listdir(folder):
    image = Image.open(folder + image_name)
    if image.mode == 'L':
        image = image.convert(mode='RGB')
    images[image_name[:image_name.index(".")]] = image

mask = [[0 for j in range(32)] for i in range(32)]
save_folder = "CIFAR10/masks/"

# y, x
class ToggleMask():
    def __init__(self, i, j):
        self.i = i
        self.j = j
    
    def __call__(self):
        global mask
        global grid
        mask[self.i][self.j] += 1
        mask[self.i][self.j] %= 2
        if mask[self.i][self.j] == 0:
            grid[self.i][self.j][0].config(background=_from_rgb(images[list(images.keys())[curr_image]].getpixel((self.j, self.i))))
        else:
            grid[self.i][self.j][0].config(background="black")

# https://stackoverflow.com/questions/51591456/can-i-use-rgb-in-tkinter
def _from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb   

curr_image = 0

grid = []
for y in range(32):
    grid.append([])
    for x in range(32):
        i = tk.PhotoImage(width=1, height=1)
        grid_button = tk.Button(m, command = ToggleMask(y, x), height = 12, width = 12, compound='c', image = i, highlightthickness = 0, bd = 0)
        grid_button.config(background = _from_rgb(images[list(images.keys())[curr_image]].getpixel((x, y))))
        grid_button.grid(row = y, column = x)
        grid[-1].append((grid_button, i))

def next_image():
    global curr_image
    global mask
    global grid
    global images
    curr_image += 1
    curr_image %= len(images.keys())
    mask = [[0 for j in range(32)] for i in range(32)]
    for y in range(32):
        for x in range(32):
            grid_button = grid[y][x][0]
            grid_button.config(background=_from_rgb(images[list(images.keys())[curr_image]].getpixel((x, y))))

def save_mask():
    np.save(f"{save_folder}{list(images.keys())[curr_image]}.npy", np.array(mask))

next_button = tk.Button(m, text = "N", command = next_image)
next_button.grid(row = 32, column = 32)
save_button = tk.Button(m, text = "S", command = save_mask)
save_button.grid(row = 32, column = 33)

m.mainloop()