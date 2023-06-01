import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2



class cyclic_group_order_4():

    def __init__(self):
        self.order = 4
        self.identity = [0.]


    def elements(self):

        return torch.tensor([torch.tensor(0.), torch.pi/2, torch.pi, torch.pi*(3/2)])

    def product(self, g1, g2):
        return (g1 + g2) % (2*torch.pi)

    def inverse(self, g):
        return (2*torch.pi - g) % (2*torch.pi)

    def matrix_representation(self, g):
        # g is angle which is element of the group
        return torch.tensor([[torch.cos(g), -torch.sin(g)], [torch.sin(g), torch.cos(g)]])

    def group_action(self, g, x):
        # g is angle which is element of the group
        # x is a vector in R^2
        R_g = self.matrix_representation(g)
        x = torch.tensor(x)
        return torch.mm(R_g, x)


def img_2_grid(img):
    # convert the image to grid map
    # img is a 2D array
    # grid is a line space where the center of the image is (0,0)
    # grid is a 2D array
    n = img.shape[0]
    line = torch.linspace(-1, 1, n)
    step = line[1] - line[0]
    grid = torch.meshgrid(line, line)
    grid = torch.stack([grid[0], grid[1]]).view(2, -1)

    return grid, step

def grid_2_img(original_img, grid, step, n):
    # convert the grid map to image
    # grid is a 2D array
    # img is a 2D array
    grid = torch.round((grid + 1) / step).long()
    img = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            img[grid[0, i*n+j], grid[1, i*n+j]] = original_img[i, j]
    return img






def Q_3_a():
    # use the cyclic group and apply a rotation on an image
    # load image
    img = cv2.imread("Lena-128-128-Original-Lena-image-a-and-its-reconstructions-using-respectively-b.png", cv2.IMREAD_GRAYSCALE)
    n = img.shape[0]
    grid, step = img_2_grid(img)
    # apply rotation on image
    # create cyclic group
    G = cyclic_group_order_4()
    # get elements of the group
    elements = G.elements()
    # get random element of the group
    for g in elements:
        # apply group action on image
        rotated_grid = G.group_action(g, grid)
        rotated_img = grid_2_img(img, rotated_grid, step, n)
        # convert tensor to numpy array
        rotated_img = rotated_img.numpy().astype(np.uint8)

        cv2.imshow("original image", rotated_img)
        cv2.waitKey(0)



Q_3_a()