import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2



class cyclic_group_order_4():

    def __init__(self):
        self.order = 4
        self.identity = [0.]


    def elements(self):

        return torch.tensor([0., torch.pi/2, torch.pi, torch.pi*(3/2)])

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
        return torch.mm(R_g, x)





def Q_3_a():
    # use the cyclic group and apply a rotation on an image
    # load image
    img = cv2.imread("photo-1592066162862-aad179660a54.png")
    img.imshow()

Q_3_a()