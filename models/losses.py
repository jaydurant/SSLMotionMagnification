
import os
import numpy as np
import torch
import torch.autograd as ag


def criterion_mag(y, batch_M, texture_AC, texture_BM, motion_BC, criterion):
    loss_y = criterion(y, batch_M)
    loss_texture_AC = criterion(*texture_AC)
    loss_texture_BM = criterion(*texture_BM)
    loss_motion_BC = criterion(*motion_BC)
    return loss_y, loss_texture_AC, loss_texture_BM, loss_motion_BC