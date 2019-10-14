import torch
import numpy as np
from torch.autograd import grad
import matplotlib.pyplot as plt

from U_Net import UNet

model_chkpt = "C:\\AML_seg_proj\\CRC-Segmentation\\model_large_drop_batch_dice_ce\\model_chkpt_60.pt"
# define input 
inp = torch.zeros(size=(1, 3, 32, 32), requires_grad=True)


model = UNet((32, 32), (32, 32), 32, 64, 128, 256, 512, droprate=0.5, Norm=torch.nn.BatchNorm2d)
model.load_state_dict(torch.load(model_chkpt, map_location='cpu'))
model.eval()

grad_tensor = grad(model(inp)[0, 0, 15, 15], inp)

# plot sum of grad tensor
np_img = np.abs(grad_tensor[0][0][0].numpy())
np_img = np_img / np.max(np_img)
coords = np.column_stack(np.where(np_img != 0))
rf_size = np.max(coords, axis=0) - np.min(coords, axis=0)
print(f"Receptive field: {rf_size[0]}x{rf_size[1]}")

plt.figure(figsize=(15, 15))
plt.imshow(np_img, cmap="gray")
plt.colorbar()
plt.savefig("eff_receptive_field_dice_ce.png")
plt.show()