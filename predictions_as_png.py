import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import imageio
from utils import MirrorPad
import os
from U_Net import NeuralNet

output_dir = os.path.join(os.getcwd(), "input_output_images")

try:
    os.mkdir(output_dir)
except FileExistsError:
    pass

def w_PNG(identifier, np_img):
    imageio.imwrite(os.path.join(output_dir, identifier), np_img)

def plot_triple(img, mask, ground_truth):
    plt.figure(figsize=(15, 8))
    plt.subplot(131)
    plt.imshow(img.transpose((1, 2, 0)))
    plt.axis("off")
    plt.subplot(132)
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    plt.subplot(133)
    plt.imshow(ground_truth, cmap="gray")
    plt.axis("off")
    plt.show()

def main():
    # frames to infer
    files = ["C:\\AML_seg_proj\\CRC-Segmentation\\data\\test\\frames\\frame#17.npz"]
    chkpt = []
    # transforms to apply
    composed = transforms.Compose([MirrorPad(((6,), (6,), (0,))), transforms.ToTensor(), transforms.Normalize(mean=(0.7979, 0.6772, 0.7768), std=(0.1997, 0.3007, 0.2039), inplace=True)])

    # model
    model = NeuralNet(64, 128, 256, 512, 1024)
    if chkpt:
        model.load_state_dict(torch.load(chkpt))
    model.use_dropout = False

    # make predictions and write images and masks to disk as png files
    with torch.no_grad():
        for file in files:
            # load img, mask
            img, ground_truth = np.load(file)["arr_0"], np.load(file.replace("frame", "mask"))["arr_0"]
            # transform img
            img = composed(img).unsqueeze(0)
            # prediction shape (1, 3, 500, 500)
            pred = model(img)
            # mask shape (1, 500, 500) 
            mask = (torch.argmax(F.softmax(pred, dim=1), dim=1).squeeze(0).numpy() / 2 * 255).astype(np.uint8)
            # prep image for writing, shape (1, 3, 512, 512)
            img = (img.squeeze(0).numpy() * 255).astype(np.uint8)
            identifier = file.split("\\")[-1].replace(".npz", ".png")

            w_PNG(identifier=identifier, np_img=img.transpose((1, 2, 0)))
            w_PNG(identifier=identifier.replace("frame", "mask"), np_img=mask)

            plot_triple(img, mask, ground_truth)


if __name__ == "__main__":
    main()