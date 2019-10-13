import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
import transforms
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from U_Net import UNet
from utils import Dice_Score, Pixel_Accuracy

output_dir = os.path.join(os.getcwd(), "input_output_images\\model_large_drop_batch_wce")

try:
    os.makedirs(output_dir)
except FileExistsError:
    pass

def w_PNG(identifier, np_img):
    imageio.imwrite(os.path.join(output_dir, identifier), np_img)

def plot_triple(img, mask, ground_truth, identifier):
    plt.figure(figsize=(15, 8))
    plt.subplot(131)
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(132)
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    plt.subplot(133)
    plt.imshow(ground_truth, cmap="gray")
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, identifier))
    plt.show()

def main():
    # frames to infer
    files = [
        "C:\\AML_seg_proj\\CRC-Segmentation\\data\\test\\frames\\frame#1058.npz",
        "C:\\AML_seg_proj\\CRC-Segmentation\\data\\test\\frames\\frame#139.npz",
        "C:\\AML_seg_proj\\CRC-Segmentation\\data\\test\\frames\\frame#26.npz"
    ]
    chkpt = "C:\\AML_seg_proj\\CRC-Segmentation\\model_large_drop_batch_wce\\model_chkpt_30.pt"
    # transforms to apply
    composed = Compose([transforms.MirrorPad(((6,), (6,), (0,))), transforms.ToTensor(), transforms.Normalize(means=(0.7942, 0.6693, 0.7722), stds=(0.1998, 0.3008, 0.2037))])

    # model
    #model = UNet((512, 512), (500, 500), 32, 64, 128, 256, 512, droprate=0.5, Norm=torch.nn.BatchNorm2d)
    model = UNet((512, 512), (500, 500), 32, 64, 128, 256, 512, droprate=0.5, Norm=torch.nn.BatchNorm2d)
    model.load_state_dict(torch.load(chkpt, map_location='cpu'))
    model.eval()

    # evaluate metrics on the fly
    dice_sc, px_acc = Dice_Score(), Pixel_Accuracy((500, 500))

    # make predictions and write images and masks to disk as png files
    with torch.no_grad():
        for file in files:
            # load img, mask
            img, ground_truth = np.load(file)["arr_0"], np.load(file.replace("frame", "mask"))["arr_0"]
            img_copy = img.copy()
            # transform img
            img, ground_truth = composed([img, ground_truth])
            # prediction shape (1, 3, 500, 500)
            pred = model(img.unsqueeze(0))
            dice, pp_acc = dice_sc(pred, ground_truth.unsqueeze(0)), px_acc(pred, ground_truth.unsqueeze(0))
            print(f"Dice Score: {dice}, PP-Accuracy: {pp_acc}")
            # mask shape (1, 500, 500) 
            mask = (torch.argmax(F.softmax(pred, dim=1), dim=1).squeeze(0).numpy() / 2 * 255).astype(np.uint8)
            # prep image for writing, shape (1, 3, 512, 512)
            img = (img.squeeze(0).numpy() * 255).astype(np.uint8)
            identifier = file.split("\\")[-1].replace(".npz", ".png").replace("#", "_")

            w_PNG(identifier=identifier, np_img=img_copy)
            w_PNG(identifier=identifier.replace("frame", "mask"), np_img=mask)

            plot_triple(img_copy, ground_truth, mask, identifier.replace(".png", "_triple.png"))


if __name__ == "__main__":
    main()