import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
import transforms
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from U_Net import UNet

output_dir = os.path.join(os.getcwd(), "input_output_images\\model_large_drop_batch_dice")

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
    chkpt = "C:\\AML_seg_proj\\CRC-Segmentation\\model_large_drop_batch_dice\\model_chkpt_175.pt"
    # transforms to apply
    composed = Compose([transforms.MirrorPad(((6,), (6,), (0,))), transforms.ToTensor(), transforms.Normalize(means=(0.7942, 0.6693, 0.7722), stds=(0.1998, 0.3008, 0.2037))])

    # model
    #model = UNet((512, 512), (500, 500), 32, 64, 128, 256, 512, droprate=0.5, Norm=torch.nn.BatchNorm2d)
    model = UNet((512, 512), (500, 500), 32, 64, 128, 256, 512, droprate=0.5, Norm=torch.nn.BatchNorm2d)
    model.load_state_dict(torch.load(chkpt))
    model.eval()

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