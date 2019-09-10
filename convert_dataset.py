import numpy as np
import SimpleITK as sitk
import multiprocessing as mp
import os
import glob
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import argparse

def drop_image(x, drop_threshold):
    # count nonzero label
    nonzero_pixel = np.count_nonzero(x)
    rel_amount = nonzero_pixel / 250000    # images are 500x500

    # maybe drop image
    if rel_amount < drop_threshold:
        return True
    else:
        return False

def check_for_valid_seg_map(image, mask, drop_threshold=0.15):
    '''
    Drop the mask from the dataset, if it does not overlap with the corresponding image sufficiently.

    Args:
        :image: RGB image 
        :mask: corresponding segmentation mask
        :drop_threshold: upper limit percentage of pixels that do not overlap 
    '''

    # convert RGB to grayscale and invert so that background pixels are black in the image
    img_gray = np.mean(image, axis=2)
    img_gray = np.abs(img_gray - 255.)

    # plotting just for checking by eye
    plt.subplot(131)
    plt.imshow(img_gray / 255, cmap="gray")

    # smooth the image, so that small black gaps in tissue do not raise false alarm
    # (masks do not these little gaps and treat them as tissue anyway)
    img_gray = gaussian_filter(img_gray, sigma=10)
    img_gray = img_gray / np.max(img_gray)

    # plotting just for checking by eye
    plt.subplot(132)
    plt.imshow(img_gray, cmap='gray')
    plt.subplot(133)
    plt.imshow(mask, cmap="gray")
    plt.show()

    # which pixel are non background according to the mask?
    pixel_mask = np.logical_or(mask == 1, mask == 2)

    # corresponding pixels in the image
    tissue_pixels = img_gray[pixel_mask]

    # which of them are empty/black? Value rather high (and arbitrary, we just found it works fine) due to the smoothing the image gets a bit brighter due to already
    # imperfections in the background
    empty_pixels = tissue_pixels[tissue_pixels < 40./255.]

    # number of empty pixels in image, which are labeled as tissue in the mask
    n_empty = len(empty_pixels) 

    # relative amount of pixels where the mask overlaps with the image
    rel_amount = n_empty / np.count_nonzero(pixel_mask)
    # for checking
    print(n_empty, np.count_nonzero(pixel_mask))
    print(rel_amount)  

    # image is valid if the amount of empty pixels is not too high
    if rel_amount < drop_threshold:
        return True
    else:
        return False

def convert_dataset(masks_fnames, output_masks_dir, output_frames_dir, drop_threshold=0.):
    '''
    Args: \n
        :mask_fnames: list of mask locations. \n
        :output_masks_dir: output directory for the masks \n
        :output_frames_dir: output directory for the frames \n
        :drop_threshold: minimum relative amount of nonzero pixels 
    '''
    
    reader = sitk.ImageFileReader()
    reader.SetImageIO("PNGImageIO")

    for file in masks_fnames:
        #print(file)
        # read in mask
        reader.SetFileName(file)
        mask = reader.Execute()

        # get raw numpy array
        mask_arr = sitk.GetArrayFromImage(mask)

        # adjust labels to be 0, 1, 2
        mask_arr[mask_arr == 26] = 0
        mask_arr[mask_arr == 51] = 1
        mask_arr[mask_arr == 77] = 2

        # if the mask is almost empty (i.e. the image is black) drop the image from the dataset
        if drop_threshold > 0.:
            if drop_image(mask_arr, drop_threshold):
                continue
            else:
                # read in corresponding image
                file_frame = file.replace("mask", "frame")
                reader.SetFileName(file_frame)
                image = reader.Execute()
                img_arr = sitk.GetArrayFromImage(image)

                # check for validity
                valid = check_for_valid_seg_map(img_arr, mask_arr)
                # print for info
                print(valid)
                if not valid:
                    continue
                else:
                    # extract filename
                    new_filename = file.split('\\')[-1].replace(".png", ".npz")

                    # save as npz
                    np.savez_compressed(os.path.join(output_masks_dir, new_filename), mask_arr)
            
                    # convert frame 
                    new_filename = file_frame.split('\\')[-1].replace(".png", ".npz")
                    np.savez_compressed(os.path.join(output_frames_dir, new_filename), img_arr)


# Parsing
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--root_dir', type=str, metavar='', required=True, help='Directory to the CRC dataset.')
parser.add_argument('-t', '--threads', type=int, metavar='', help='Number of threads to spawn. Default: all available Cores.')
args = parser.parse_args()

if __name__ == "__main__":
    
    # input frames
    masks = glob.glob(os.path.join(args.root_dir, "masks\\*.png"))
    # for checking random check valid func
    masks = np.asarray(masks)[np.random.randint(low=0, high=len(masks), size=30)]

    # output directories
    output_frames = os.path.join(os.getcwd(), "data\\frames")
    output_masks = os.path.join(os.getcwd(), "data\\masks")

    print(f"Write images .npz files to {output_frames}")
    print(f"Write masks .npz files to {output_masks}")

    # create directory if not present
    try: 
        print("Trying to create data directories:")
        os.makedirs(output_frames)
        os.makedirs(output_masks)
        print("Finished.")
    except FileExistsError:
        print("Some of the directories already exist.")
    
    # gather number of processes to spawn (# of available cpu cores)
    if args.threads:
        num_cores = args.threads
    else:
        num_cores = mp.cpu_count()
    print(f"Number of available CPU Cores: {num_cores}")

    # split for multiprocessing
    mask_splits = np.array_split(masks, num_cores)

    # execute the conversion
    processes = []
    for split in mask_splits:
        p = mp.Process(target=convert_dataset, args=(split, output_masks, output_frames, 0.01))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    
    print("Finished converting the Dataset.")