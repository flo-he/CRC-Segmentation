import numpy as np
import imageio
import multiprocessing as mp
import os
import glob
#import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import argparse
import shutil

def drop_empty_image(x, drop_threshold):
    '''
    Removes image and mask pair from consideration if it contains not enought information. \n
    Args: \n
        :x: mask    \n
        :drop_threshold: minimum relative amount of nonzero pixels 
    '''
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
    #plt.subplot(131)
    #plt.imshow(img_gray / 255, cmap="gray")

    # smooth the image, so that small black gaps in tissue do not raise false alarm
    # (masks do not these little gaps and treat them as tissue anyway)
    img_gray = gaussian_filter(img_gray, sigma=10)
    img_gray = img_gray / np.max(img_gray)

    # plotting just for checking by eye
    #plt.subplot(132)
    #plt.imshow(img_gray, cmap='gray')
    #plt.subplot(133)
    #plt.imshow(mask, cmap="gray")
    #plt.show()

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
    #print(n_empty, np.count_nonzero(pixel_mask))
    #print(rel_amount)  

    # image is valid if the amount of empty pixels is not too high
    if rel_amount < drop_threshold:
        return True
    else:
        return False

def convert_dataset(masks_fnames, output_dir, drop_threshold=0.):
    '''
    Args: \n
        :mask_fnames: list of mask locations. \n
        :output_masks_dir: output directory for the masks \n
        :output_frames_dir: output directory for the frames \n
        :drop_threshold: minimum relative amount of nonzero pixels 
    '''

    for file in masks_fnames:

        # get raw numpy array
        mask_arr = imageio.imread(file)

        # adjust labels to be 0, 1, 2
        mask_arr[mask_arr == 26] = 0
        mask_arr[mask_arr == 51] = 1
        mask_arr[mask_arr == 77] = 2

        # if the mask is almost empty (i.e. the image is black) drop the image from the dataset
        if drop_threshold > 0.:
            if drop_empty_image(mask_arr, drop_threshold):
                continue
            else:
                # read in corresponding image
                file_frame = file.replace("mask", "frame")
                img_arr = imageio.imread(file_frame)

                # check for validity
                valid = check_for_valid_seg_map(img_arr, mask_arr)
                # print for info
                #print(valid)
                if not valid:
                    continue
                else:
                    # extract filename
                    new_filename = file.split('\\')[-1].replace(".png", ".npz")

                    # save as npz
                    np.savez_compressed(os.path.join(output_dir, new_filename), mask_arr)
            
                    # convert frame 
                    new_filename = file_frame.split('\\')[-1].replace(".png", ".npz")
                    np.savez_compressed(os.path.join(output_dir, new_filename), img_arr)

def get_train_test_split(files, test_proportion=0.20, shuffle=True):
    '''
    Args:   \n
        :masks: array of mask files  \n
        :test_proportion: percentage of instances used for testing  \n
        :shuffle: shuffle dataset before splitting
    '''

    n_instances = len(files)

    if shuffle:
        files = np.random.permutation(files)

    n_test = int(n_instances*test_proportion)

    test_files = files[:n_test].copy()
    train_files = files[n_test:].copy()

    return train_files, test_files

def run_multithreaded_conversion(files, output_dir, n_jobs, drop_threshold=0.01):
    '''
    Apply convert_dataset via multiprocessing.  \n
    
    Args: \n
        :files: files to prepare    \n
        :output_dir: directory to which the .npz files are written  \n
        :n_jobs: number of threads to spawn \n
        :drop_threshold: see doc of convert_dataset
    '''

    # split for multiprocessing
    mask_splits = np.array_split(files, n_jobs)

    # execute the conversion
    processes = []
    for split in mask_splits:
        p = mp.Process(target=convert_dataset, args=(split, output_dir, drop_threshold))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

def move_files(files, to_dir):
    '''
    Args:
        :files: files to move
        :to_dir: directory to move the files to
    '''

    for file in files:
        shutil.move(file, os.path.join(to_dir, file.split("\\")[-1]))

def run_multithreaded_move(files, to_dir, n_jobs):
    '''
    Args:   \n
        :files: files to move   \n
        :to_dir: directory to move the files to \n
        :n_jobs: number of threads to spawn
    '''
    # split for mp
    splits = np.array_split(files, n_jobs)

    # run mp
    processes = []
    for split in splits:
        p = mp.Process(target=move_files, args=(split, to_dir))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

# Parsing
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--root_dir', type=str, metavar='', required=True, help='Directory to the CRC dataset.')
parser.add_argument('-t', '--threads', type=int, metavar='', help='Number of threads to spawn. Default: all available Cores.')
parser.add_argument('-s', '--test_split', type=float, metavar='', help='rel. amount of test instances to split from all data.')
args = parser.parse_args()

if __name__ == "__main__":
    # seed for split between training and test set
    np.random.seed(42)

    # get current working directory
    cwd = os.getcwd()

    # input masks
    masks = np.asarray(glob.glob(os.path.join(args.root_dir, "masks\\*.png")))

    # for checking random check valid func
    #masks = np.asarray(masks)[np.random.randint(low=0, high=len(masks), size=30)]

    # output directories
    output_dir = os.path.join(cwd, "data\\temp")

    output_tr_frames = os.path.join(cwd, "data\\train\\frames")
    output_tr_masks = os.path.join(cwd, "data\\train\\masks")

    output_ts_frames = os.path.join(os.getcwd(), "data\\test\\frames")
    output_ts_masks = os.path.join(os.getcwd(), "data\\test\\masks")

    print(f"Write train images .npz files to {output_tr_frames}")
    print(f"Write train masks .npz files to {output_tr_masks}")
    print(f"Write test images .npz files to {output_ts_frames}")
    print(f"Write test masks .npz files to {output_ts_masks}")
    # create directory if not present
    try: 
        print("Trying to create data directories:")
        os.makedirs(output_dir)
        os.makedirs(output_tr_frames)
        os.makedirs(output_tr_masks)
        os.makedirs(output_ts_frames)
        os.makedirs(output_ts_masks)
        print("Finished.")
    except FileExistsError:
        print("Some of the directories already exist.")
    
    # gather number of processes to spawn (# of available cpu cores)
    if args.threads:
        num_cores = args.threads
    else:
        num_cores = mp.cpu_count()
    print(f"Number of available CPU Cores: {num_cores}")

    # convert dataset
    try:
        run_multithreaded_conversion(masks, output_dir, num_cores)
        print("Finished conversion of data.")
    except:
        raise RuntimeError("Conversion of data failed.")
    
    #############################

    chosen_masks = glob.glob(os.path.join(output_dir, "mask*.npz"))

    # get split
    if args.test_split:
        train_masks, test_masks = get_train_test_split(chosen_masks, args.test_split)
        train_images, test_images = [name.replace("mask", "frame") for name in train_masks], [name.replace("mask", "frame") for name in test_masks]
    else:
        train_masks, test_masks = get_train_test_split(chosen_masks)
        train_images, test_images = [name.replace("mask", "frame") for name in train_masks], [name.replace("mask", "frame") for name in test_masks]

    # split into train and test set
    try:
        run_multithreaded_move(train_masks, output_tr_masks, num_cores)
        run_multithreaded_move(train_images, output_tr_frames, num_cores)
        run_multithreaded_move(test_masks, output_ts_masks, num_cores)
        run_multithreaded_move(test_images, output_ts_frames, num_cores)
        print("Finished splitting data.")
    except:
        raise RuntimeError("Splitting data failed.")
    
    try:
        os.rmdir(output_dir)
        print("Removed temporary dirs.")
    except:
        raise RuntimeError("Failed to remove temporary dirs.")