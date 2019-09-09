import numpy as np
import SimpleITK as sitk
import multiprocessing as mp
import os
import glob

def drop_image(x, drop_threshold):
    # count nonzero label
    nonzero_pixel = np.count_nonzero(x)
    rel_amount = nonzero_pixel / 250000    # images are 500x500

    # maybe drop image
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
        # read in mask
        reader.SetFileName(file)
        image = reader.Execute()

        # get raw numpy array
        raw_arr = sitk.GetArrayFromImage(image)

        # adjust labels to be 0, 1, 2
        raw_arr[raw_arr == 26] = 0
        raw_arr[raw_arr == 51] = 1
        raw_arr[raw_arr == 77] = 2

        # if the mask is almost empty (i.e. the image is black) drop the image from the dataset
        if drop_threshold > 0.:
            if drop_image(raw_arr, drop_threshold):
                continue
            else:
                # extract filename
                new_filename = file.split('\\')[-1].replace(".png", ".npz")

                # save as npz
                np.savez_compressed(os.path.join(output_masks_dir, new_filename), raw_arr)
        
                # convert frame 
                file_frame = file.replace("mask", "frame")
                reader.SetFileName(file_frame)
                image = reader.Execute()
                arr = sitk.GetArrayFromImage(image)
                new_filename = file_frame.split('\\')[-1].replace(".png", ".npz")
                np.savez_compressed(os.path.join(output_frames_dir, new_filename), arr)

if __name__ == "__main__":

    # input frames
    masks = glob.glob("C:\\AML_seg_proj\\data\\masks\\*.png")

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
    num_cores = mp.cpu_count()
    print(f"Number of available CPU Cores: {num_cores}")

    # split for multiprocessing
    mask_splits = np.array_split(masks, num_cores)

    # execute the conversion
    processes = []
    for split in mask_splits:
        p = mp.Process(target=convert_dataset, args=(split, output_masks, output_frames, 0.03))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    
    print("Finished converting the Dataset.")