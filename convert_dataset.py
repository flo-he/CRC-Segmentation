import numpy as np
import SimpleITK as sitk
import multiprocessing as mp
import os
import glob

def png_to_npy(files, output_dir):
    '''
    Args: \n
    :files: list of filenames to apply the conversion on. \n
    :output_dir: output directory
    '''
    
    reader = sitk.ImageFileReader()
    reader.SetImageIO("PNGImageIO")

    for file in files:
        # read in file
        reader.SetFileName(file)
        image = reader.Execute()

        # get raw numpy array
        raw_arr = sitk.GetArrayFromImage(image)

        # extract filename
        new_filename = file.split('\\')[-1].replace(".png", ".npy")

        # save as npy
        np.save(os.path.join(output_dir, new_filename), raw_arr)

if __name__ == "__main__":

    # input frames
    frames = glob.glob("C:\\AML_seg_proj\\data\\frames\\*.png")
    masks = glob.glob("C:\\AML_seg_proj\\data\\masks\\*.png")

    # output directory
    output_frames = os.path.join(os.getcwd(), "data\\frames")
    output_masks = os.path.join(os.getcwd(), "data\\masks")

    print(f"Write images .npy files to {output_frames}")
    print(f"Write masks .npy files to {output_masks}")

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
    frame_splits = np.array_split(frames, num_cores)
    mask_splits = np.array_split(masks, num_cores)

    # execute the conversion
    # frames
    processes = []
    for split in frame_splits:
        p = mp.Process(target=png_to_npy, args=(split, output_frames))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("Finished converting Images.")

    # masks
    processes = []
    for split in mask_splits:
        p = mp.Process(target=png_to_npy, args=(split, output_masks))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    
    print("Finished converting Masks.")