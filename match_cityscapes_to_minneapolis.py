import os
import glob
from PIL import Image
from PIL import ImageFilter
from shutil import copyfile
import numpy as np
import cv2

help_msg = """
Processing Script to match Cityscapes data to the minneapolis data.

Example usage:

python match_cityscapes_to_minneapolis.py --cityscapes_dir <left training dir Images> --minneapolis_dir <minneapolis dataset training dir> --output_dir ./datasets/cityscapes/
"""

def filter_and_correlation(path1, path2):
    img1 = Image.open(path1).convert('L').resize((256, 256))
    img2 = Image.open(path2).convert('L').resize((256, 256))
    img1 = cv2.Laplacian(np.array(img1), cv2.CV_8U)
    img2 = cv2.Laplacian(np.array(img2), cv2.CV_8U)
    img1 = img1.reshape((1, 256*256))[0]
    img2 = img2.reshape((1, 256*256))[0]
    a = (img1 - np.mean(img1))
    b = (img2 - np.mean(img2))
    corr = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    print(corr)
    return corr


def match_cityscapes(msp_dir, cs_dir, output_dir, phase):
    save_phase = 'test' if phase == 'val' else 'train'
    savedir = os.path.join(output_dir, save_phase)
    os.makedirs(savedir + '_CITYSCAPES', exist_ok=True)
    os.makedirs(savedir + '_MSP', exist_ok=True)
    print("Directory structure prepared at %s" % output_dir)
    

    msp_map_expr = msp_dir + "/*.jpeg"
    msp_map_paths = glob.glob(msp_map_expr)
    msp_map_paths = sorted(msp_map_paths)
    cs_expr = cs_dir + "/*/*_leftImg8bit.png"
    cs_paths = glob.glob(cs_expr)
    cs_paths = sorted(cs_paths)


    # NOTE: This mapping for this function will need to be changed.
    # ATM it just goes in order.  There needs to be some logic being these pairings.
    #for i, (msp_map_path, cs_path) in enumerate(zip(msp_map_paths, cs_paths)):
    count = 0
    for j, (cs_path) in enumerate(zip(cs_paths)):
        cs_save_dir = os.path.join(savedir + "_CITYSCAPES","%d.jpg" % count)
        print(j)
        for i, (msp_map_path) in enumerate(zip(msp_map_paths)):
            msp_save_dir = os.path.join(savedir + "_MSP","%d.jpg" % count)
            if (filter_and_correlation(msp_map_path[0], cs_path[0])) > 0.06:
                count = count = 1
                copyfile(msp_map_path[0], msp_save_dir)
                copyfile(cs_path[0], cs_save_dir)
                break

    if j % (len(msp_map_paths) // 10) == 0:
        print("%d / %d: last image saved at %s, " % (i, len(cs_paths), savedir))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cityscapes_dir', type=str, required=True,
                        help='Path to the Cityscapes gtCoarse directory.')
    parser.add_argument('--minneapolis_dir', type=str, required=True,
                        help='Path to the Minneapolis directory.')
    parser.add_argument('--output_dir', type=str, required=True,
                        default='./datasets/cityscapes',
                        help='Directory the output images will be written to.')
    opt = parser.parse_args()

    print(help_msg)
    
    print('Matching up the Cityscapes images to the Minneapolis for train phase of cGAN A')
    match_cityscapes(opt.minneapolis_dir, opt.cityscapes_dir, opt.output_dir, "train")

    print('Done')

    

