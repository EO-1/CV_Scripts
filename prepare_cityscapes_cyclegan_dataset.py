import os
import glob
from PIL import Image
import random

help_msg = """
The dataset can be downloaded from https://cityscapes-dataset.com.
Please download the datasets [gtFine_trainvaltest.zip] and [leftImg8bit_trainvaltest.zip] and unzip them.
gtFine contains the semantics segmentations. Use --gtFine_dir to specify the path to the unzipped gtFine_trainvaltest directory. 
leftImg8bit contains the dashcam photographs. Use --leftImg8bit_dir to specify the path to the unzipped leftImg8bit_trainvaltest directory. 
The processed images will be placed at --output_dir.

Example usage:

python prepare_cityscapes_dataset.py --gitFine_dir <B Directory> --leftImg8bit_dir <A Directory> --output_dir ./datasets/cityscapes/
"""

def load_resized_img(path):
    return Image.open(path).convert('RGB').resize((256, 256))

def process_cityscapes(gtFine_dir, leftImg8bit_dir, output_dir, num_cs,  phase):
    save_phase = 'test' if phase == 'val' else 'train'
    savedir = os.path.join(output_dir, save_phase)
    os.makedirs(savedir + 'A', exist_ok=True)
    os.makedirs(savedir + 'B', exist_ok=True)
    print("Directory structure prepared at %s" % output_dir)
    
    segmap_expr = os.path.join(gtFine_dir, phase) + "/*/*.png"
    segmap_paths = glob.glob(segmap_expr)
    random.shuffle(segmap_paths)

    photo_expr = os.path.join(leftImg8bit_dir, phase) + "/*.jpeg"
    photo_paths = glob.glob(photo_expr)
    photo_paths = sorted(photo_paths)

    for i, (photo_path) in enumerate(zip(photo_paths)):
        photo = load_resized_img(photo_path[0])
        savepath = os.path.join(savedir + 'A', "%d_A.jpg" % i)
        photo.save(savepath, format='JPEG', subsampling=0, quality=100)
        
        if i % (len(photo_paths) // 10) == 0:
            print("%d / %d: last image saved at %s, " % (i, len(photo_paths), savepath))

    for i, (segmap_path) in enumerate(zip(segmap_paths)):
        if i >= num_cs:
            break
        segmap = load_resized_img(segmap_path[0])
        savepath = os.path.join(savedir + 'B', "%d_B.jpg" % i)
        segmap.save(savepath, format='JPEG', subsampling=0, quality=100)
        
        if i % (len(segmap_paths) // 10) == 0:
            print("%d / %d: last image saved at %s, " % (i, len(segmap_paths), savepath))





        
        
        
        
        
    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gtFine_dir', type=str, required=True,
                        help='Path to the Cityscapes gtFine directory.')
    parser.add_argument('--leftImg8bit_dir', type=str, required=True,
                        help='Path to the Cityscapes leftImg8bit_trainvaltest directory.')
    parser.add_argument('--num_cs', type=int, required=True,
                        help='number of cityscapes images to use')
    parser.add_argument('--output_dir', type=str, required=True,
                        default='./datasets/cityscapes',
                        help='Directory the output images will be written to.')
    opt = parser.parse_args()

    print(help_msg)
    
    print('Preparing Cityscapes Dataset for val phase')
    process_cityscapes(opt.gtFine_dir, opt.leftImg8bit_dir, opt.output_dir, opt.num_cs, "val")
    print('Preparing Cityscapes Dataset for train phase')
    process_cityscapes(opt.gtFine_dir, opt.leftImg8bit_dir, opt.output_dir, opt.num_cs, "train")

    print('Done')

    

