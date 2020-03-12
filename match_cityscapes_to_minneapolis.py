import os
import glob
from PIL import Image

help_msg = """
Processing Script to match Cityscapes data to the minneapolis data.

Example usage:

python match_cityscapes_to_minneapolis.py --cityscapes_dir <left training dir Images> --minneapolis_dir <minneapolis dataset training dir> --output_dir ./datasets/cityscapes/
"""

def match_cityscapes(msp_dir, cs_dir, output_dir, phase):
    save_phase = 'test' if phase == 'val' else 'train'
    savedir = os.path.join(output_dir, save_phase)
    os.makedirs(savedir + '_CITYSCAPES', exist_ok=True)
    os.makedirs(savedir + '_CITYSCAPES_LABELS', exist_ok=True)
    os.makedirs(savedir + '_MSP', exist_ok=True)
    print("Directory structure prepared at %s" % output_dir)
    
    msp_map_expr = msp_dir + "/*.jpeg"
    msp_map_paths = glob.glob(msp_map_expr)
    msp_map_paths = sorted(msp_map_paths)
    cs_expr = cs_dir + "/*/*_leftImg8bit.png"
    print(cs_expr)
    cs_paths = glob.glob(cs_expr)
    cs_paths = sorted(cs_paths)
    print(cs_paths)

    for i, (msp_map_path) in enumerate(zip(msp_map_paths)):
        for j, (cs_path) in enumerate(zip(cs_paths)):
            print(msp_map_path)
            print(cs_path)

       # check_matching_pair(segmap_path, photo_path)
       # segmap = load_resized_img(segmap_path)
        #photo = load_resized_img(photo_path)

        #data for pix2pix where the two images are placed side-by-side
        ##sidebyside = Image.new('RGB', (512, 256))
        #sidebyside.paste(segmap, (256, 0))
        #sidebyside.paste(photo, (0, 0))
        #savepath = os.path.join(savedir, "%d.jpg" % i)
        #sidebyside.save(savepath, format='JPEG', subsampling=0, quality=100)

        #data for cyclegan where the two images are stored at two distinct directories
        #savepath = os.path.join(savedir + 'A', "%d_A.jpg" % i)
        #photo.save(savepath, format='JPEG', subsampling=0, quality=100)
        #savepath = os.path.join(savedir + 'B', "%d_B.jpg" % i)
        #segmap.save(savepath, format='JPEG', subsampling=0, quality=100)
        
        #if i % (len(segmap_paths) // 10) == 0:
        #    print("%d / %d: last image saved at %s, " % (i, len(segmap_paths), savepath))


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

    

