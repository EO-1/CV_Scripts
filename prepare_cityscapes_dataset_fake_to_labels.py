import os
import glob
from PIL import Image

help_msg = """
The dataset can be downloaded from https://cityscapes-dataset.com.
Please download the datasets [gtFine_trainvaltest.zip] and [leftImg8bit_trainvaltest.zip] and unzip them.
gtFine contains the semantics segmentations. Use --gtFine_dir to specify the path to the unzipped gtFine_trainvaltest directory. 
leftImg8bit contains the dashcam photographs. Use --leftImg8bit_dir to specify the path to the unzipped leftImg8bit_trainvaltest directory. 
The processed images will be placed at --output_dir.

Example usage:

python prepare_cityscapes_dataset.py --gitFine_dir ./gtFine/ --leftImg8bit_dir ./leftImg8bit --output_dir ./datasets/cityscapes/
"""

def load_resized_img(path):
    return Image.open(path).convert('RGB').resize((256, 256))

def check_matching_pair(msp_map_path, cs_path):
    msp_map_identifier = os.path.basename(msp_map_path).replace('_leftImg8bit_fake_B', '')
    cs_identifier = os.path.basename(cs_path).replace('_gtCoarse_color','')
    
    assert msp_map_identifier == cs_identifier, \
        "[%s] and [%s] don't seem to be matching. Aborting." % (msp_map_path, cs_path)
    

def process_cityscapes(msp_dir, cs_dir, output_dir, phase):
    save_phase = 'test' if phase == 'val' else 'train'
    savedir = os.path.join(output_dir, save_phase)
    os.makedirs(savedir, exist_ok=True)
    print("Directory structure prepared at %s" % output_dir)
    
    #segmap_expr = os.path.join(gtFine_dir, phase) + "/*/*_color.png"

    #photo_expr = os.path.join(leftImg8bit_dir, phase) + "/*/*_leftImg8bit.png"

    msp_map_expr = msp_dir + "/*_leftImg8bit_fake_B.png"
    msp_map_paths = glob.glob(msp_map_expr)
    msp_map_paths = sorted(msp_map_paths)
    print(msp_map_expr)

    cs_expr = cs_dir + "/*/*_color.png"
    cs_paths = glob.glob(cs_expr)
    cs_paths = sorted(cs_paths)
    print(cs_expr)

    assert len(msp_map_paths) == len(cs_paths), \
        "%d images that match [%s], and %d images that match [%s]. Aborting." % (len(msp_map_paths), msp_map_expr, len(cs_paths), cs_expr)

    for i, (msp_map_path, cs_path) in enumerate(zip(msp_map_paths, cs_paths)):
        check_matching_pair(msp_map_path, cs_path)
        msp_map = load_resized_img(msp_map_path)
        cs = load_resized_img(cs_path)

        # data for pix2pix where the two images are placed side-by-side
        sidebyside = Image.new('RGB', (512, 256))
        sidebyside.paste(msp_map, (256, 0))
        sidebyside.paste(cs, (0, 0))
        savepath = os.path.join(savedir, "%d.jpg" % i)
        sidebyside.save(savepath, format='JPEG', subsampling=0, quality=100)
        
        if i % (len(msp_map_paths) // 10) == 0:
            print("%d / %d: last image saved at %s, " % (i, len(msp_map_paths), savepath))


        
        
        
        
        
    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--msp_dir', type=str, required=True,
                        help='Path to the Cityscapes gtFine directory.')
    parser.add_argument('--cityscapes_dir', type=str, required=True,
                        help='Path to the Cityscapes leftImg8bit_trainvaltest directory.')
    parser.add_argument('--output_dir', type=str, required=True,
                        default='./datasets/cityscapes',
                        help='Directory the output images will be written to.')
    opt = parser.parse_args()

    print(help_msg)
    
    print('Preparing Cityscapes Dataset for train phase')
    process_cityscapes(opt.msp_dir, opt.cityscapes_dir, opt.output_dir, "train")

    print('Done')

    

