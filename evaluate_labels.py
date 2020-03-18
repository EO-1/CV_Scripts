import os
import glob
from PIL import Image
import numpy as np
from sklearn.neighbors import NearestNeighbors

help_msg = """
python evaluate_labels.py --input_dir <Inputs> --results_dir <dir>
"""

label_colors = [
        [0, 0, 0],
        [111, 74, 0],
        [81, 0, 81],
        [128, 64, 128],
        [244, 35, 232],
        [250, 170, 160],
        [230, 150, 140],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [180, 165, 180],
        [150, 100, 100],
        [150, 120, 90],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 0, 90],
        [0, 0, 110],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [0, 0, 142]
        ]


# NOTE, This function computes the jacard index
def calc_per_pixel_accuracy(f_path, r_path, nn):
    f_img = np.array(Image.open(f_path).convert("RGB"))
    r_img = np.array(Image.open(r_path).convert("RGB"))
    r, c, l = f_img.shape

    area_of_overlap = np.zeros(30)
    area_of_union = np.zeros(30)
    num = 0
    for i in range(0, r):
        for j in range(0, c):
            f_neigh = nn.kneighbors([f_img[i][j]])
            r_neigh = nn.kneighbors([r_img[i][j]])
            f_class = f_neigh[1][0][0]
            r_class = r_neigh[1][0][0]
            if f_class == r_class:
                num = num + 1
                area_of_union[f_class] = area_of_union[f_class] + 1
                area_of_overlap[f_class] = area_of_overlap[f_class] + 1
            else:
                area_of_union[f_class] = area_of_union[f_class] + 1
                area_of_union[r_class] = area_of_union[r_class] + 1

    results = np.zeros(30)
    for i in range(0, 30):
        results[i] = area_of_overlap[i]/(area_of_union[i] + 0.0000001)

    # Remove the void class
    results = np.delete(results, 0)
    print(results.shape)

    return np.average(results), num/(r*c)

def check_matching_pair(f_path, r_path):
    f_identifier = os.path.basename(f_path).replace('_fake_B', '')
    r_identifier = os.path.basename(r_path).replace('_real_B', '')
                    
    assert f_identifier == r_identifier, \
        "[%s] and [%s] don't seem to be matching. Aborting." % (f_path, r_path)

def process_labels(input_dir, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    results_path = results_dir + "results.txt"
    print(results_path)
    PPA_list = []
    IoU_list = []
    f = open(results_path, "w")
    print("Creating Results dir as %s" % results_dir)


    fake_expr = input_dir + "/*_fake_B.png"
    fake_paths = glob.glob(fake_expr)
    fake_paths = sorted(fake_paths)

    real_expr = input_dir + "/*_real_B.png"
    real_paths = glob.glob(real_expr)
    real_paths = sorted(real_paths)

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(label_colors)

    for i, (fake_path, real_path) in enumerate(zip(fake_paths, real_paths)):
        check_matching_pair(fake_path, real_path)
        print(fake_path, real_path)
        IoU, PPA = calc_per_pixel_accuracy(fake_path, real_path, nn)
        print(IoU, PPA)
        IoU_list.append(IoU)
        PPA_list.append(PPA)
        f.write("Image ")
        f.write(str(i))
        f.write("\n")
        f.write("PPA: ")
        f.write(str(PPA))
        f.write("\n")
        f.write("IoU: ")
        f.write(str(IoU))
        f.write("\n")

        if i % ((len(fake_paths) // 10) + 1) == 0:
            print("%d / %d: last image saved at %s, " % (i, len(fake_paths), results_path))
 
    avg_PPA = np.average(PPA_list)
    avg_IoU = np.average(IoU_list)
    print("PPA Average: " ,PPA_list)
    print("IoU Average: " ,IoU_list)

    f.write("Average PPA: ")
    f.write(str(avg_PPA))
    f.write("\n")
    f.write("Average IoU: ")
    f.write(str(avg_IoU))
    f.write("\n")

    f.close()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the the fake and real labeled data')
    parser.add_argument('--results_dir', type=str, required=True,
    #                    default='/content/evaluated_results',
                        default='./Results',
                        help='location results will be written to')
    opt = parser.parse_args()

    print(help_msg)

    print('Preparing Cityscapes Dataset for train phase')
    process_labels(opt.input_dir, opt.results_dir)

    print('Done')

    
