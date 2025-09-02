import os
import cv2
import math

# example of calculating the frechet inception distance
import numpy as np
from matplotlib import pyplot as plt

def calculate_mean(arr):
    summ = arr.sum()
    mean = 0
    for i, count in enumerate(arr):
        mean += i*count
    mean/=summ
 
    cov = 0
    for i, count in enumerate(arr):
        cov+= ((i-mean)**2)*count
    cov/=summ

    return mean, np.sqrt(cov)

def calcFIDbyHist(etal_hist, gen_hist):
    m1, s1 = calculate_mean(etal_hist)
    m2, s2 = calculate_mean(gen_hist)

    print(m1, m2, s1, s2)
    #cv2.imshow("etalon", etalon)
    #cv2.imshow("slice", slice)
    #cv2.waitKey()

    score = (m1-m2)**2 + (s1-s2)**2
    return score

def calcPirsen(etal_hist, gen_hist):
    assert len(etal_hist) == len(gen_hist)
    
    normal_etal_hist = etal_hist.astype(np.float64)/etal_hist.sum()
    normal_gen_hist = gen_hist.astype(np.float64)/gen_hist.sum()

    m1 = normal_etal_hist.mean()
    m2 = normal_gen_hist.mean()

    s1 = np.sqrt(sum([(x-m1)**2 for x in normal_etal_hist])/256)
    s2 = np.sqrt(sum([(x-m2)**2 for x in normal_gen_hist]) /256)

    covr = sum([(normal_etal_hist[i]-m1)*(normal_gen_hist[i]-m2) for i in range(len(normal_etal_hist))]) / len(normal_etal_hist)
    score = covr/(s1*s2)

    return score

def calcCMMD(etal_hist, gen_hist, g=10):

    normal_etal_hist = etal_hist.astype(np.float64)/etal_hist.sum()
    normal_gen_hist = gen_hist.astype(np.float64)/gen_hist.sum()

    two_std = (2*(g**2))
    k_xx = np.mean([np.mean([math.exp(-(np.int64(normal_etal_hist[i])-normal_etal_hist[j])**2/two_std) for i in range(len(normal_etal_hist)) if i != j]) for j in range(len(normal_etal_hist))])
    k_yy = np.mean([np.mean([math.exp(-(np.int64(normal_gen_hist[i]) -normal_gen_hist[j])**2 /two_std) for i in range(len(normal_gen_hist))  if i != j]) for j in range(len(normal_gen_hist))]) 

    k_xy = np.mean([np.mean([math.exp(-(np.int64(normal_etal_hist[i])-normal_gen_hist[j])**2 /two_std) for i in range(len(normal_etal_hist))]) for j in range(len(normal_gen_hist))])

    score = k_xx + k_yy - 2*k_xy
    return score * 10**9

def calcAllHist(imgs_path, img_names, mask_use):
    calc_hist = np.zeros(256, np.uint64)

    for img_name in img_names:
        img_path = os.path.join(imgs_path, 'original', img_name)
        img = cv2.imread(img_path, 0)
        img = np.expand_dims(img, -1)
        
        if mask_use is None:
            mask = None
        else:
            if not os.path.isdir(os.path.join(imgs_path, mask_use)):
                if mask_use == "mitochondrial_boundaries":
                    mask_use = "mitochondrial boundaries"
                else:
                    raise Exception(f"Don't find {mask_use}\nin {os.path.join(imgs_path, mask_use)}")

            path_mask = os.path.join(imgs_path, mask_use, img_name)
            mask = cv2.imread(path_mask, 0)
            mask[mask>127]=255
            mask[mask<128]=0   

        hist = (np.array(cv2.calcHist([img], [0], mask, [256], [0,256]))).squeeze().squeeze().astype(np.uint64)
        calc_hist += hist

    return(calc_hist)

def viewHistPlot(etal_hist, gen_hist, name_plot, view = False, save_dir = "FID_HIST 6_classes"):
      
    plt.rcParams['figure.figsize'] = (20, 8)
    plt.rcParams.update({'font.size': 30})
    #plt.subplots_adjust(left=0.16, bottom=0.19, top=0.82)
    linewidth = 5
    
    plt.plot(gen_hist /  np.linalg.norm(gen_hist), color = 'r', label="Synthetic data", linewidth=linewidth)
    plt.plot(etal_hist / np.linalg.norm(etal_hist), color = 'g', label="Real data", linewidth=linewidth)
    plt.title(f'Histograms of {name_plot}')
    plt.legend(loc='upper right')#, prop={'size': 20})

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if not view:
        plt.savefig(f"{save_dir}/{name_plot}.png", dpi=300)
    #plt.show()
    plt.close()

etal_dir = "D:/Data/Unet_multiclass/data/original data"
#gen_dir = "C:/Users/Sokol-PC/UnetClass/pytorch/diffusion/result_t_dataset_1_classes_01_mask"
#gen_dir = "C:/Users/Sokol-PC/UnetClass/pytorch/diffusion/result_t_dataset_1_classes_100_slices_01_mask"
#gen_dir = "C:/Users/Sokol-PC/UnetClass/pytorch/diffusion/result_t_dataset_5_classes_01_mask"


mask_names = [
              None,
              "mitochondria",
              "axon",
              "boundaries",
              "vesicles",
              "PSD",
              "mitochondrial_boundaries"
              ]
map_plot = {None: "All dataset",
            "axon": "Axon",
            "boundaries": "Membrane",
            "mitochondria": "Mitochondrion",
            "vesicles": "Vesicles",
            "PSD": "PSD",            
            "mitochondrial_boundaries": "Mitochondrial boundaries"}

all_FID_stat = []
all_PLCC_stat = []
all_CMMD_stat = []

def add_name_score(name, score, max_len):
    len_name  = len(name)
    len_score = len(score)

    add_name  = f"{name}"
    add_score = f"{score}"

    if len_name < max_len:
        add_name  += f"{' '*(max_len - len_name)}"
    add_name += " | "
    if len_score < max_len:
        add_score += f"{' '*(max_len - len_score)}"
    add_score += " | "
        
    return add_name, add_score


list_max_len_cols_fid = [len(map_plot[key]) for key in mask_names]
list_max_len_cols_plcc = [len(map_plot[key]) for key in mask_names]
list_max_len_cols_cmmd = [len(map_plot[key]) for key in mask_names]


for slices in reversed([165, 100, 42, 30, 20, 15, 10, 5]):
    for num_classes in [1, 5, 6]:
        if slices > 42 and num_classes != 1:
            continue
    
        gen_dir=f"D:/Projects/UnetClass/pytorch/diffusion/result_t_dataset_{slices}_slices_{num_classes}_classes"
   
        #gen_dir=f"D:/Projects/UnetClass/pytorch/diffusion/synt_and_dif_result_t_dataset_42_slices_6_classes"
        
        work_mask_names = mask_names[:num_classes+1]

        save_dir = f"FID_HIST {slices}_slices {num_classes}_classes"

        etal_names = [name for name in os.listdir(os.path.join(etal_dir, 'original')) if name.endswith(".png")]
        gen_names =  [name for name in os.listdir(os.path.join(gen_dir, 'original'))  if name.endswith(".png")]

        print(f"number etalons: {len(etal_names)}")
        print(f"number gen: {len(gen_names)}")

        assert(len(etal_names) > 0)
        assert(len(gen_names) > 0)

        Score_list = []

        for mask_use in work_mask_names:
            print(f"calculate etal {map_plot[mask_use]}")

            print("\n\tcalculate etal")
            etal_hist = calcAllHist(etal_dir, etal_names, mask_use)
            print(etal_hist)

            print("\n\tcalculate gen")
            gen_hist = calcAllHist(gen_dir, gen_names, mask_use)
            print(gen_hist)

            scoreFID = calcFIDbyHist(etal_hist, gen_hist)
            print(f"\n\tcalculate FID: {scoreFID}\n")
            scorePLCC = calcPirsen(etal_hist, gen_hist)
            #scorePLCC = pearsonr(etal_hist, gen_hist)[0]

            print(f"\n\tcalculate PLCC: {scorePLCC}\n")
            scoreCMMD = calcCMMD(etal_hist, gen_hist)
            print(f"\n\tcalculate PLCC: {scoreCMMD}\n")
            
            Score_list.append((scoreFID, scorePLCC, scoreCMMD))

            viewHistPlot(etal_hist, gen_hist, map_plot[mask_use], save_dir=save_dir)

        for i, (scoreFID, scorePLCC, scoreCMMD) in enumerate(Score_list):
            str_fid_now = f"{scoreFID:.4f}" if not math.isnan(scoreFID) else "nan"
            str_plcc_now = f"{scorePLCC:.4f}" if not math.isnan(scorePLCC) else "nan"
            str_cmmd_now = f"{scoreCMMD:.4f}" if not math.isnan(scoreCMMD) else "nan"

            if list_max_len_cols_fid[i] < len(str_fid_now):
                list_max_len_cols_fid[i] = len(str_fid_now)
            if list_max_len_cols_plcc[i] < len(str_plcc_now):
                list_max_len_cols_plcc[i] = len(str_plcc_now)
            if list_max_len_cols_cmmd[i] < len(str_cmmd_now):
                list_max_len_cols_cmmd[i] = len(str_cmmd_now)

        str_name_fid  = "| "
        str_name_plcc = "| "
        str_name_cmmd = "| "

        str_fid  = "| "
        str_plcc = "| "
        str_cmmd = "| "

        for i in range(len(Score_list)):
            print_name = map_plot[work_mask_names[i]]
            str_fid_now = f"{Score_list[i][0]:.4f}"
            str_plcc_now = f"{Score_list[i][1]:.4f}"
            str_cmmd_now = f"{Score_list[i][2]:.4f}"

            add_name1, add_score1 = add_name_score(print_name, str_fid_now, max_len = list_max_len_cols_fid[i])
            str_name_fid+=add_name1
            str_fid += add_score1
            add_name2, add_score2 = add_name_score(print_name, str_plcc_now, max_len = list_max_len_cols_plcc[i])
            str_name_plcc+=add_name2
            str_plcc += add_score2
            add_name3, add_score3 = add_name_score(print_name, str_cmmd_now, max_len = list_max_len_cols_cmmd[i])
            str_name_cmmd+=add_name3
            str_cmmd += add_score3
                
        print("___________________________________________")
        print(str_name_fid) 
        print(str_fid)
        
        print(str_name_plcc) 
        print(str_plcc)
        
        print(str_name_cmmd) 
        print(str_cmmd)
        
        all_FID_stat.append([f"{slices}_slices {num_classes}_classes", str_fid])
        all_PLCC_stat.append([f"{slices}_slices {num_classes}_classes", str_plcc])
        all_CMMD_stat.append([f"{slices}_slices {num_classes}_classes", str_cmmd])

        with open(os.path.join(save_dir, "FID.txt"), 'w') as f:
            f.write(str_name_fid)
            f.write("\n")
            f.write(str_fid)

        with open(os.path.join(save_dir, "PLCC.txt"), 'w') as f:
            f.write(str_name_plcc)
            f.write("\n")
            f.write(str_plcc)

        with open(os.path.join(save_dir, "CMMD.txt"), 'w') as f:
            f.write(str_name_cmmd)
            f.write("\n")
            f.write(str_cmmd)


all_str_name_fid  = "| "
all_str_name_plcc = "| "
all_str_name_cmmd = "| "

for i, key in enumerate(mask_names):
    name = map_plot[key]
    all_str_name_fid  += f"{name}{' '*(list_max_len_cols_fid[i]  - len(name))} | "
    all_str_name_plcc += f"{name}{' '*(list_max_len_cols_plcc[i] - len(name))} | "
    all_str_name_cmmd += f"{name}{' '*(list_max_len_cols_cmmd[i] - len(name))} | "

len_name_max = len("100_slices 1_classes")

with open("ALL FID.txt", 'w') as f:
    f.write(f"name {" "*(len_name_max-6)}|{all_str_name_fid}")
    f.write("\n")
    for name, stat in all_FID_stat:
        f.write(f"{name}{" "*(len_name_max-len(name))}")
        f.write(f"{stat} ")
        f.write("\n")

with open("ALL PLCC.txt", 'w') as f:
    f.write(f"name {" "*(len_name_max-6)}|{all_str_name_plcc}")
    f.write("\n")
    for name, stat in all_PLCC_stat:
        f.write(f"{name}{" "*(len_name_max-len(name))}")
        f.write(stat)
        f.write("\n")

with open("ALL CMMD.txt", 'w') as f:
    f.write(f"name {" "*(len_name_max-6)}|{all_str_name_cmmd}")
    f.write("\n")
    for name, stat in all_CMMD_stat:
        f.write(f"{name}{" "*(len_name_max-len(name))}")
        f.write(stat)
        f.write("\n")


