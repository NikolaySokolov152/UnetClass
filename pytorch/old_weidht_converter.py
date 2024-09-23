import torch
import os
import sys
sys.path.append("src/")

path_dir = "segmentation/Multiple_diffusion_42_slices_6_classes/"

def recursive_open(path, tabs=""):
    if tabs=="":
        print(f"Start work with {path}")

    if os.path.isdir(path):
        print(f"open {tabs}{os.path.basename(path)}")
        for file in os.listdir(path):
            recursive_open(os.path.join(path, file), tabs=tabs+"\t")
    else:
        print_str = f"check file {tabs}{os.path.basename(path)}"
        if path.endswith(".pt"):
            model = torch.load(path)
            torch.save(model.state_dict(), path+"h")
            print_str += f"save {os.path.basename(path)} as {os.path.basename(path)+'h'}"

        print (print_str)

recursive_open(path_dir)
