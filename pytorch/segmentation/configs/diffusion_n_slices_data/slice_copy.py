import os
import json
import copy



config_names = [name for name in os.listdir("./") if name.endswith(".json") and name.startswith("config_")]

slices = [10, 15, 20, 30, 42]

for config_name in config_names:
    with open(config_name) as config_buffer:
        print(f"open config {config_name}")
        config = json.loads(config_buffer.read())

    
    for n_slices in slices:
        new_config = copy.deepcopy(config)
        for data_info in new_config["data_info"]:
            data_info["dir_img_path"] = data_info["dir_img_path"].replace("5_slices", f"{n_slices}_slices")
            data_info["dir_mask_path_without_name"] =  data_info["dir_mask_path_without_name"].replace("5_slices", f"{n_slices}_slices")

            data_info["dir_img_path"] = data_info["dir_img_path"].replace("5 slices", f"{n_slices} slices")
            data_info["dir_mask_path_without_name"] =  data_info["dir_mask_path_without_name"].replace("5 slices", f"{n_slices} slices")
            
            #   new_config["data_info"][-1] = data_info
        
        new_name = config_name.replace("5_slices", f"{n_slices}_slices")
        
        with open(new_name, 'w') as f:
            json.dump(new_config, f, indent=4)