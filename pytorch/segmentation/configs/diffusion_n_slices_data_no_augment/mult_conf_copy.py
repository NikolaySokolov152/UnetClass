import os
import json

config_names = [name for name in os.listdir("./") if name.endswith(".json") and name.startswith("multi_")]

slices = [10, 15, 20, 30, 42]

for config_name in config_names:
    with open(config_name) as config_buffer:
        print(f"open config {config_name}")
        config = config_buffer.read()

    for n_slices in slices:
        new_config = config.replace("5_slices", f"{n_slices}_slices")
        new_name = config_name.replace("5_slices", f"{n_slices}_slices")
        
        with open(new_name, 'w') as f:
            f.write(new_config)