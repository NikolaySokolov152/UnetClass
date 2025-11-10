import torch
import os

# Load the .pth file
model_path = "test_data/model_data/model_by_config_diffusion_data_42_slices_6_classes_dataset_mix_6_classes_seed_1466947709_tiny_unet_v3.pth"

print(f"Loading model from: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")

# Load the state dictionary
state_dict = torch.load(model_path, map_location=torch.device('cpu'))

print("\nType of loaded data:", type(state_dict))

if isinstance(state_dict, dict):
    print("\nKeys in state_dict:")
    for key in state_dict.keys():
        print(f"  {key}")
    
    print("\nDetailed information about each key:")
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"  {key}: type {type(value)}")
            if isinstance(value, dict):
                print(f"    Contains keys: {list(value.keys())}")
else:
    print("Loaded data is not a dictionary, it's a model directly")