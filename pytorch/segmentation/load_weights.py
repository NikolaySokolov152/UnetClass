import torch
from model_definition import Tiny_unet_v3

def load_model_weights():
    # Define the model with correct parameters
    # Based on the file name: 6 classes, 1 channel input
    model = Tiny_unet_v3(n_channels=1, n_classes=6)
    
    # Load the state dictionary from the .pth file
    state_dict = torch.load("test_data/model_data/model_by_config_diffusion_data_42_slices_6_classes_dataset_mix_6_classes_seed_1466947709_tiny_unet_v3.pth", 
                            map_location=torch.device('cpu'))
    
    # Load the weights into the model
    model.load_state_dict(state_dict)
    
    # Set the model to evaluation mode
    model.eval()
    
    print("Model loaded successfully!")
    print(f"Model input channels: {model.n_channels}")
    print(f"Model output classes: {model.n_classes}")
    
    # Test the model with a dummy input
    dummy_input = torch.randn(1, 1, 256, 256)  # batch_size=1, channels=1, height=256, width=256
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
    
    return model

if __name__ == "__main__":
    model = load_model_weights()
    print("Model is ready for conversion to TorchScript.")