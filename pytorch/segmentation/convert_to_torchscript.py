import torch
from model_definition import Tiny_unet_v3

def convert_to_torchscript():
    # Define and load the model
    model = Tiny_unet_v3(n_channels=1, n_classes=6)
    
    # Load the state dictionary from the .pth file
    state_dict = torch.load("test_data/model_data/model_by_config_diffusion_data_42_slices_6_classes_dataset_mix_6_classes_seed_1466947709_tiny_unet_v3.pth", 
                            map_location=torch.device('cpu'))
    
    # Load the weights into the model
    model.load_state_dict(state_dict)
    
    # Set the model to evaluation mode
    model.eval()
    
    print("Model loaded successfully!")
    
    # Create a dummy input tensor for tracing
    dummy_input = torch.randn(1, 1, 256, 256)  # batch_size=1, channels=1, height=256, width=256
    
    # Convert the model to TorchScript using tracing
    print("Converting model to TorchScript...")
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Save the traced model as .pt file
    output_path = "test_data/model_data/traced_model.pt"
    traced_model.save(output_path)
    
    print(f"TorchScript model saved to: {output_path}")
    
    # Verify the saved model by loading it back
    loaded_model = torch.jit.load(output_path)
    loaded_model.eval()
    
    # Test the loaded model
    with torch.no_grad():
        original_output = model(dummy_input)
        traced_output = loaded_model(dummy_input)
        
        print(f"Original model output shape: {original_output.shape}")
        print(f"Traced model output shape: {traced_output.shape}")
        
        # Check if outputs are close
        is_close = torch.allclose(original_output, traced_output, atol=1e-5)
        print(f"Outputs are close: {is_close}")
    
    return traced_model

if __name__ == "__main__":
    traced_model = convert_to_torchscript()
    print("Model successfully converted to TorchScript (.pt) format!")