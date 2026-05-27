import torch
import imageio
import numpy as np

def visualize_video_tensor(tensor, output_path, fps=16):
    """
    Visualize and save a video tensor as mp4 file.
    
    Args:
        tensor: torch.Tensor of shape [B, C, T, H, W] with values in range [-1, 1]
        output_path: str, path to save the mp4 file
        fps: int, frames per second for the output video
    """
    # Remove batch dimension if present
    if tensor.dim() == 5:
        tensor = tensor[0]  # [C, T, H, W]
    
    # Convert from [-1, 1] to [0, 255]
    tensor = (tensor + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    tensor = torch.clamp(tensor, 0, 1)  # Ensure values are in valid range
    tensor = (tensor * 255).to(torch.uint8)  # [0, 1] -> [0, 255]
    
    # Convert to numpy and transpose to [T, H, W, C]
    # From [C, T, H, W] to [T, C, H, W] to [T, H, W, C]
    video_np = tensor.permute(1, 2, 3, 0).cpu().numpy()
    
    # Save as mp4
    imageio.mimwrite(output_path, video_np, fps=fps, codec='libx264', quality=8)
    print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    # Assuming you have a tensor like animate_pose_video
    # animate_pose_video shape: torch.Size([1, 3, 77, 480, 832])
    
    # Load or create your tensor here
    # For demonstration, create a random tensor
    # animate_pose_video = torch.randn(1, 3, 77, 480, 832) * 2 - 1  # Random values in [-1, 1]
    
    # If you already have the tensor, just pass it to the function
    # visualize_video_tensor(animate_pose_video, "output_video.mp4", fps=16)
    
    print("Usage:")
    print("from visualize_video_tensor import visualize_video_tensor")
    print("visualize_video_tensor(your_tensor, 'output.mp4', fps=16)")
