import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 1. Load your trained policy directly from the Hugging Face Hub
model_repo_id = "AdamAxelrod/red_dot_model"
print(f"Loading model {model_repo_id} from Hub...")
policy = ACTPolicy.from_pretrained(model_repo_id)
policy.eval()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
policy.to(device)

# 2. Load the dataset directly from the Hugging Face Hub
dataset_repo_id = "AdamAxelrod/red_dot"
print(f"Loading dataset {dataset_repo_id} from Hub...")
dataset = LeRobotDataset(dataset_repo_id)

# 3. Extract a single frame (e.g., the very first frame of the dataset)
frame_index = 5000
frame = dataset[frame_index]

# Dynamically find the camera key used in your dataset
camera_key = dataset.meta.camera_keys[1]

# Extract the image for visualization.
# LeRobotDataset provides images as PyTorch tensors of shape (C, H, W) normalized to [0, 1].
# We transpose it to (H, W, C) for matplotlib.
original_image = frame[camera_key].permute(1, 2, 0).numpy()
img_h, img_w = original_image.shape[:2]

# 4. Prepare the batch dictionary for the model
# We must add a batch dimension (unsqueeze) to all tensors and move them to the GPU/CPU
batch = {
    k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v 
    for k, v in frame.items()
}

# 5. Define and Register the PyTorch Hook
attention_maps = []

def get_attention_hook(module, input, output):
    # nn.MultiheadAttention returns (attn_output, attn_weights)
    # We grab the weights at index 1 and move them to CPU
    attn_weights = output[1] 
    attention_maps.append(attn_weights.detach().cpu())

# Attach the hook to the last cross-attention layer of the ACT decoder
cross_attn_layer = policy.model.decoder.layers[-1].multihead_attn
hook_handle = cross_attn_layer.register_forward_hook(get_attention_hook)

# 6. Run the forward pass
print("Running forward pass to extract attention...")
with torch.no_grad():
    _ = policy.select_action(batch)

# Clean up the hook
hook_handle.remove()

# 7. Extract backbone feature map to check if red dot is preserved
img_tensor = batch[camera_key]  # (1, C, H, W)
with torch.no_grad():
    feature_map = policy.model.backbone(img_tensor)["feature_map"]  # (1, 512, feat_H, feat_W)

feat_avg = feature_map[0].mean(dim=0).cpu().numpy()  # (feat_H, feat_W)

feat_avg = (feat_avg - feat_avg.min()) / (feat_avg.max() - feat_avg.min() + 1e-8)

# current
#feat_upsampled = cv2.resize(feat_avg, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

# blocky — shows true resolution
feat_upsampled = cv2.resize(feat_avg, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

raw_weights = attention_maps[0][0] # Grab the first item in the batch

# Calculate where the image tokens start in the sequence
start_idx = 1 # 1 token for the latent variable
if policy.config.robot_state_feature:
    start_idx += 1 # 1 token for robot state
if policy.config.env_state_feature:
    start_idx += 1 # 1 token for env state

# Figure out the spatial dimensions (H, W) of the CNN feature map
dummy_feature_map = policy.model.backbone(batch[camera_key])["feature_map"]
_, _, feat_H, feat_W = dummy_feature_map.shape
num_image_tokens = feat_H * feat_W

# Slice out the image tokens from the 1D sequence
image_attn_1d = raw_weights[:, start_idx : start_idx + num_image_tokens]

# Reshape back into a 2D grid: (num_action_queries, feat_H, feat_W)
image_attn_2d = image_attn_1d.view(-1, feat_H, feat_W).numpy()

# Visualize the Heatmap
# Select the attention for the first action step in the chunk (index 0)
step_0_attn = image_attn_2d[0]

# Normalize the attention map to [0, 1] for visualization
step_0_attn = (step_0_attn - step_0_attn.min()) / (step_0_attn.max() - step_0_attn.min() + 1e-8)

# Resize the tiny feature map back to the original image size
attn_heatmap = cv2.resize(step_0_attn, (img_w, img_h))

# Apply a color map (Jet makes high attention red, low attention blue)
heatmap_colored = plt.get_cmap('jet')(attn_heatmap)[:, :, :3]

# Overlay the heatmap onto the original image
# (0.4 and 0.6 control the blending transparency)
overlay = original_image * 0.4 + heatmap_colored * 0.6
overlay = np.clip(overlay, 0, 1) # Ensure values remain in valid RGB range

# Plot the result
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title(f"Original Image (Frame {frame_index})")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(original_image, alpha=0.5)
plt.imshow(feat_upsampled, cmap='hot', alpha=0.5)
plt.title(f"Backbone Feature Map ({feature_map.shape[2]}×{feature_map.shape[3]} → upsampled)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title("Cross-Attention Heatmap")
plt.axis('off')

plt.tight_layout()
plt.show()