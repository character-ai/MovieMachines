import torch
import torch.nn as nn
# import devicetorch
# device = devicetorch.get(torch)

backwarp_tenGrid = {} # Cache for grid tensors

def warp(tenInput, tenFlow):
    # The key for caching should be based on tenFlow's properties, including its device
    k = (str(tenFlow.device), str(tenFlow.size())) 

    if k not in backwarp_tenGrid:
        # Create grid tensors on the same device as tenFlow
        flow_device = tenFlow.device 
        
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=flow_device)
            .view(1, 1, 1, tenFlow.shape[3])
            .expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        )
        tenVertical = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=flow_device)
            .view(1, 1, tenFlow.shape[2], 1)
            .expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        )
        # Concatenate; the result will be on flow_device if inputs are.
        # No need for an extra .to(device) if flow_device is used for components.
        backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1)

    # tenFlow is already on its correct device.
    # backwarp_tenGrid[k] is now also on that same device.

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        1,
    )

    # Ensure grid used by grid_sample is on the same device as tenInput and tenFlow.
    # backwarp_tenGrid[k] is already on tenFlow.device.
    # tenFlow is on tenFlow.device.
    # If tenInput can be on a different device than tenFlow, that's a separate issue.
    # Assuming tenInput and tenFlow are on the same device for grid_sample.
    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    
    return torch.nn.functional.grid_sample(
        input=tenInput, grid=g, mode="bilinear", padding_mode="border", align_corners=True
    )