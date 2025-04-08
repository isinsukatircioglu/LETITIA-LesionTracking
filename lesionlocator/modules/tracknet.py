import torch
from torch import nn
from torch.nn import functional as F
from icon_registration.mermaidlite import compute_warped_image_multiNC

class TrackNet(nn.Module):
    def __init__(self, reg_net, reg_net_patch_size, unet, unet_patch_size):
        super(TrackNet, self).__init__()
        assert len(reg_net_patch_size) == 3, "Input shape must be 3D"

        self.reg_net = reg_net
        self.unet = unet
        self.input_shape = reg_net_patch_size # [175, 175, 175]
        #  register as buffer
        self.register_buffer('unet_patch_size', torch.tensor(unet_patch_size))


    def optional_z_translation(self, x0, x1, difference_threshold=20):
        """
        Adjusts the z-dimension of two 3D tensors to align them based on their size difference.
        This function compares the z-dimensions of two input tensors `x0` and `x1`. If the difference
        in their z-dimensions exceeds the specified `difference_threshold`, it slides the smaller tensor
        along the z-dimension of the larger tensor to find the best alignment based on the lowest mean
        squared error (MSE). The tensors are then cropped to match their z-dimensions.
        Args:
            x0 (torch.Tensor): The first input tensor with shape (batch_size, channels, depth, height, width).
            x1 (torch.Tensor): The second input tensor with shape (batch_size, channels, depth, height, width).
            difference_threshold (int, optional): The maximum allowable difference in z-dimensions before
                alignment is performed. Defaults to 20.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
                - The adjusted tensor `x0` with its z-dimension cropped if necessary.
                - The adjusted tensor `x1` with its z-dimension cropped if necessary.
                - A list `[start, end]` indicating the cropping window for `x0` along the z-dimension.
                - A list `[start, end]` indicating the cropping window for `x1` along the z-dimension.
        """

        # If z dimension of images is very different, slide the smaller image to best crop the larger one
        x0_z, x1_z = x0.size(2), x1.size(2)
        x0_window, x1_window = [0, x0_z], [0, x1_z]
        if abs(x0_z - x1_z) > difference_threshold:
            # resample xy dimensions to 175x175
            x0_res = F.interpolate(x0, size=(x0_z, *self.input_shape[1:]), mode='trilinear', align_corners=False)
            x1_res = F.interpolate(x1, size=(x1_z, *self.input_shape[1:]), mode='trilinear', align_corners=False)
            if x0_z < x1_z:
                # slide the smaller image over the larger one and pick the spot with the lowest mse
                mse = torch.inf
                for i in range(x1_z - x0_z):
                    current_mse = F.mse_loss(x0_res, x1_res[:, :, i:i+x0_z])
                    if current_mse < mse:
                        mse = current_mse
                        x1_window = [i, i+x0_z]
            else:
                mse = torch.inf
                for i in range(x0_z - x1_z):
                    current_mse = F.mse_loss(x0_res[:, :, i:i+x1_z], x1_res)
                    if current_mse < mse:
                        mse = current_mse
                        x0_window = [i, i+x1_z]
            x0 = x0[:, :, x0_window[0]:x0_window[1]]
            x1 = x1[:, :, x1_window[0]:x1_window[1]]
        return x0, x1, x0_window, x1_window


    def get_patch_around_mask(self, prompt: torch.Tensor, is_inference: bool = False):
        """
        Extracts a patch around a mask from a 5D tensor input.
        This method processes a batch of 5D tensors, where each tensor represents 
        a volumetric mask. For each mask in the batch, it identifies a region of 
        interest (ROI) and extracts a patch centered around a foreground pixel 
        or a random pixel if no foreground is present.
        Args:
            prompt (torch.Tensor): A 5D tensor of shape (batch_size, channels, depth, height, width) 
                representing the input masks. The method assumes the mask is located in the first channel.
            is_inference (bool, optional): Flag indicating whether the method is being used for inference.
        Returns:
            list: A list of slicers for each mask in the batch. Each slicer is a list of six integers 
            [xmin, xmax, ymin, ymax, zmin, zmax] defining the bounds of the patch in 3D space.
        Raises:
            AssertionError: If the input tensor `prompt` is not 5D.
        """

        assert prompt.dim() == 5, "Mask must be 5D"

        # Iterate over batch
        slicers = []
        for i in range(prompt.size(0)):
            mask = prompt[i, 0]

            # sample random foreground pixel
            fg_indices = torch.nonzero(mask)
            if fg_indices.size(0) == 0:
                # if there is no foreground pixel, sample a random pixel
                shape = mask.size()
                fg_idx = torch.tensor([torch.randint(0, shape[0], (1,)), torch.randint(0, shape[1], (1,)), torch.randint(0, shape[2], (1,))], device=mask.device)
            else:
                if is_inference:
                    # Get center of mass
                    fg_idx = fg_indices.float().mean(0).round().int()
                else:
                    fg_idx = fg_indices[torch.randint(0, fg_indices.size(0), (1,))][0]

            # get patch around the foreground pixel
            min = fg_idx - self.unet_patch_size // 2
            max = fg_idx + self.unet_patch_size // 2

            for i in range(3):
                if min[i] < 0:
                    max[i] = self.unet_patch_size[i]
                    min[i] = 0
                if max[i] > mask.size(i):
                    min[i] = mask.size(i) - self.unet_patch_size[i]
                    max[i] = mask.size(i)
                if min[i] < 0:
                    min[i] = 0
            slicers.append([min[0], max[0], min[1], max[1], min[2], max[2]])  #xmin, xmax, ymin, ymax, zmin, zmax
        return slicers
    

    def pad_input(self, bb_input: torch.Tensor):
        """
        Pads the input tensor to match the UNet patch size.
        This method pads the input tensor to match the expected patch size of the UNet.
        Args:
            bb_input (torch.Tensor): The input tensor to be padded.
        Returns:
            torch.Tensor: The padded input tensor.
            bool: A flag indicating whether padding was applied.
            Tuple[int, int, int, int, int, int]: A tuple of padding values for each dimension.
        """
        if bb_input[0,0].shape != torch.Size(self.unet_patch_size):
            pad_x_min = (self.unet_patch_size[0] - bb_input[0,0].shape[0]) // 2
            pad_x_max = self.unet_patch_size[0] - bb_input[0,0].shape[0] - pad_x_min
            pad_y_min = (self.unet_patch_size[1] - bb_input[0,0].shape[1]) // 2
            pad_y_max = self.unet_patch_size[1] - bb_input[0,0].shape[1] - pad_y_min
            pad_z_min = (self.unet_patch_size[2] - bb_input[0,0].shape[2]) // 2
            pad_z_max = self.unet_patch_size[2] - bb_input[0,0].shape[2] - pad_z_min
            bb_input = F.pad(bb_input, (0, 0, 0, 0, pad_x_min, pad_x_max, pad_y_min, pad_y_max, pad_z_min, pad_z_max), mode='constant', value=0)
            return bb_input, True, (pad_x_min, pad_x_max, pad_y_min, pad_y_max, pad_z_min, pad_z_max)
        return bb_input, False, None
    

    def undo_pad(self, output: torch.Tensor, pad: bool, pad_values: tuple):
        """
        Reverts padding applied to the output tensor.
        This method reverts padding that was previously applied to the output tensor.
        Args:
            output (torch.Tensor): The output tensor to be reverted.
            pad (bool): A flag indicating whether padding was applied.
            pad_values (tuple): A tuple of padding values for each dimension.
        Returns:
            torch.Tensor: The reverted output tensor.
        """
        if pad:
            pad_x_min, pad_x_max, pad_y_min, pad_y_max, pad_z_min, pad_z_max = pad_values
            output = F.pad(output, (0, 0, 0, 0, -pad_x_min, -pad_x_max, -pad_y_min, -pad_y_max, -pad_z_min, -pad_z_max), mode='constant', value=0)
        return output
    

    def forward(self, x0, x1, prompt, is_inference=False):
        """
        Forward pass of the model.
        This method performs the forward computation for the model, including 
        resampling, registration, warping, patch extraction, and UNet inference. 
        It supports both training and inference modes.
        Args:
            x0 (torch.Tensor): The first input tensor, typically the reference image.
            x1 (torch.Tensor): The second input tensor, typically the moving image.
            prompt (torch.Tensor): A tensor representing the segmentation or prompt 
                associated with the reference image.
            is_inference (bool, optional): Flag indicating whether the method is 
                being used for inference. Defaults to False.
        Returns:
            torch.Tensor: 
                - During inference, returns the logits output tensor with the same 
                  spatial dimensions as `x1`.
                - During training, returns a tuple containing:
                    - The output tensor from the UNet.
                    - The registration loss.
                    - The cropped mask tensor for the moving image.
        """
        
        out_shape = x1.size()[2:]

        # Optional z-translation for inference
        if is_inference:
            x0, x1, x0_window, x1_window = self.optional_z_translation(x0, x1)
            prompt = prompt[:, :, x0_window[0]:x0_window[1]]

        # Resample to input shape
        x0_resampled = F.interpolate(x0, size=self.input_shape, mode='trilinear', align_corners=False)
        x1_resampled = F.interpolate(x1, size=self.input_shape, mode='trilinear', align_corners=False)
        prompt_resampled = F.interpolate(prompt, size=self.input_shape, mode='nearest-exact')
        
        # Registration
        reg_loss = self.reg_net(x0_resampled, x1_resampled)

        # Warp segmentation
        warped_prompt = compute_warped_image_multiNC(
            prompt_resampled,
            self.reg_net.phi_AB_vectorfield,
            self.reg_net.spacing,
            spline_order=0,
            zero_boundary=True
        )
        
        warped_prompt = F.interpolate(warped_prompt, size=x1.size()[2:], mode='nearest-exact')

        # Patch around the warped segmentation
        slicers = self.get_patch_around_mask(warped_prompt, is_inference)

        patch = torch.stack([warped_prompt[b, :, slicer[0]:slicer[1], slicer[2]:slicer[3], slicer[4]:slicer[5]] for b, slicer in enumerate(slicers)])
        x1 = x1.repeat(len(slicers), 1, 1, 1, 1)
        x1 = torch.stack([x1[b, :, slicer[0]:slicer[1], slicer[2]:slicer[3], slicer[4]:slicer[5]] for b, slicer in enumerate(slicers)])
        bb_input = torch.cat([x1, patch], dim=1)

        bb_input, pad, pad_properties = self.pad_input(bb_input)

        # Run the UNet
        output = self.unet(bb_input)

        # Revert padding if necessary
        output = self.undo_pad(output, pad, pad_properties)

        if is_inference: # expects batch size of 1
            assert len(slicers) == 1, "Batch size must be 1 for inference"

            logits_out = torch.cat([torch.ones((1, 1, *out_shape), device=x1.device), torch.zeros((1, 1, *out_shape), device=x1.device)], 1)
            slc = slicers[0]
            # adapt to x1_window
            slc[0] = slc[0] + x1_window[0]
            slc[1] = slc[1] + x1_window[0]
            logits_out[:, :, slc[0]:slc[1], slc[2]:slc[3], slc[4]:slc[5]] = output[:, :]

            return logits_out
        else:
            # x1_mask is only needed to be passed to the method during training
            x1_mask = torch.stack([x1_mask[b, :, slicer[0]:slicer[1], slicer[2]:slicer[3], slicer[4]:slicer[5]] for b, slicer in enumerate(slicers)])

            return output, reg_loss, x1_mask # for training
        