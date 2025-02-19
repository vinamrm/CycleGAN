"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from torchvision import transforms
import os
import h5py
import numpy as np
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
import random
# from data.image_folder import make_dataset
# from PIL import Image

def split_dataset_synthrad(data_path, split_ratio=(0.6, 0.1, 0.15, 0.15), seed=123, region='brain'):
    """
    Splits the dataset by patient ID while preserving the train-test ratio within each center group.

    Parameters:
        data_path (str): Path to the HDF5 dataset file.
        split_ratio (tuple): Ratio of data (between 0 and 1) for training, testing and calibration.
        seed (int): Random seed for reproducibility.
        region (str): Region to include in the split. Options are 'brain', 'pelvis', 'both'.

    Returns:
        train_ids (list): List of training patient IDs.
        test_ids (list): List of testing patient IDs.
    """
    assert sum(split_ratio) == 1, "Split ratio must sum to 1."
    
    # Initialize dictionary to hold patient IDs for each center
    center_patient_groups = {'A': [], 'B': [], 'C': []}
    
    with h5py.File(data_path, 'r') as f:
        # Loop over regions (brain, pelvis)
        for group_name in f.keys():  # 'brain', 'pelvis'
            # Only proceed if the region matches the desired one or 'both' is selected
            if region in ['both', group_name]:
                group = f[group_name]
                # Loop over centers (A, B, C)
                for subgroup_name in group.keys():  # 'A', 'B', etc.
                    if subgroup_name in center_patient_groups:
                        # Add patients of this center to the respective center group
                        center_patient_groups[subgroup_name].extend(list(group[subgroup_name].keys()))

    # Set random seed for reproducibility
    random.seed(seed)

    # Initialize train and test lists
    train_ids, val_ids, test_ids, calib_ids = [], [], [], []

    # Split patient IDs within each center
    for center, patients in center_patient_groups.items():
        random.shuffle(patients)

        train_split_idx = int(len(patients) * split_ratio[0])
        val_split_idx = int(len(patients) * (split_ratio[0] + split_ratio[1]))
        test_split_idx = int(len(patients) * (split_ratio[0] + split_ratio[1] + split_ratio[2]))

        train_ids.extend(patients[:train_split_idx])
        val_ids.extend(patients[train_split_idx:val_split_idx])
        test_ids.extend(patients[val_split_idx:test_split_idx])
        calib_ids = patients[test_split_idx:]

    return train_ids, val_ids, test_ids, calib_ids

class MultispectralDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(input_nc = 5, output_nc = 5)  # specify dataset-specific default values

        return parser
    
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        BaseDataset.__init__(self, opt)
        self.data_path = opt.dataroot  # get the image directory

        print(f'Parsing file {self.data_path}')
        
        
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        

        
        #get patient ids using split_dataset_synthrad method
        #TODO: check if region should be pelvis
        self.patient_ids = split_dataset_synthrad(self.data_path, split_ratio=(1, 0, 0, 0), region="pelvis")[0]

        self.input_label = "cbct.nii.gz"
        self.gt_label = "ct.nii.gz"
        self.mask_label = "mask.nii.gz"
        self.minmax_ctt = (-1000, 3000)
        self.contains_mask = True
        self.step_counter = 0
        
        #TODO: check if these variables should be these values of if they should be cmdline options
        self.normalize = "minmax"
        self.extended_slices = 2

        self.is_test = not opt.isTrain
        HW_DIM = 416
        IMG_SHAPE = (HW_DIM, HW_DIM)
        self.stride = 1
        self.transform = transforms.Compose([
        transforms.CenterCrop(IMG_SHAPE)
        ])

        #TODO: add model segmentation stuff and mask transform if necessary
        self.model_segmentation = False
        self.index_mapping, self.frame_mapping, self.total_length = self.build_mapping()

    def build_mapping(self):
        index_mapping = {}
        frame_mapping = {}
        total_index = 0
        
        with h5py.File(self.data_path, 'r') as f:
            for group_name in f.keys():  # 'brain', 'pelvis'
                group = f[group_name]
                for subgroup_name in group.keys():  # 'A', 'B', etc.
                    subgroup = group[subgroup_name]
                    for patient_id in subgroup.keys():  # '2BA001', '2BA002', etc.
                        if self.patient_ids is not None and patient_id not in self.patient_ids:
                            continue
                            
                        patient_group = subgroup[patient_id]
                        if self.input_label in patient_group:
                            image_data = patient_group[self.gt_label] # gt_label mask_label
                            _, _, l = image_data.shape

                            start_frame = 5
                            for i in range(start_frame, l, self.stride):

                                #check frame for a >75% 0s in the mask
                                # if np.mean(image_data[:, :, i][:] == 0) > 0.75:
                                #     continue  # Skip this frame
                                # if not (tfmae.return_min_presence(image_data[:, :, i]), ratio=0.3):
                                #     continue
                                    
                                index_mapping[total_index] = (group_name, subgroup_name, patient_id)
                                frame_mapping[total_index] = int(i)
                                total_index += 1
        
        return index_mapping, frame_mapping, total_index
    
    def normalize_image(self, image, patient_id):
        if self.normalize == "minmax":
            minmax_ctt = self.minmax_ctt
            normalized_image = -1 + 2 * ((image - minmax_ctt[0]) / (minmax_ctt[1] - minmax_ctt[0]))  
        elif self.normalize == "zscore":
            normalized_image = (image - image.mean()) / image.std()
        return normalized_image

    def __getitem__(self, idx):

        group_name, subgroup_name, patient_id = self.index_mapping[idx]
        frame_index = self.frame_mapping[idx]
        
        #print(group_name, subgroup_name, patient_id)
        
        with h5py.File(self.data_path, 'r') as f:
            patient_group = f[group_name][subgroup_name][patient_id]

            # Input image
            image_dataset = patient_group[self.input_label]
            image_data = image_dataset[:]

            spacing = image_dataset.attrs['spacing']
            origin = image_dataset.attrs['origin']
            
            # print(f"Loaded spacing: {spacing}, type: {type(spacing)}")
            # print(f"Loaded origin: {origin}, type: {type(origin)}")

            # Input image
            h, w, l = image_data.shape

            # Update frame index randomly for the stride window so we explore more slices when stride > 1
            # if self.stride > 1 and not self.is_test:
            #     np.random.seed(self.step_counter)
            #     max_index = min(frame_index + self.stride, l)
            #     if frame_index < max_index:
            #         frame_index = np.random.randint(frame_index, max_index)
            #     self.step_counter += 1
            # elif self.stride > 1 and self.is_test:
            #     np.random.seed(idx)
            #     max_index = min(frame_index + self.stride, l)
            #     if frame_index < max_index:
            #         frame_index = np.random.randint(frame_index, max_index)
            
            image = np.zeros((h, w, 2*self.extended_slices + 1), dtype=np.float32)
            start_index = frame_index - self.extended_slices
            end_index = frame_index + self.extended_slices
            
            valid_start = max(0, start_index)
            valid_end = min(l, end_index+1)
            copy_start = max(0, -start_index)
            copy_end = copy_start + (valid_end - valid_start)
            image[:, :, copy_start:copy_end] = image_data[:, :, valid_start:valid_end]
            
            # Ground truth image
            gt_image_data = patient_group[self.gt_label][:]
            gt_image = gt_image_data[:, :, frame_index]
            
            # Mask image
            if self.contains_mask:
                mask_image_data = patient_group[self.mask_label][:]
                mask_image = mask_image_data[:, :, frame_index]
            
        # Convert to torch tensors
        image = torch.tensor(image).unsqueeze(0)  # Add channel dimension
        gt_image = torch.tensor(gt_image).unsqueeze(0)
        if self.contains_mask:
            mask_image = torch.tensor(mask_image).unsqueeze(0)
        
        # Apply transforms
        if self.transform:
            transformed_frames = []
            # state = torch.get_rng_state()
            
            if self.is_test:
                state = torch.manual_seed(idx).get_state()
            else:
                state = torch.manual_seed(self.step_counter).get_state()
                self.step_counter += 1            
                
            for frame in range(image.shape[-1]):
                
                torch.set_rng_state(state)
                    
                transformed_frame = self.transform(image[:, :, :, frame])
                transformed_frames.append(transformed_frame)
            
            torch.set_rng_state(state)
            gt_image = self.transform(gt_image)
            
            if self.contains_mask:
                torch.set_rng_state(state)
                mask_image = self.mask_transform(mask_image)
            
            image = torch.stack(transformed_frames, dim=1).squeeze(0)
        
        # Clipping and Normalization
        image = torch.clamp(image, min=self.minmax_ctt[0], max=self.minmax_ctt[1])
        gt_image = torch.clamp(gt_image, min=self.minmax_ctt[0], max=self.minmax_ctt[1])

        #TODO: if segmentation is needed, uncomment below code
        # CT segmentation before norm
        #gt_image_masked = torch.where(mask_image == 0.0, torch.tensor(self.minmax_ctt[0]),gt_image)
        #segmenter = SegmentationCT2D(gt_image_masked.numpy())
        #body_mask = segmenter.segment_body()
        #bones_mask = segmenter.segment_bones(body_mask)
        #seg_ct = torch.stack([torch.tensor(body_mask), torch.tensor(bones_mask)], dim=0).float()
                
        image = self.normalize_image(image, patient_id)
        gt_image = self.normalize_image(gt_image, patient_id)
        
        # Without control architecture
        # image = torch.cat((image, seg_ct), 0)

        #if we are using segmentation model for cbct seg instead of ct seg
        #if self.segmentation_model is not None:
        #    with torch.no_grad():
        #        seg_ct = self.segmentation_model(image.unsqueeze(0)).squeeze(0)        

        #segmentation only
        #if self.mode == "onlyseg":
        #    image = seg_ct
        #elif self.mode == "withseg":
        #    image = torch.cat((image, seg_ct), 0)

        if self.contains_mask:
            data_dict = {"data_A": image, "data_B": gt_image, "mask": mask_image.float(), "spacing": spacing, "origin": origin, "d" : frame_index, "d_max" : l, "patient_id": patient_id} #, "seg_ct": seg_ct}
        else:
            data_dict = {"data_A": image.squeeze(0), "data_B": gt_image, "spacing": spacing, "origin": origin}
        
        return data_dict

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
