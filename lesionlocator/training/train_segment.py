import itertools
import multiprocessing
import os
import traceback
import json
import numpy as np
from queue import Queue
from threading import Thread
from time import sleep
from typing import Tuple, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
import json
import SimpleITK
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json, subfiles
from torch._dynamo import OptimizedModule
from tqdm import tqdm

import lesionlocator
from lesionlocator.preprocessing.resampling.default_resampling import compute_new_shape
from lesionlocator.configuration import default_num_processes
from lesionlocator.training.data_iterators import preprocessing_iterator_fromfiles
from lesionlocator.inference.export_prediction import export_prediction_from_logits
from lesionlocator.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from lesionlocator.utilities.file_path_utilities import check_workers_alive_and_busy
from lesionlocator.utilities.find_class_by_name import recursive_find_python_class
from lesionlocator.utilities.helpers import empty_cache, dummy_context
from lesionlocator.utilities.label_handling.label_handling import determine_num_input_channels
from lesionlocator.utilities.plans_handling.plans_handler import PlansManager
from lesionlocator.utilities.prompt_handling.prompt_handler import sparse_to_dense_prompt
from lesionlocator.utilities.surface_distance_based_measures import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient, compute_robust_hausdorff


def dice_loss(pred, target, epsilon=1e-6):
    """
    Dice loss function for segmentation training.
    
    Args:
        pred: Model predictions [B, C, H, W, D] (logits)
        target: Ground truth labels [B, H, W, D] (class indices)
        epsilon: Small value to avoid division by zero
        
    Returns:
        Dice loss value (1 - dice_coefficient)
    """
    pred_soft = torch.softmax(pred, dim=1)
    target_onehot = nn.functional.one_hot(target.long(), num_classes=pred.shape[1])
    target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float() if pred.dim() == 5 else target_onehot.permute(0, 3, 1, 2).float()
    
    dims = (0,) + tuple(range(2, pred.dim()))
    intersection = torch.sum(pred_soft * target_onehot, dims)
    union = torch.sum(pred_soft, dims) + torch.sum(target_onehot, dims)
    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    return 1 - dice.mean()


def create_cv_folds(input_files, prompt_files, output_files, n_folds=5, random_seed=42):
    """
    Create cross-validation folds for training data.
    
    Args:
        input_files: List of training input files
        prompt_files: List of training prompt files  
        output_files: List of training output files
        n_folds: Number of folds for cross-validation
        random_seed: Random seed for reproducibility
        
    Returns:
        List of fold dictionaries, each containing train and val splits
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Create indices for the files
    indices = list(range(len(input_files)))
    
    # Create KFold splitter
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    folds = []
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(indices)):
        fold = {
            'fold_idx': fold_idx,
            'train': {
                'input_files': [input_files[i] for i in train_indices],
                'prompt_files': [prompt_files[i] for i in train_indices],
                'output_files': [output_files[i] for i in train_indices]
            },
            'val': {
                'input_files': [input_files[i] for i in val_indices],
                'prompt_files': [prompt_files[i] for i in val_indices], 
                'output_files': [output_files[i] for i in val_indices]
            }
        }
        folds.append(fold)
        
    return folds


class LesionDatasetWrapper(IterableDataset):
    """
    PyTorch IterableDataset wrapper that preserves the existing multiprocessing pipeline
    while providing PyTorch DataLoader compatibility for training.
    
    This wrapper:
    1. Preserves all existing multiprocessing data loading
    2. Converts each lesion instance into a training sample
    3. Provides PyTorch DataLoader compatibility
    4. Maintains all preprocessing logic unchanged
    
    Example usage:
        # Create trainer instance
        trainer = LesionLocatorSegmenter(device=torch.device('cuda'))
        trainer.initialize_from_trained_model_folder(model_dir, track_dir, folds)
        
        # Create datasets
        train_dataset = trainer.create_training_dataset(
            input_files=['img1.nii.gz', 'img2.nii.gz'],
            prompt_files=['prompt1.nii.gz', 'prompt2.nii.gz'],
            output_files=['out1', 'out2'],
            prompt_type='box'
        )
        
        # Train
        trainer.train(train_dataset, epochs=100, lr=1e-4)
    """
    def __init__(self, input_files, prompt_files, output_files, prompt_type, 
                 plans_manager, dataset_json, configuration_manager, 
                 num_processes=3, pin_memory=False, verbose=False, track=False):
        self.input_files = input_files
        self.prompt_files = prompt_files
        self.output_files = output_files
        self.prompt_type = prompt_type
        self.plans_manager = plans_manager
        self.dataset_json = dataset_json
        self.configuration_manager = configuration_manager
        self.num_processes = num_processes
        self.pin_memory = pin_memory
        self.verbose = verbose
        self.track = track
        
    def __iter__(self):
        """
        Create the multiprocessing data iterator and yield training samples.
        This preserves the existing preprocessing pipeline completely.
        """
        data_iterator = preprocessing_iterator_fromfiles(
            self.input_files, self.prompt_files, self.output_files,
            self.prompt_type, self.plans_manager, self.dataset_json,
            self.configuration_manager, self.num_processes, self.pin_memory,
            self.verbose, self.track
        )
        
        for preprocessed in data_iterator:
            data = preprocessed['data']
            prompt = preprocessed['prompt']
            seg_mask = preprocessed['seg']
            properties = preprocessed['data_properties']
            
            # Convert each lesion instance into a training sample
            for inst_id, p in enumerate(prompt):
                if len(p) == 0:
                    continue
                    
                inst_id += 1
                gt_mask = (seg_mask == inst_id).astype(np.uint8)
                
                # Convert sparse prompt to dense format
                p_dense = sparse_to_dense_prompt(p, self.prompt_type, array=data)
                if p_dense is None:
                    continue
                
                # Yield training sample - convert to torch tensors for consistent batching
                # Handle both numpy arrays and already converted tensors
                if isinstance(data, torch.Tensor):
                    data_tensor = data.float()
                else:
                    data_tensor = torch.from_numpy(data).float()
                
                if isinstance(p_dense, torch.Tensor):
                    prompt_tensor = p_dense.float()
                else:
                    prompt_tensor = torch.from_numpy(p_dense).float()
                
                if isinstance(gt_mask[0], torch.Tensor):
                    target_tensor = gt_mask[0].long()
                else:
                    target_tensor = torch.from_numpy(gt_mask[0]).long()
                
                yield {
                    'data': data_tensor,                    # Input image [C, H, W, D]
                    'prompt': prompt_tensor,               # Dense prompt [1, H, W, D]
                    'target': target_tensor,               # Ground truth [H, W, D]
                    'properties': properties,               # Metadata
                    'lesion_id': inst_id,                  # Lesion instance ID
                    'filename': preprocessed['ofile']      # Original filename
                }


def training_collate_fn(batch):
    """
    Custom collate function for batching training samples.
    Now supports true batching with fixed patch sizes from TrainingPreprocessor.
    """
    if len(batch) == 1:
        return batch[0]
    
    # Stack all batch items into proper tensors
    batch_data = []
    batch_prompts = []
    batch_targets = []
    batch_properties = []
    batch_lesion_ids = []
    batch_filenames = []
    
    for item in batch:
        batch_data.append(item['data'])
        batch_prompts.append(item['prompt'])
        batch_targets.append(item['target'])
        batch_properties.append(item['properties'])
        batch_lesion_ids.append(item['lesion_id'])
        batch_filenames.append(item['filename'])
    
    # Stack tensors - all should have same dimensions due to crop_to_patch_size
    try:
        stacked_data = torch.stack(batch_data, dim=0)       # [B, C, H, W, D]
        stacked_prompts = torch.stack(batch_prompts, dim=0) # [B, 1, H, W, D]
        stacked_targets = torch.stack(batch_targets, dim=0) # [B, H, W, D]
        
        return {
            'data': stacked_data,
            'prompt': stacked_prompts,
            'target': stacked_targets,
            'properties': batch_properties,
            'lesion_id': batch_lesion_ids,
            'filename': batch_filenames
        }
    except RuntimeError as e:
        # Print shapes for debugging
        print(f"Batch stacking failed: {e}")
        print(f"Data shapes: {[d.shape for d in batch_data]}")
        print(f"Prompt shapes: {[p.shape for p in batch_prompts]}")
        print(f"Target shapes: {[t.shape for t in batch_targets]}")
        # Fallback: if shapes don't match, process as batch_size=1
        print(f"Warning: Could not stack batch, falling back to single sample processing")
        return batch[0]


class LesionLocatorSegmenter(object):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True,
                 visualize: bool = False,
                 track: bool = False,
                 adaptive_mode: bool = False):
        self.verbose = verbose
        self.verbose_preprocessing = verbose_preprocessing
        self.allow_tqdm = allow_tqdm

        self.plans_manager, self.configuration_manager, self.list_of_parameters, self.network, self.dataset_json, \
        self.trainer_name, self.allowed_mirroring_axes, self.label_manager = None, None, None, None, None, None, None, None

        # Training-specific attributes
        self.optimizer = None
        self.loss_function = None
        self.scheduler = None
        self.training_mode = False

        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        else:
            print(f'perform_everything_on_device=True is only supported for cuda devices! Setting this to False')
            perform_everything_on_device = False
        self.device = device
        self.perform_everything_on_device = perform_everything_on_device
        self.visualize = visualize
        self.track = track
        self.adaptive_mode = adaptive_mode
        
        print('Tracking: ', self.track)
        print('Adaptive mode: ', self.adaptive_mode)

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             model_track_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             modality: str = 'ct',
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        print("Loading segmentation model.")
        if use_folds is None:
            use_folds = LesionLocatorSegmenter.auto_detect_available_folds(model_training_output_dir, checkpoint_name)
        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'), weights_only=False)
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None

            parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name, modality=modality)
        configuration_manager.set_preprocessor_name('TrainingPreprocessor')
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(lesionlocator.__path__[0], "training", "LesionLocatorTrainer"),
                                                    trainer_name, 'lesionlocator.training.LesionLocatorTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in lesionlocator.training.LesionLocatorTrainer. '
                               f'Please place it there (in any .py file)!')
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        
        # Store configuration name for checkpoint saving
        self.configuration_name = configuration_name

        network.load_state_dict(parameters[0])
        
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if ('LesionLocator_compile' in os.environ.keys()) and (os.environ['LesionLocator_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

        #Tracker network
        dataset_json_tracker = load_json(join(model_track_training_output_dir, 'dataset.json'))
        plans_tracker = load_json(join(model_track_training_output_dir, 'plans.json'))
        plans_manager_tracker = PlansManager(plans_tracker)

        parameters_tracker = []
        for i, f in enumerate(use_folds):
            print(f'Loading fold {f}')
            f = int(f) if f != 'all' else f
            checkpoint_tracker = torch.load(join(model_track_training_output_dir, f'fold_{f}', "checkpoint_final.pth"),
                                    map_location=torch.device('cpu'), weights_only=False)
            if i == 0:
                trainer_name_tracker = checkpoint_tracker['trainer_name']
                configuration_name_tracker = checkpoint_tracker['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint_tracker['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint_tracker.keys() else None

            parameters_tracker.append(checkpoint_tracker['network_weights'])

        configuration_manager_tracker = plans_manager_tracker.get_configuration(configuration_name_tracker)
        # set spacing
        configuration_manager.set_spacing([1.5, 1.5, 1.5])
        configuration_manager_tracker.set_spacing([1.5, 1.5, 1.5])
        # restore networks
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager_tracker, dataset_json_tracker)
        trainer_class = recursive_find_python_class(join(lesionlocator.__path__[0], "training", "LesionLocatorTrainer"),
                                                    trainer_name_tracker, 'lesionlocator.training.LesionLocatorTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name_tracker} in lesionlocator.training.LesionLocatorTrainer. '
                               f'Please place it there (in any .py file)!')
        network_tracker = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            configuration_manager.patch_size,
            enable_deep_supervision=False
        )
       
        self.plans_manager_tracker = plans_manager_tracker
        self.configuration_manager_tracker = configuration_manager_tracker
        self.list_of_parameters_tracker = parameters_tracker

        network_tracker.load_state_dict(parameters_tracker[0])
        self.network_tracker = network_tracker
        self.dataset_json_tracker = dataset_json_tracker
        self.trainer_name_tracker = trainer_name_tracker
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json_tracker)
        self.tile_step_size = 0.5
        self.use_gaussian = True
        self.use_mirroring = True
        self.target_spacing = self.configuration_manager.spacing
        if self.track:
            self.target_spacing = self.configuration_manager_tracker.spacing
        print('Using target spacing: ', self.target_spacing)
        print('Segmentation configuration: ', self.configuration_manager)
        print('Tracking configuration: ', self.configuration_manager_tracker)

    @staticmethod
    def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
        print('use_folds is None, attempting to auto detect available folds')
        fold_folders = subdirs(model_training_output_dir, prefix='fold_', join=False)
        fold_folders = [i for i in fold_folders if i != 'fold_all']
        fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
        use_folds = [int(i.split('_')[-1]) for i in fold_folders]
        print(f'found the following folds: {use_folds}')
        return use_folds


    def predict_from_files(self,
                           source_folder_or_file: str,
                           output_folder_or_file: str,
                           prompt_folder_or_file: str,
                           prompt_type: str,
                           overwrite: bool = True,
                           num_processes_preprocessing: int = default_num_processes,
                           num_processes_segmentation_export: int = default_num_processes,
                           num_parts: int = 1,
                           part_id: int = 0):
        """
        This is the default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        assert part_id <= num_parts, ("Part ID must be smaller than num_parts. Remember that we start counting with 0. "
                                      "So if there are 3 parts then valid part IDs are 0, 1, 2")
        if os.path.isdir(source_folder_or_file):
            assert os.path.isdir(output_folder_or_file) and os.path.isdir(prompt_folder_or_file), \
                "If '-i' is a folder then '-o' (output) and '-p' (prompt) must also be folders."
            # list and sort all the files
            input_files = subfiles(source_folder_or_file, suffix=self.dataset_json['file_ending'], join=True, sort=True)
            prompt_files_json = subfiles(prompt_folder_or_file, suffix='.json', join=True, sort=True)
            prompt_files_mask = subfiles(prompt_folder_or_file, suffix=self.dataset_json['file_ending'], join=True, sort=True)
            output_files = [join(output_folder_or_file, os.path.basename(i)) for i in input_files]
            
            # Assertions
            if len(input_files) == 0:
                print(f'No files found in {source_folder_or_file}')
                return
            assert len(prompt_files_json) == 0 or len(prompt_files_mask) == 0, \
                "Prompt folder must contain either json files or mask files, not both."
            assert len(input_files) == len(prompt_files_json) or len(input_files) == len(prompt_files_mask), \
                "Number of files in source folder and prompt folder must be the same."
            
            prompt_files = prompt_files_json if len(prompt_files_json) > 0 else prompt_files_mask
            
            # Check if the output folder exists
            if not os.path.isdir(output_folder_or_file):
                os.makedirs(output_folder_or_file)
            else:
                if not overwrite:
                    # Remove already predicted files from the lists
                    existing_files = [os.path.isfile(i) for i in output_files]
                    not_existing_indices = [i for i, j in enumerate(input_files) if j not in existing_files]
                    input_files = [input_files[i] for i in not_existing_indices]
                    prompt_files = [prompt_files[i] for i in not_existing_indices]
                    output_files = [output_files[i] for i in not_existing_indices]
        else:
            assert not os.path.isdir(prompt_folder_or_file), \
                "If '-i' is a file then '-p' (prompt) must also be files not folders."
            input_files = [source_folder_or_file]
            prompt_files = [prompt_folder_or_file]
            output_files = [join(output_folder_or_file, os.path.basename(source_folder_or_file))]

        # Truncate output files
        output_files = [i.replace(self.dataset_json['file_ending'], '') for i in output_files]
        data_iterator = preprocessing_iterator_fromfiles(input_files, prompt_files,
                                                output_files, prompt_type, self.plans_manager, self.dataset_json,
                                                self.configuration_manager, num_processes_preprocessing, self.device.type == 'cuda',
                                                self.verbose_preprocessing, self.track)
       
        return self.predict_from_data_iterator(data_iterator, prompt_type, output_folder_or_file, num_processes_segmentation_export)


    def predict_from_data_iterator(self,
                                   data_iterator,
                                   prompt_type: str,
                                   output_folder_or_file: str,
                                   num_processes_segmentation_export: int = default_num_processes):
        """
        This function takes a data iterator and makes predictions and saves each instance (lesion) as a separate file.
        """
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            error_all={'dice': {'mean':0, 'TP0':{'all':[], 'mean':0}, 'TP1': {'all':[], 'mean':0}, 'TP2': {'all':[], 'mean':0}}, 
               'nsd': {'mean':0, 'TP0':{'all':[], 'mean':0}, 'TP1': {'all':[], 'mean':0}, 'TP2': {'all':[], 'mean':0}},
               'hausdorff': {'mean':0, 'TP0':{'all':[], 'mean':0}, 'TP1': {'all':[], 'mean':0}, 'TP2': {'all':[], 'mean':0}},
               'lesion_found':{'all':0, 'mean':0, 'TP0':{'all':0, 'mean':0}, 'TP1':{'all':0, 'mean':0}, 'TP2':{'all':0, 'mean':0}},
               'lesion_all': {'all':0, 'TP0':{'all':0}, 'TP1':{'all':0}, 'TP2':{'all':0}}}
            dice_score_all = []
            hausdorff_score_all = []
            nsd_score_all = []
            metrics = {
                'dice': 0.0,
                'hausdorff': 0.0,
                'nsd': 0.0
            }
            for preprocessed in data_iterator:
                data = preprocessed['data']
                #baseline data, None for TP0 scans
                bl_data = preprocessed['bl_data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)
                ofile = preprocessed['ofile']
                print(f'\n === Predicting {os.path.basename(ofile)} === ')
                patient_tp = os.path.basename(ofile)
                timepoint = os.path.basename(ofile).split('_')[0]
                properties = preprocessed['data_properties']
                prompt = preprocessed['prompt']
                seg_mask = preprocessed['seg']
                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with files
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                if len(prompt) == 0:
                    print(f" No prompt found for {os.path.basename(ofile)}")
                else:
                    for inst_id, p in enumerate(prompt):
                        inst_id += 1
                        gt_mask = ((seg_mask == inst_id).astype(np.uint8))
                        if len(p) == 0:
                            print(f"--- No prompt found for Lesion ID {inst_id} ---")
                            continue                 
                        print(f'\n Lesion ID {inst_id}: ')
                        for k in error_all.keys():
                            if k == 'lesion_all' or k == 'lesion_found':
                                continue
                            if os.path.basename(ofile) not in error_all[k].keys():
                                error_all[k][patient_tp]={'mean':0, 'per_lesion':[]}
                        p_sparse = p
                        p = sparse_to_dense_prompt(p, prompt_type, array=data)
                        if p is None:
                            print(f" Invalid prompt found for {os.path.basename(ofile)}")
                            continue
                        #Check if there is an existing segmentation mask for the baseline image
                        tp_order = ['TP2', 'TP1', 'TP0']
                        current_tp = None
                        for tp in tp_order:
                            if tp in os.path.basename(ofile):
                                current_tp = tp
                                break

                        prev_tp = None
                        use_prev_tp = False
                        low_score = False
                        if current_tp == 'TP2':
                            # Try TP1 first, then TP0
                            for candidate in ['TP1', 'TP0']:
                                prev_tp_candidate = os.path.basename(ofile).replace('TP2', candidate)
                                if os.path.exists(os.path.join(output_folder_or_file, prev_tp_candidate + '_lesion_' + str(inst_id) + '.nii.gz')):
                                    prev_tp = prev_tp_candidate
                                    break
                        elif current_tp == 'TP1':
                            prev_tp_candidate = os.path.basename(ofile).replace('TP1', 'TP0')
                            if os.path.exists(os.path.join(output_folder_or_file, prev_tp_candidate + '_lesion_' + str(inst_id) + '.nii.gz')):
                                prev_tp = prev_tp_candidate
                        
                        if self.track and (prev_tp is not None):
                            use_prev_tp = True
                            prev_seg_sitk = SimpleITK.ReadImage(os.path.join(output_folder_or_file, prev_tp+'_lesion_'+str(inst_id)+'.nii.gz'))
                            original_spacing = prev_seg_sitk.GetSpacing()[::-1]
                            print('Reading segmentation mask with spacing: ', original_spacing, ', target spacing is: ', self.target_spacing)
                            # Convert to numpy and compute new shape
                            prev_seg_np = SimpleITK.GetArrayFromImage(prev_seg_sitk)
                            new_shape = compute_new_shape(prev_seg_np.shape, original_spacing, self.target_spacing)
                            bl_spacing = (prev_seg_np.shape[0]* original_spacing[0] /  bl_data.shape[1],
                                        prev_seg_np.shape[1] * original_spacing[1] /  bl_data.shape[2],
                                        prev_seg_np.shape[2] * original_spacing[2] /  bl_data.shape[3])
                            #print('BL SPACING: ', bl_spacing)
                            #print('New shape for resampling: ', new_shape)
                            #print('BL DATA SHAPE FOR RESAMPLING: ', bl_data.shape)
                            prev_seg_resampled = self.configuration_manager.resampling_fn_seg(
                                prev_seg_np[None], 
                                bl_data.shape[1:], 
                                original_spacing, 
                                bl_spacing
                            )[0]
                            print('Use previous timepoint prediction as prompt: ', prev_tp+'_lesion_'+str(inst_id))
                            prompt_bl = torch.from_numpy(prev_seg_resampled).unsqueeze(0).to(self.device).half()
                            print('Resampled prompt shape: ', prompt_bl.shape)
                            # Predict the logits using the preprocessed data and the prompt
                            prediction = self.track_single_lesion(torch.from_numpy(bl_data[np.newaxis,:]).to(self.device), data.unsqueeze(0).to(self.device), prompt_bl.unsqueeze(0)).cpu()
                            seg = torch.softmax(prediction, 0).argmax(0)
                            pred = seg.detach().cpu().numpy().astype(np.uint8)
                            print('Prediction shape: ', pred.shape)
                            print('Ground truth shape: ',  gt_mask[0].shape)
                            dice_score = compute_dice_coefficient(gt_mask[0], pred)
                            if dice_score < 0.1:
                                print(f'Low Dice score {dice_score:.2f} for lesion {inst_id} at timepoint {timepoint}. Disabling tracking...')
                                low_score = True
                            
                        if (not self.track) or (prev_tp is None) or (low_score and self.adaptive_mode):
                            use_prev_tp = False
                            print('Use current timepoint ground truth as prompt: ', p.shape)
                            # Predict the logits using the preprocessed data and the prompt
                            prediction = self.predict_logits_from_preprocessed_data(data, p).cpu()
                            seg = torch.softmax(prediction, 0).argmax(0)
                            pred = seg.detach().cpu().numpy().astype(np.uint8)
                            print('Prediction shape: ', pred.shape)
                            print('Ground truth shape: ', gt_mask[0].shape)
                            dice_score = compute_dice_coefficient(gt_mask[0], pred)
                        
                        error_all['lesion_all']['all']+=1
                        error_all['lesion_all'][timepoint]['all']+=1
                        if dice_score >= 0.1:
                            error_all['lesion_found']['all']+=1
                            error_all['lesion_found'][timepoint]['all']+=1
                        print('Dice Score: ', dice_score)
                        surface_distances = compute_surface_distances(gt_mask[0], pred, self.target_spacing)
                        hausdorff_score = compute_robust_hausdorff(surface_distances, 95)
                        nsd_score = compute_surface_dice_at_tolerance(surface_distances, 2)
                        
                        dice_score_all.append(dice_score)
                        hausdorff_score_all.append(hausdorff_score)
                        nsd_score_all.append(nsd_score)
                        metrics = {
                            'dice': dice_score,
                            'hausdorff': hausdorff_score,
                            'nsd': nsd_score
                        }
                        # Update all metrics in a loop
                        for metric_name, score in metrics.items():
                            error_all[metric_name][timepoint]['all'].append(score)
                            error_all[metric_name][patient_tp]['per_lesion'].append(score)
                        print('Avg Mean Dice: ', np.mean(dice_score_all))
                        print('Avg Mean Hausdorff: ',  np.mean(hausdorff_score_all))
                        print('Avg Mean NSD: ', np.mean(nsd_score_all))
                        print('Avg Lesion Detection Score: {:.2f}%'.format((error_all['lesion_found']['all'] / error_all['lesion_all']['all']) * 100))
                        with open(os.path.join(output_folder_or_file, 'error_dict.json'), 'w') as fjson:
                            json.dump(error_all, fjson)
                        print('----------')
                        out_file = ofile + f'_lesion_{inst_id}'
                        # Visualize the prediction
                        if self.visualize:
                            subplot_count = 3
                            if use_prev_tp:
                                subplot_count = 4
                            
                            # Find axial slice with most lesion pixels
                            mask_ones_gt_axial = np.where(gt_mask[0] == 1)
                            if len(mask_ones_gt_axial[0]) > 0:  # Check if mask is not empty
                                # Find the z-slice with most mask voxels for axial view
                                largest_mask_slice_id_axial = np.bincount(mask_ones_gt_axial[0]).argmax()
                            
                            # Find coronal slice with most lesion pixels
                            mask_ones_gt_coronal = np.where(gt_mask[0] == 1)
                            if len(mask_ones_gt_coronal[1]) > 0:  # Check if mask is not empty
                                # Find the y-slice with most mask voxels for coronal view
                                largest_mask_slice_id_coronal = np.bincount(mask_ones_gt_coronal[1]).argmax()

                            # Create subplot with 2 rows
                            fig, axs = plt.subplots(2, subplot_count, figsize=(subplot_count * 4, 8))
                            
                            # First row - Axial View
                            # Original img
                            axs[0,0].imshow(data[0][largest_mask_slice_id_axial, :, :].detach().cpu().numpy(), cmap='gray')
                            axs[0,0].set_title('Image (Axial)') 
                            axs[0,0].axis('off')
                            # Ground truth
                            axs[0,1].imshow(data[0][largest_mask_slice_id_axial, :, :].detach().cpu().numpy(), cmap='gray')
                            axs[0,1].imshow(gt_mask[0][largest_mask_slice_id_axial, :, :]*255, alpha=0.5)
                            axs[0,1].set_title('Ground truth') 
                            axs[0,1].axis('off')
                            # Predictions
                            axs[0,2].imshow(data[0][largest_mask_slice_id_axial, :, :].detach().cpu().numpy(), cmap='gray')
                            axs[0,2].imshow(pred[largest_mask_slice_id_axial, :, :], alpha=0.5)
                            axs[0,2].set_title('Prediction') 
                            axs[0,2].axis('off')

                            # Second row - Coronal View
                            # Original img
                            axs[1,0].imshow(data[0][:, largest_mask_slice_id_coronal, :].detach().cpu().numpy(), cmap='gray', origin='lower')
                            axs[1,0].set_title('Image (Coronal)') 
                            axs[1,0].axis('off')
                            # Ground truth
                            axs[1,1].imshow(data[0][:, largest_mask_slice_id_coronal, :].detach().cpu().numpy(), cmap='gray', origin='lower')
                            axs[1,1].imshow(gt_mask[0][:, largest_mask_slice_id_coronal, :]*255, alpha=0.5, origin='lower')
                            axs[1,1].set_title('Ground truth') 
                            axs[1,1].axis('off')
                            # Predictions
                            axs[1,2].imshow(data[0][:, largest_mask_slice_id_coronal, :].detach().cpu().numpy(), cmap='gray', origin='lower')
                            axs[1,2].imshow(pred[:, largest_mask_slice_id_coronal, :], alpha=0.5, origin='lower')
                            axs[1,2].set_title('Prediction') 
                            axs[1,2].axis('off')
                            if use_prev_tp:
                                prompt_bl = prompt_bl[0].detach().cpu().numpy()
                                try:
                                    # Axial view for baseline
                                    mask_ones_gt_axial_bl = np.where(prev_seg_resampled == 1)
                                    largest_mask_slice_id_axial_bl = np.bincount(mask_ones_gt_axial_bl[0]).argmax()
                                    axs[0,3].imshow(bl_data[0][largest_mask_slice_id_axial_bl, :, :], cmap='gray')
                                    axs[0,3].imshow(prev_seg_resampled[largest_mask_slice_id_axial_bl, :, :], alpha=0.5)
                                    axs[0,3].set_title('Baseline prompt') 
                                    axs[0,3].axis('off')

                                    # Coronal view for baseline
                                    mask_ones_gt_coronal_bl = np.where(prev_seg_resampled == 1)
                                    largest_mask_slice_id_coronal_bl = np.bincount(mask_ones_gt_coronal_bl[1]).argmax()
                                    axs[1,3].imshow(bl_data[0][:, largest_mask_slice_id_coronal_bl, :], cmap='gray', origin='lower')
                                    axs[1,3].imshow(prev_seg_resampled[:, largest_mask_slice_id_coronal_bl, :], alpha=0.5, origin='lower')
                                    axs[1,3].set_title('Baseline prompt') 
                                    axs[1,3].axis('off')
                                except Exception as e:
                                    print(f'Error visualizing baseline prompt: {e}')

                            fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0.05, hspace=0.15)
                            plt.savefig(os.path.join(output_folder_or_file, f'{out_file}_dice_{dice_score:.2f}.png'), bbox_inches='tight')
                            plt.close()

                        
                        r.append(
                            export_pool.starmap_async(
                                export_prediction_from_logits,
                                ((prediction, properties, self.configuration_manager, self.plans_manager,
                                    self.dataset_json, out_file, False),)
                            )
                        )

                        # no multiprocessing
                        # export_prediction_from_logits(prediction, properties, self.configuration_manager, self.plans_manager,
                        #     self.dataset_json, out_file, False)
                    for metric_name in metrics.keys():
                        error_all[metric_name][patient_tp]['mean'] = np.mean(error_all[metric_name][patient_tp]['per_lesion'])
                print(f'done with {os.path.basename(ofile)}')
            error_all['dice']['mean']= np.mean(dice_score_all)
            error_all['hausdorff']['mean'] = np.mean(hausdorff_score_all)
            error_all['nsd']['mean'] = np.mean(nsd_score_all)
            error_all['lesion_found']['mean'] = (error_all['lesion_found']['all']/error_all['lesion_all']['all'])*100
            for tp in ['TP0','TP1','TP2']:
                error_all['dice'][tp]['mean']=np.mean(error_all['dice'][tp]['all'])
                error_all['hausdorff'][tp]['mean']=np.mean(error_all['hausdorff'][tp]['all'])
                error_all['nsd'][tp]['mean']=np.mean(error_all['nsd'][tp]['all'])
                error_all['lesion_found'][tp]['mean'] = (error_all['lesion_found'][tp]['all']/error_all['lesion_all'][tp]['all'])*100
            with open(os.path.join(output_folder_or_file, 'error_dict.json'), 'w') as fjson:
                json.dump(error_all, fjson)
            
            ret = [i.get()[0] for i in r]

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret


    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor, dense_prompt: torch.Tensor) -> torch.Tensor:
        """
        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
        SEE convert_predicted_logits_to_segmentation_with_correct_shape
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
        prediction = None

        # Add the dense prompt to the data
        data = torch.cat([data, dense_prompt], dim=0)

        for params in self.list_of_parameters:

            # messing with state dict names...
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)
        
            # why not leave prediction on device if perform_everything_on_device? Because this may cause the
            # second iteration to crash due to OOM. Grabbing that with try except cause way more bloated code than
            # this actually saves computation time
            if prediction is None:
                prediction = self.predict_sliding_window_return_logits(data, dense_prompt).to('cpu')
            else:
                prediction += self.predict_sliding_window_return_logits(data, dense_prompt).to('cpu')

        if len(self.list_of_parameters) > 1:
            prediction /= len(self.list_of_parameters)

        if self.verbose: print('Prediction done')
        torch.set_num_threads(n_threads)
        return prediction
    

    def mirror_and_predict(self, x0, x1, prompt):
        output = self.network_tracker(x0, x1, prompt, is_inference=True)
        prediction = output[0] if isinstance(output, tuple) else output
        reg_loss = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        
        total_reg_loss = reg_loss.all_loss.item() if reg_loss is not None else 0
        num_predictions = 1  # Count original prediction
        
        if reg_loss is not None:
            print('Registration Loss:', reg_loss.all_loss.item())

        if self.use_mirroring:
            mirror_axes = [2, 3, 4]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]

            for axes in axes_combinations:
                mirror_output = self.network_tracker(torch.flip(x0, axes), torch.flip(x1, axes), torch.flip(prompt, axes), is_inference=True)
                mirror_pred = mirror_output[0] if isinstance(mirror_output, tuple) else mirror_output
                mirror_reg_loss = mirror_output[1] if isinstance(mirror_output, tuple) and len(mirror_output) > 1 else None
                
                if mirror_reg_loss is not None:
                    mirror_loss = mirror_reg_loss.all_loss.item()
                    total_reg_loss += mirror_loss
                    num_predictions += 1
                
                prediction += torch.flip(mirror_pred, axes)
            
            prediction /= (len(axes_combinations) + 1)
            
            # Print average registration loss
            if num_predictions > 0:
                print('Average Registration Loss: {:.4f}'.format(total_reg_loss / num_predictions))
        
        prediction = prediction[0]
        return prediction

    @torch.inference_mode()
    def track_single_lesion(self, bl: torch.Tensor, fu: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=True) if self.device.type == 'cuda' else dummy_context():
            prediction = None
            for params in self.list_of_parameters_tracker: # fold iteration
                self.network_tracker.load_state_dict(params)
                self.network_tracker = self.network_tracker.to(self.device)
                self.network_tracker.eval()
                print('BL shape', bl.shape, bl.dtype)
                print('FU shape', fu.shape, fu.dtype)
                print('PROMPT shape',prompt.shape, prompt.dtype)
                if prediction is None:
                    prediction = self.mirror_and_predict(bl, fu, prompt).to('cpu')
                else:
                    prediction += self.mirror_and_predict(bl, fu, prompt).to('cpu')

            if len(self.list_of_parameters) > 1:
                prediction /= len(self.list_of_parameters)
            return prediction

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...], dense_prompt: torch.Tensor = None) -> List:
        slicers = []
        if len(self.configuration_manager.patch_size) < len(image_size):
            raise NotImplementedError('This predictor only supports 3D images')
        else:
            # No bbox will yield all slices
            if dense_prompt is None:
                steps = compute_steps_for_sliding_window(image_size, self.configuration_manager.patch_size,
                                                        self.tile_step_size)
                if self.verbose: print(
                    f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {self.configuration_manager.patch_size}, '
                    f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
                for sx in steps[0]:
                    for sy in steps[1]:
                        for sz in steps[2]:
                            slicers.append(
                                tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                    zip((sx, sy, sz), self.configuration_manager.patch_size)]]))
            else:
                prompt_coords = torch.where(dense_prompt[0] > 0)
                prompt_coords = [int(prompt_coords[i].min().item()) for i in range(3)] + [int(prompt_coords[i].max().item()) for i in range(3)]
                prompt_coords = torch.tensor(prompt_coords)
                # Bbox focused
                if all(prompt_coords[3:] - prompt_coords[:3] < torch.tensor(self.configuration_manager.patch_size)):
                    # Return a slicer that covers the bbox in the middle of the patch
                    slicer = [slice(None)]
                    for i in range(3):
                        start = int((prompt_coords[i] + prompt_coords[i + 3] - self.configuration_manager.patch_size[i]) / 2)
                        end = start + self.configuration_manager.patch_size[i]
                        if start < 0:
                            start = 0
                            end = self.configuration_manager.patch_size[i]
                        elif end > image_size[i]:
                            end = image_size[i]
                            start = end - self.configuration_manager.patch_size[i]
                        slicer.append(slice(start, end))
                    slicers.append(slicer)
                else: # Non bbox focused, return all slices which overlap with the bbox
                    steps = compute_steps_for_sliding_window(image_size, self.configuration_manager.patch_size,
                                                            self.tile_step_size)
                    if self.verbose: print(
                        f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {self.configuration_manager.patch_size}, '
                        f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
                    for sx in steps[0]:
                        for sy in steps[1]:
                            for sz in steps[2]:
                                slc = tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                        zip((sx, sy, sz), self.configuration_manager.patch_size)]])
                                # Make sure the slicer has some overlap with the bbox
                                if (prompt_coords[0] < slc[1].stop and prompt_coords[3] > slc[1].start) and \
                                        (prompt_coords[1] < slc[2].stop and prompt_coords[4] > slc[2].start) and \
                                        (prompt_coords[2] < slc[3].stop and prompt_coords[5] > slc[3].start):
                                    slicers.append(slc)
        return slicers

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        print('NETWORK INPUT SHAPE: ', x.shape)
        prediction = self.network(x)

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                prediction += torch.flip(self.network(torch.flip(x, axes)), axes)
            prediction /= (len(axes_combinations) + 1)
        return prediction

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        def producer(d, slh, q):
            for s in slh:
                q.put((torch.clone(d[s][None], memory_format=torch.contiguous_format).to(self.device), s))
            q.put('end')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)
            queue = Queue(maxsize=2)
            t = Thread(target=producer, args=(data, slicers, queue))
            t.start()

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')

            with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
                while True:
                    item = queue.get()
                    if item == 'end':
                        queue.task_done()
                        break
                    workon, sl = item
                    prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)
                    
                    if self.use_gaussian:
                        prediction *= gaussian
                    predicted_logits[sl] += prediction
                    n_predictions[sl[1:]] += gaussian
                    queue.task_done()
                    pbar.update()
            queue.join()

            # predicted_logits /= n_predictions
            torch.div(predicted_logits, n_predictions, out=predicted_logits)
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits

    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor, dense_prompt: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(input_image, torch.Tensor)
       
        self.network = self.network.to(self.device)
        self.network.eval()

        empty_cache(self.device)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
        # and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
        # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

            if self.verbose:
                print(f'Input shape: {input_image.shape}')
                print("step_size:", self.tile_step_size)
                print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                       'constant', {'value': 0}, True,
                                                       None)

            # Make sure we get only the patches we need to predict, i.e. overlab with the prompt
            slicers = self._internal_get_sliding_window_slicers(data.shape[1:], dense_prompt)

            if self.perform_everything_on_device and self.device != 'cpu':
                # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
                try:
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                           self.perform_everything_on_device)
                except RuntimeError:
                    print(
                        'Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                    empty_cache(self.device)
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
            else:
                predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                       self.perform_everything_on_device)

            empty_cache(self.device)
            # revert padding
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits


    def setup_training(self, learning_rate=1e-4, weight_decay=1e-5, use_scheduler=True, finetune_mode='all'):
        """
        Setup training components: optimizer, loss function, and scheduler.
        
        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            use_scheduler: Whether to use learning rate scheduler
            finetune_mode: Which part to finetune ('encoder', 'decoder', 'all')
        """
        if self.network is None:
            raise RuntimeError("Network not initialized. Call initialize_from_trained_model_folder first.")
        
        # Set network to training mode
        self.network.train()
        self.training_mode = True
        
        # Freeze/unfreeze parameters based on finetune_mode
        self._configure_trainable_parameters(finetune_mode)
        
        # Get trainable parameters for optimizer
        trainable_params = [p for p in self.network.parameters() if p.requires_grad]
        
        # Setup optimizer with only trainable parameters
        self.optimizer = optim.Adam(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup loss function (CrossEntropy + Dice loss)
        self.loss_function = self._combined_loss
        
        # Setup learning rate scheduler
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        else:
            self.scheduler = None
        
        # Count and print trainable parameters
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params_count = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        frozen_params_count = total_params - trainable_params_count
        
        print(f"Training setup complete. Mode: {finetune_mode}, LR: {learning_rate}, Device: {self.device}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params_count:,} ({trainable_params_count/1e6:.2f}M)")
        print(f"Frozen parameters: {frozen_params_count:,} ({frozen_params_count/1e6:.2f}M)")
        print(f"Trainable ratio: {100*trainable_params_count/total_params:.1f}%")

    def _configure_trainable_parameters(self, finetune_mode='all'):
        """
        Configure which parameters are trainable based on finetune mode.
        
        Args:
            finetune_mode: 'encoder', 'decoder', or 'all'
        """
        print(f"Configuring trainable parameters for mode: {finetune_mode}")
        
        # Print network architecture for inspection
        print("\n" + "="*80)
        print("NETWORK ARCHITECTURE:")
        print("="*80)
        for name, param in self.network.named_parameters():
            print(f"{name}: {param.shape}")
        print("="*80 + "\n")
        
        if finetune_mode == 'all':
            # Enable gradients for all parameters
            for param in self.network.parameters():
                param.requires_grad = True
            print("All parameters enabled for training")
            
        elif finetune_mode == 'encoder':
            # Freeze all parameters first
            for param in self.network.parameters():
                param.requires_grad = False
            
            # Enable encoder parameters (encoder.stem.* and encoder.stages.*)
            enabled_count = 0
            
            for name, param in self.network.named_parameters():
                # Check if parameter belongs to encoder
                if name.startswith('encoder.'):
                    param.requires_grad = True
                    enabled_count += 1
                    print(f"  Enabled: {name}")
            
            print(f"Encoder mode: {enabled_count} parameter groups enabled")
            
        elif finetune_mode == 'decoder':
            # Freeze all parameters first
            for param in self.network.parameters():
                param.requires_grad = False
            
            # Enable decoder parameters (decoder.stages.*, decoder.transpconvs.*, decoder.seg_layers.*)
            enabled_count = 0
            
            for name, param in self.network.named_parameters():
                # Check if parameter belongs to decoder
                if name.startswith('decoder.'):
                    param.requires_grad = True
                    enabled_count += 1
                    print(f"  Enabled: {name}")
            
            print(f"Decoder mode: {enabled_count} parameter groups enabled")
            
        else:
            raise ValueError(f"Unknown finetune_mode: {finetune_mode}. Use 'encoder', 'decoder', or 'all'")
        
        # Print summary of enabled/disabled parameters
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.network.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    def _combined_loss(self, predictions, targets):
        """Combined CrossEntropy + Dice loss for segmentation training."""
        # CrossEntropy loss
        ce_loss = nn.CrossEntropyLoss()(predictions, targets.long())
        
        # Dice loss (use external function)
        dice_loss_val = dice_loss(predictions, targets)
        
        return ce_loss + dice_loss_val

    def _visualize_validation_sample(self, data, target, prediction, filename, output_folder, epoch):
        """
        Visualize validation sample with overlaid masks in both axial and coronal views.
        Shows the cropped data that the network actually sees during training.
        
        Args:
            data: Input image [C, H, W, D] (cropped data seen by network)
            target: Ground truth mask [H, W, D] (cropped data seen by network)
            prediction: Predicted mask [H, W, D] (cropped data seen by network)
            filename: Sample filename
            output_folder: Output folder path
            epoch: Current epoch number
        """
        # Create epoch folder
        epoch_folder = os.path.join(output_folder, f'epoch_{epoch}')
        os.makedirs(epoch_folder, exist_ok=True)
        
        # Convert to numpy arrays
        if isinstance(data, torch.Tensor):
            data_np = data[0].cpu().numpy() if len(data.shape) > 3 else data.cpu().numpy()
        else:
            data_np = data[0] if len(data.shape) > 3 else data
            
        if isinstance(target, torch.Tensor):
            target_np = target.cpu().numpy()
        else:
            target_np = target
            
        if isinstance(prediction, torch.Tensor):
            pred_np = prediction.cpu().numpy()
        else:
            pred_np = prediction
        
        # Find the axial slice with the most target pixels (first dimension)
        target_sums_axial = np.sum(target_np, axis=(1, 2))  # Sum over each axial slice
        max_axial_slice = np.argmax(target_sums_axial) if np.max(target_sums_axial) > 0 else target_np.shape[0] // 2
        
        # Find the coronal slice with the most target pixels (second dimension)
        target_sums_coronal = np.sum(target_np, axis=(0, 2))  # Sum over each coronal slice
        max_coronal_slice = np.argmax(target_sums_coronal) if np.max(target_sums_coronal) > 0 else target_np.shape[1] // 2
        
        # Create visualization with 2 rows and 3 columns
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # First row - Axial view
        # Original image
        axes[0, 0].imshow(data_np[max_axial_slice, :, :], cmap='gray')
        axes[0, 0].set_title('Original Image (Axial)')
        axes[0, 0].axis('off')
        
        # Ground truth overlay
        axes[0, 1].imshow(data_np[max_axial_slice, :, :], cmap='gray')
        axes[0, 1].imshow(target_np[max_axial_slice, :, :], alpha=0.5, cmap='Reds')
        axes[0, 1].set_title('Ground Truth (Axial)')
        axes[0, 1].axis('off')
        
        # Prediction overlay
        axes[0, 2].imshow(data_np[max_axial_slice, :, :], cmap='gray')
        axes[0, 2].imshow(pred_np[max_axial_slice, :, :], alpha=0.5, cmap='Blues')
        axes[0, 2].set_title('Prediction (Axial)')
        axes[0, 2].axis('off')
        
        # Second row - Coronal view
        # Original image
        axes[1, 0].imshow(data_np[:, max_coronal_slice, :], cmap='gray', origin='lower')
        axes[1, 0].set_title('Original Image (Coronal)')
        axes[1, 0].axis('off')
        
        # Ground truth overlay
        axes[1, 1].imshow(data_np[:, max_coronal_slice, :], cmap='gray', origin='lower')
        axes[1, 1].imshow(target_np[:, max_coronal_slice, :], alpha=0.5, cmap='Reds', origin='lower')
        axes[1, 1].set_title('Ground Truth (Coronal)')
        axes[1, 1].axis('off')
        
        # Prediction overlay
        axes[1, 2].imshow(data_np[:, max_coronal_slice, :], cmap='gray', origin='lower')
        axes[1, 2].imshow(pred_np[:, max_coronal_slice, :], alpha=0.5, cmap='Blues', origin='lower')
        axes[1, 2].set_title('Prediction (Coronal)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(epoch_folder, f'{filename}_axial_{max_axial_slice}_coronal_{max_coronal_slice}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def train_cross_validation(self, train_input_files, train_prompt_files, train_output_files,
                              test_dataset=None, epochs=10, batch_size=2, lr=1e-4, 
                              device=None, output_folder=None, n_folds=5, num_workers=4, prompt_type='box',
                              ckpt_path=None, finetune_mode='all'):
        """
        Perform 5-fold cross-validation training.
        
        Args:
            train_input_files: Training input files
            train_prompt_files: Training prompt files  
            train_output_files: Training output files
            test_dataset: Test dataset for evaluation and visualization
            epochs: Number of epochs per fold
            batch_size: Batch size
            lr: Learning rate
            device: Training device
            output_folder: Output folder for checkpoints
            n_folds: Number of cross-validation folds
            num_workers: Number of preprocessing workers
            prompt_type: Type of prompt ('box', 'point', etc.)
        """
        if device is None:
            device = self.device
        
        print(f"Starting {n_folds}-fold cross-validation...")
        print(f"Total training samples: {len(train_input_files)}")
        
        # Create cross-validation folds
        folds = create_cv_folds(train_input_files, train_prompt_files, train_output_files, n_folds)
        
        # Store results from all folds
        all_fold_results = []
        
        for fold_idx, fold_data in enumerate(folds):
            print(f"\n{'='*50}")
            print(f"Starting Fold {fold_idx + 1}/{n_folds}")
            print(f"{'='*50}")
            
            # Reset network weights for each fold (reload from checkpoint)
            self.network.load_state_dict(self.list_of_parameters[0])
            
            # Train this fold
            fold_results = self.train_cv_fold(
                fold_data=fold_data,
                test_dataset=test_dataset,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
                output_folder=output_folder,
                fold_idx=fold_idx,
                num_workers=num_workers,
                prompt_type=prompt_type,
                ckpt_path=ckpt_path,
                finetune_mode=finetune_mode
            )
            
            all_fold_results.append(fold_results)
            
            print(f"Fold {fold_idx} completed!")
            print(f"Best validation loss: {fold_results['best_val_loss']:.4f}")
            if fold_results['test_dice_scores']:
                print(f"Final test dice: {fold_results['test_dice_scores'][-1]:.4f}")
        
        # Compute cross-validation statistics
        final_val_losses = [fold['val_losses'][-1] for fold in all_fold_results]
        best_val_losses = [fold['best_val_loss'] for fold in all_fold_results]
        
        if all_fold_results[0]['test_dice_scores']:
            final_test_dice = [fold['test_dice_scores'][-1] for fold in all_fold_results]
            print(f"\nCross-Validation Results:")
            print(f"Mean validation loss: {np.mean(final_val_losses):.4f} ± {np.std(final_val_losses):.4f}")
            print(f"Mean best validation loss: {np.mean(best_val_losses):.4f} ± {np.std(best_val_losses):.4f}")
            print(f"Mean test dice: {np.mean(final_test_dice):.4f} ± {np.std(final_test_dice):.4f}")
        
        # Save cross-validation summary
        if output_folder:
            cv_summary = {
                'n_folds': n_folds,
                'fold_results': all_fold_results,
                'mean_val_loss': np.mean(final_val_losses),
                'std_val_loss': np.std(final_val_losses),
                'mean_best_val_loss': np.mean(best_val_losses),
                'std_best_val_loss': np.std(best_val_losses)
            }
            
            if all_fold_results[0]['test_dice_scores']:
                cv_summary['mean_test_dice'] = np.mean(final_test_dice)
                cv_summary['std_test_dice'] = np.std(final_test_dice)
            
            with open(os.path.join(output_folder, 'cv_summary.json'), 'w') as f:
                json.dump(cv_summary, f, indent=2)
        
        return all_fold_results

    def train_cv_fold(self, fold_data, test_dataset=None, epochs=10, batch_size=2, lr=1e-4, 
                      device=None, output_folder=None, fold_idx=0, num_workers=4, prompt_type='box',
                      ckpt_path=None, finetune_mode='all'):
        """
        Training function for a single cross-validation fold.
        
        Args:
            fold_data: Dictionary containing train and val data for this fold
            test_dataset: Test dataset for dice score computation and visualization
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            device: Training device
            output_folder: Folder to save checkpoints
            fold_idx: Current fold index
            num_workers: Number of preprocessing workers
            prompt_type: Type of prompt ('box', 'point', etc.)
        """
        if device is None:
            device = self.device
            
        print(f"\n=== Training Fold {fold_idx} ===")
        
        # Create datasets for this fold
        train_dataset = self.create_training_dataset(
            input_files=fold_data['train']['input_files'],
            prompt_files=fold_data['train']['prompt_files'],
            output_files=fold_data['train']['output_files'],
            prompt_type=prompt_type,
            num_processes=3,
            verbose=False,
            track=False
        )
        
        val_dataset = self.create_training_dataset(
            input_files=fold_data['val']['input_files'],
            prompt_files=fold_data['val']['prompt_files'],
            output_files=fold_data['val']['output_files'],
            prompt_type=prompt_type,
            num_processes=3,
            verbose=False,
            track=False
        )
        
        # Setup training components
        self.setup_training(learning_rate=lr, finetune_mode=finetune_mode)
        self.network.to(device)
        
        # Create DataLoaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=training_collate_fn,
            num_workers=0
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=training_collate_fn,
            num_workers=0
        )
        
        test_dataloader = None
        if test_dataset is not None:
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                collate_fn=training_collate_fn,
                num_workers=0
            )
        
        # Training history for this fold
        fold_train_losses = []
        fold_val_losses = []
        fold_test_dice_scores = []
        best_val_loss = float('inf')
        
        print(f"Fold {fold_idx} - Training samples: {len(fold_data['train']['input_files'])}")
        print(f"Fold {fold_idx} - Validation samples: {len(fold_data['val']['input_files'])}")
        if test_dataset is not None:
            print(f"Test samples: {len(test_dataset.input_files)}")
        
        for epoch in range(epochs):
            # Training phase
            self.network.train()
            epoch_train_loss = 0.0
            num_train_batches = 0
            
            print(f"\nFold {fold_idx}, Epoch {epoch+1}/{epochs}")
            print("Training...")
            
            for batch_idx, batch in enumerate(train_dataloader):
                try:
                    data = batch['data'].to(device)
                    prompt = batch['prompt'].to(device)
                    target = batch['target'].to(device)
                    
                    if data.dim() == 4:
                        data = data.unsqueeze(0)
                        prompt = prompt.unsqueeze(0)
                        target = target.unsqueeze(0)
                    
                    combined_input = torch.cat([data, prompt], dim=1)
                    
                    self.optimizer.zero_grad()
                    outputs = self.network(combined_input)
                    loss = self.loss_function(outputs, target)
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_train_loss += loss.item()
                    num_train_batches += 1
                    
                    if batch_idx % 10 == 0:
                        print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
                        
                except Exception as e:
                    print(f"Error in training batch {batch_idx}: {e}")
                    continue
            
            avg_train_loss = epoch_train_loss / max(num_train_batches, 1)
            fold_train_losses.append(avg_train_loss)
            print(f"Training Loss: {avg_train_loss:.4f}")
            
            # Validation phase (for loss computation only)
            self.network.eval()
            epoch_val_loss = 0.0
            num_val_batches = 0
            
            print("Validating (loss computation on CV fold)...")
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    try:
                        data = batch['data'].to(device)
                        prompt = batch['prompt'].to(device)
                        target = batch['target'].to(device)
                        
                        if data.dim() == 4:
                            data = data.unsqueeze(0)
                            prompt = prompt.unsqueeze(0)
                            target = target.unsqueeze(0)
                        
                        combined_input = torch.cat([data, prompt], dim=1)
                        outputs = self.network(combined_input)
                        loss = self.loss_function(outputs, target)
                        
                        epoch_val_loss += loss.item()
                        num_val_batches += 1
                        
                    except Exception as e:
                        print(f"Error in validation batch {batch_idx}: {e}")
                        continue
            
            avg_val_loss = epoch_val_loss / max(num_val_batches, 1)
            fold_val_losses.append(avg_val_loss)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            
            # Test phase (for dice computation and visualization)
            if test_dataset is not None:
                print("Testing (dice computation and visualization on test data)...")
                epoch_test_dice_scores = []
                
                with torch.no_grad():
                    for batch_idx, batch in enumerate(test_dataloader):
                        try:
                            data = batch['data'].to(device)
                            prompt = batch['prompt'].to(device)
                            target = batch['target'].to(device)
                            filenames = batch['filename']
                            
                            if data.dim() == 4:
                                data = data.unsqueeze(0)
                                prompt = prompt.unsqueeze(0)
                                target = target.unsqueeze(0)
                                filenames = [filenames]
                            
                            combined_input = torch.cat([data, prompt], dim=1)
                            outputs = self.network(combined_input)
                            
                            # Process each sample for dice computation and visualization
                            for i in range(data.shape[0]):
                                filename = os.path.basename(filenames[i]).replace('.nii.gz', '')
                                
                                output_single = outputs[i:i+1]
                                pred_probs = torch.softmax(output_single, dim=1)
                                pred_classes = torch.argmax(pred_probs, dim=1).squeeze(0)
                                
                                data_single = data[i]
                                target_single = target[i]
                                
                                pred_cropped = pred_classes.cpu().numpy()
                                target_cropped = target_single.cpu().numpy()
                                
                                # Compute Dice score on test data
                                dice_score = compute_dice_coefficient(target_cropped, pred_cropped)
                                epoch_test_dice_scores.append(dice_score)
                                
                                # Visualize test samples (first few batches only)
                                if batch_idx < 1 and output_folder:
                                    test_viz_folder = os.path.join(output_folder, f'fold_{fold_idx}', 'test_visualizations')
                                    self._visualize_validation_sample(
                                        data_single, target_cropped, pred_cropped,
                                        f'{filename}_fold_{fold_idx}_epoch_{epoch}_batch_{batch_idx}_sample_{i}',
                                        test_viz_folder, epoch
                                    )
                            
                        except Exception as e:
                            print(f"Error in test batch {batch_idx}: {e}")
                            continue
                
                avg_test_dice = np.mean(epoch_test_dice_scores) if epoch_test_dice_scores else 0.0
                fold_test_dice_scores.append(avg_test_dice)
                print(f"Test Dice Score: {avg_test_dice:.4f}")
            
            # Update learning rate scheduler
            if self.scheduler:
                self.scheduler.step(avg_val_loss)
            
            # Save best model for this fold (based on validation loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if output_folder:
                    fold_folder = os.path.join(output_folder, f'fold_{fold_idx}')
                    self._save_checkpoint(fold_folder, 'best_model.pth', epoch, fold_idx=fold_idx, 
                                        ckpt_path=ckpt_path, prompt_type=prompt_type)
                    print(f"New best model saved for fold {fold_idx} (val_loss: {avg_val_loss:.4f})")
            
            # Save periodic checkpoint
            if output_folder and (epoch + 1) % 10 == 0:
                fold_folder = os.path.join(output_folder, f'fold_{fold_idx}')
                self._save_checkpoint(fold_folder, f'checkpoint_epoch_{epoch+1}.pth', epoch, fold_idx=fold_idx)
        
        # Save final checkpoint for this fold
        if output_folder:
            fold_folder = os.path.join(output_folder, f'fold_{fold_idx}')
            self._save_checkpoint(fold_folder, 'final_checkpoint.pth', epochs-1, fold_idx=fold_idx)
        
        return {
            'fold_idx': fold_idx,
            'train_losses': fold_train_losses,
            'val_losses': fold_val_losses,
            'test_dice_scores': fold_test_dice_scores,
            'best_val_loss': best_val_loss
        }

    def train(self, train_dataset, val_dataset=None, epochs=10, batch_size=2, lr=1e-4, 
              device=None, output_folder=None, num_workers=4, ckpt_path=None, prompt_type='point',
              finetune_mode='all'):
        """
        Training function that uses the existing multiprocessing data loading pipeline
        with PyTorch DataLoader integration and support for batching.
        
        Args:
            train_dataset: LesionDatasetWrapper for training data
            val_dataset: LesionDatasetWrapper for validation data (optional)
            epochs: Number of training epochs
            batch_size: Batch size (supports batch_size > 1 with fixed patch sizes from TrainingPreprocessor)
            lr: Learning rate
            device: Training device (uses self.device if None)
            output_folder: Folder to save checkpoints
            num_workers: Number of preprocessing workers (multiprocessing)
        """
        if device is None:
            device = self.device
            
        # Setup training components
        self.setup_training(learning_rate=lr, finetune_mode=finetune_mode)
        
        # Move network to device
        self.network.to(device)
        
        # Create DataLoaders with existing multiprocessing pipeline
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,  # Use actual batch_size parameter
            collate_fn=training_collate_fn,
            num_workers=0  # Use 0 since we handle multiprocessing internally
        )
        
        val_dataloader = None
        if val_dataset is not None:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,  # Use actual batch_size for validation too
                collate_fn=training_collate_fn,
                num_workers=0
            )
        
        # Training history
        train_losses = []
        val_losses = []
        val_dice_scores = []
        best_val_loss = float('inf')
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {device}, Learning rate: {lr}")
        print(f"Training samples: {len(train_dataset.input_files)}")
        if val_dataset is not None:
            print(f"Validation samples: {len(val_dataset.input_files)}")
        
        for epoch in range(epochs):
            # Training phase
            self.network.train()
            epoch_train_loss = 0.0
            num_train_batches = 0
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("Training...")
            
            for batch_idx, batch in enumerate(train_dataloader):
                # Skip batches to speed up training (process every 50th batch)
                    
                try:
                    # Extract data from batch
                    data = batch['data'].to(device)      # [B, C, H, W, D] or [C, H, W, D]
                    prompt = batch['prompt'].to(device)  # [B, 1, H, W, D] or [1, H, W, D]
                    target = batch['target'].to(device)  # [B, H, W, D] or [H, W, D]
                    print('Data shape ', data.shape)
                    print('Prompt shape ', prompt.shape)
                    print('Target shape ', target.shape)
                    # Handle both batched and single sample data
                    if data.dim() == 4:  # Single sample [C, H, W, D]
                        data = data.unsqueeze(0)      # [1, C, H, W, D]
                        prompt = prompt.unsqueeze(0)  # [1, 1, H, W, D]
                        target = target.unsqueeze(0)  # [1, H, W, D]
                    
                    # Combine image and prompt as input: [B, C+1, H, W, D]
                    combined_input = torch.cat([data, prompt], dim=1)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.network(combined_input)  # Network expects [B, C+1, H, W, D]
                    
                    # Calculate loss
                    loss = self.loss_function(outputs, target)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    # Update metrics
                    epoch_train_loss += loss.item()
                    num_train_batches += 1
                    
                    if batch_idx % 10 == 0:
                        batch_size_actual = data.shape[0]
                        print(f"  Batch {batch_idx} (size={batch_size_actual}), Loss: {loss.item():.4f}")
                        
                except Exception as e:
                    print(f"Error in training batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            avg_train_loss = epoch_train_loss / max(num_train_batches, 1)
            train_losses.append(avg_train_loss)
            print(f"Training Loss: {avg_train_loss:.4f}")
            
            # Validation phase
            if val_dataset is not None:
                self.network.eval()
                epoch_val_loss = 0.0
                epoch_val_dice_scores = []
                num_val_batches = 0
                
                print("Validating...")
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_dataloader):
                        try:
                            # Extract data from batch
                            data = batch['data'].to(device)      # [B, C, H, W, D] or [C, H, W, D]
                            prompt = batch['prompt'].to(device)  # [B, 1, H, W, D] or [1, H, W, D]
                            target = batch['target'].to(device)  # [B, H, W, D] or [H, W, D]
                            properties = batch['properties']     # Metadata
                            filenames = batch['filename']        # Filenames for visualization
                            
                            # Handle both batched and single sample data
                            if data.dim() == 4:  # Single sample [C, H, W, D]
                                data = data.unsqueeze(0)      # [1, C, H, W, D]
                                prompt = prompt.unsqueeze(0)  # [1, 1, H, W, D]
                                target = target.unsqueeze(0)  # [1, H, W, D]
                                properties = [properties]     # Make it a list
                                filenames = [filenames]       # Make it a list
                            
                            # Combine image and prompt as input: [B, C+1, H, W, D]
                            combined_input = torch.cat([data, prompt], dim=1)
                            outputs = self.network(combined_input)
                            
                            # Calculate loss on cropped data
                            loss = self.loss_function(outputs, target)
                            epoch_val_loss += loss.item()
                            num_val_batches += 1
                            
                            # Process each sample in the batch for dice computation and visualization
                            for i in range(data.shape[0]):
                                filename = os.path.basename(filenames[i]).replace('.nii.gz', '')
                                
                                # Get predictions (convert to class predictions)
                                output_single = outputs[i:i+1]  # Keep batch dimension
                                pred_probs = torch.softmax(output_single, dim=1)
                                pred_classes = torch.argmax(pred_probs, dim=1).squeeze(0)  # [H, W, D]
                                
                                # Get cropped data and target (what the network actually sees)
                                data_single = data[i]      # [C, H, W, D] - cropped
                                target_single = target[i]  # [H, W, D] - cropped
                                
                                # Convert to numpy for dice computation (use cropped data)
                                pred_cropped = pred_classes.cpu().numpy()
                                target_cropped = target_single.cpu().numpy()
                                
                                # Compute Dice score on cropped data (same as what model sees)
                                dice_score = compute_dice_coefficient(target_cropped, pred_cropped)
                                epoch_val_dice_scores.append(dice_score)
                                
                                # Visualize first few samples of each epoch using cropped data
                                if batch_idx < 1 and output_folder:  # Save first 3 validation samples
                                    filename = filenames[i] if isinstance(filenames, list) else f"sample_{i}"
                                    
                                    self._visualize_validation_sample(
                                        data_single, target_cropped, pred_cropped,
                                        f'{filename}_batch_{batch_idx}_sample_{i}',
                                        output_folder, epoch
                                    )
                            
                        except Exception as e:
                            print(f"Error in validation batch {batch_idx}: {e}")
                            traceback.print_exc()
                            continue
                
                # Compute averages
                avg_val_loss = epoch_val_loss / max(num_val_batches, 1)
                avg_val_dice = np.mean(epoch_val_dice_scores) if epoch_val_dice_scores else 0.0
                
                val_losses.append(avg_val_loss)
                val_dice_scores.append(avg_val_dice)
                print(f"Validation Loss: {avg_val_loss:.4f}")
                print(f"Validation Dice Score: {avg_val_dice:.4f}")
                
                # Update learning rate scheduler
                if self.scheduler:
                    self.scheduler.step(avg_val_loss)
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    if output_folder:
                        self._save_checkpoint(output_folder, 'best_model.pth', epoch, fold_idx=0, 
                                            ckpt_path=ckpt_path, prompt_type=prompt_type)
            
            # Save periodic checkpoint
            if output_folder and (epoch + 1) % 10 == 0:
                self._save_checkpoint(output_folder, f'checkpoint_epoch_{epoch+1}.pth', epoch, fold_idx=0)
        
        # Save final checkpoint
        if output_folder:
            self._save_checkpoint(output_folder, 'final_checkpoint.pth', epochs-1, fold_idx=0)
            
        print("Training completed!")
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_dice_scores': val_dice_scores,
            'best_val_loss': best_val_loss
        }

    def _save_checkpoint(self, output_folder, filename, epoch, fold_idx=None, ckpt_path=None, prompt_type='point'):
        """Save model checkpoint and optionally save inference-compatible checkpoint."""
        os.makedirs(output_folder, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'network_weights': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'trainer_name': self.trainer_name,
        }
        if self.scheduler:
            checkpoint['scheduler_state'] = self.scheduler.state_dict()
            
        checkpoint_path = os.path.join(output_folder, filename)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save inference-compatible checkpoint if ckpt_path is provided and this is a best model
        if ckpt_path and filename == 'best_model.pth' and fold_idx is not None:
            # Create inference-compatible directory structure
            optimized_folder = "point_optimized" if prompt_type == 'point' else "bbox_optimized"
            inference_dir = os.path.join(ckpt_path, 'LesionLocatorSeg', optimized_folder, f'fold_{fold_idx}')
            os.makedirs(inference_dir, exist_ok=True)
            
            # Get configuration name - this should match what's expected in the loading code
            # The loading code expects checkpoint['init_args']['configuration']
            config_name = getattr(self, 'configuration_name', None)
            if config_name is None:
                # Try to get it from the configuration manager or use a default
                config_name = 'default'  # This should be set to the actual configuration name used
                print(f"Warning: configuration_name not found, using default: {config_name}")
            
            # Create inference-compatible checkpoint with the same structure as the loaded checkpoints
            inference_checkpoint = {
                'network_weights': self.network.state_dict(),
                'trainer_name': self.trainer_name,
                'init_args': {
                    'configuration': config_name,
                },
                'inference_allowed_mirroring_axes': getattr(self, 'allowed_mirroring_axes', None),
            }
            
            inference_checkpoint_path = os.path.join(inference_dir, 'checkpoint_final.pth')
            torch.save(inference_checkpoint, inference_checkpoint_path)
            print(f"Inference-compatible checkpoint saved: {inference_checkpoint_path}")
            print(f"  - trainer_name: {self.trainer_name}")
            print(f"  - configuration: {config_name}")
            print(f"  - allowed_mirroring_axes: {getattr(self, 'allowed_mirroring_axes', None)}")

    def create_training_dataset(self, input_files, prompt_files, output_files, prompt_type,
                               num_processes=3, verbose=False, track=False):
        """
        Create a training dataset using the existing multiprocessing pipeline.
        
        Args:
            input_files: List of input image files
            prompt_files: List of prompt files (JSON or segmentation masks)
            output_files: List of output file paths
            prompt_type: Type of prompt ('point', 'box', etc.)
            num_processes: Number of preprocessing workers
            verbose: Verbose output
            track: Whether to include tracking data
            
        Returns:
            LesionDatasetWrapper instance
        """
        return LesionDatasetWrapper(
            input_files=input_files,
            prompt_files=prompt_files,
            output_files=output_files,
            prompt_type=prompt_type,
            plans_manager=self.plans_manager,
            dataset_json=self.dataset_json,
            configuration_manager=self.configuration_manager,
            num_processes=num_processes,
            pin_memory=self.device.type == 'cuda',
            verbose=verbose,
            track=track
        )


def train_from_prompt():
    import argparse
    parser = argparse.ArgumentParser(description='This function handles the LesionLocator single timepoint segmentation'
                                     'training using a point or 3D box prompt. Prompts can be the coordinates of a '
                                     'point or a 3D box as .json files or also (ground truth) instance segmentation maps.')
    parser.add_argument('-i', type=str, required=True,
                        help='Input image file or folder containing images to be predicted. File endings should be .nii.gz'
                        ' or specify another file_ending in the dataset.json file of the downloaded checkpoint.')
    parser.add_argument('-iv', type=str, required=False,
                        help='TEST image files or folder. Used for dice computation and visualization after each epoch in cross-validation. File endings should be .nii.gz')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If the folder does not exist it will be created. Cross-validation results and checkpoints'
                             'will be saved with fold-specific subfolders.')
    parser.add_argument('-p', type=str, required=True,
                        help='TRAINING prompt file or folder with prompts. Can contain .json files with a point or 3D box or instance'
                        'segmentation maps (.nii.gz). The file containing the prompt must have the same name as the image it belongs to.'
                        'If instance segmentation maps are used, they must be in the same shape as the input images. Binary masks '
                        'will be converted to instance segmentations.')
    parser.add_argument('-pv', type=str, required=False,
                        help='TEST prompt files or folder with test segmentation maps (.nii.gz). Used for dice computation and visualization after each epoch. The file containing the prompt must have the same name as the image it belongs to.'
                        'If instance segmentation maps are used, they must be in the same shape as the input images. Binary masks '
                        'will be converted to instance segmentations.')
    parser.add_argument('-t', type=str, required=True, choices=['point', 'box'], default='box',
                        help='Specify the type of prompt. Options are "point" or "box". Default: box')
    parser.add_argument('-m', type=str, required=True,
                        help='Folder of the LesionLocator model called "LesionLocatorCheckpoint"')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--continue_prediction', '--c', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2).")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')
    parser.add_argument('--visualize', action='store_true', required=False, default=False,
                        help='Set this flag to visualize the prediction. This will open a napari viewer. You may need to '
                        ' run python -m pip install "napari[all]" first to use this feature.')
    parser.add_argument('--track', action='store_true', required=False, default=False,
                        help='Set this flag to enable tracking. This will use the LesionLocatorTrack model to track lesions.')
    parser.add_argument('--modality', type=str, required=True, choices=['ct', 'pet'], default='ct', help="Use this to set the modality")
    parser.add_argument('--adaptive_mode', action='store_true', help='Enable selection between segmentation and tracking based on Dice/NSD scores.')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, required=False, default=1,
                        help='Number of training epochs. Default: 1')
    parser.add_argument('--lr', type=float, required=False, default=1e-4,
                        help='Learning rate for training. Default: 1e-4')
    parser.add_argument('--batch_size', type=int, required=False, default=3,
                        help='Batch size for training. Default: 3 (supports batching with fixed patch sizes from TrainingPreprocessor)')
    parser.add_argument('--num_workers', type=int, required=False, default=4,
                        help='Number of workers for data loading. Default: 4')
    parser.add_argument('--ckpt_path', type=str, required=False, default=None,
                        help='Path to save inference-compatible checkpoints. Will create LesionLocatorSeg/point_optimized/fold_X structure. Default: None (no inference checkpoints saved)')
    parser.add_argument('--finetune', type=str, required=False, default='all', choices=['encoder', 'decoder', 'all'],
                        help='Which part of the model to finetune. Options: encoder, decoder, all. Default: all')
    
    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using LesionLocator:\n"
        "Rokuss, M., Kirchhoff, Y., Akbal, S., Kovacs, B., Roy, S., Ulrich, C., ... & Maier-Hein, K. (2025).\n"
        "LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging. "
        "CVPR.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    if not isdir(args.o):
        print(args.o)
        maybe_mkdir_p(args.o)

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = LesionLocatorSegmenter(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                allow_tqdm=not args.disable_progress_bar,
                                verbose_preprocessing=args.verbose,
                                visualize=args.visualize,
                                track=args.track,
                                adaptive_mode=args.adaptive_mode)
    optimized_ckpt = "bbox_optimized" if args.t == 'box' else "point_optimized"
    checkpoint_folder = join(args.m, 'LesionLocatorSeg', optimized_ckpt)
    checkpoint_folder_track = join(args.m, 'LesionLocatorTrack')
    predictor.initialize_from_trained_model_folder(checkpoint_folder, checkpoint_folder_track, args.f, args.modality, "checkpoint_final.pth")
    
    # Training mode
    print("Starting training mode...")
    
    # Print checkpoint path information if provided
    if args.ckpt_path:
        optimized_folder = "point_optimized" if args.t == 'point' else "bbox_optimized"
        print(f"Inference-compatible checkpoints will be saved to:")
        print(f"  {args.ckpt_path}/LesionLocatorSeg/{optimized_folder}/fold_X/checkpoint_final.pth")
        print(f"  This structure is compatible with the inference code.")
    else:
        print("No inference checkpoint path specified (--ckpt_path). Only training checkpoints will be saved.")
    
    # Print fine-tuning mode information
    print(f"Fine-tuning mode: {args.finetune}")
    if args.finetune == 'encoder':
        print("  Only encoder parameters will be trained (decoder frozen)")
    elif args.finetune == 'decoder':
        print("  Only decoder parameters will be trained (encoder frozen)")
    elif args.finetune == 'all':
        print("  All model parameters will be trained")
    
    # Get training files
    if os.path.isdir(args.i):
        train_input_files = subfiles(args.i, suffix=predictor.dataset_json['file_ending'], join=True, sort=True)
    else:
        train_input_files = [args.i]
        
    if os.path.isdir(args.p):
        train_prompt_files = subfiles(args.p, suffix=predictor.dataset_json['file_ending'], join=True, sort=True)
    else:
        train_prompt_files = [args.p]
        
    # Create output file names for training
    train_output_files = [join(args.o, 'train_' + os.path.basename(i).replace(predictor.dataset_json['file_ending'], '')) for i in train_input_files]
    
    # Get TEST files (renamed from validation - these are your actual test data)
    test_input_files = None
    test_prompt_files = None
    test_output_files = None
    test_dataset = None
    
    if hasattr(args, 'iv') and args.iv:
        if os.path.isdir(args.iv):
            test_input_files = subfiles(args.iv, suffix=predictor.dataset_json['file_ending'], join=True, sort=True)
        else:
            test_input_files = [args.iv]
            
    if hasattr(args, 'pv') and args.pv:
        if os.path.isdir(args.pv):
            test_prompt_files = subfiles(args.pv, suffix=predictor.dataset_json['file_ending'], join=True, sort=True)
        else:
            test_prompt_files = [args.pv]
            
    if test_input_files and test_prompt_files:
        test_output_files = [join(args.o, 'test_' + os.path.basename(i).replace(predictor.dataset_json['file_ending'], '')) for i in test_input_files]
        
        # Create test dataset for dice computation and visualization
        test_dataset = predictor.create_training_dataset(
            input_files=test_input_files,
            prompt_files=test_prompt_files,
            output_files=test_output_files,
            prompt_type=args.t,
            num_processes=args.npp,
            verbose=args.verbose,
            track=args.track
        )
        print(f"Test dataset created with {len(test_input_files)} samples")
    
    print(f"Training dataset created with {len(train_input_files)} samples")
    
    # Start cross-validation training
    all_fold_results = predictor.train_cross_validation(
        train_input_files=train_input_files,
        train_prompt_files=train_prompt_files,
        train_output_files=train_output_files,
        test_dataset=test_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        output_folder=args.o,
        n_folds=5,
        num_workers=args.num_workers,
        prompt_type=args.t,
        ckpt_path=args.ckpt_path,
        finetune_mode=args.finetune
    )
    
    print("Cross-validation training completed!")
    
    # Print summary statistics from all folds
    if all_fold_results:
        # Get final metrics from each fold
        final_train_losses = [fold['train_losses'][-1] for fold in all_fold_results]
        final_val_losses = [fold['val_losses'][-1] for fold in all_fold_results]
        best_val_losses = [fold['best_val_loss'] for fold in all_fold_results]
        
        print(f"Mean final training loss across folds: {np.mean(final_train_losses):.4f} ± {np.std(final_train_losses):.4f}")
        print(f"Mean final validation loss across folds: {np.mean(final_val_losses):.4f} ± {np.std(final_val_losses):.4f}")
        print(f"Mean best validation loss across folds: {np.mean(best_val_losses):.4f} ± {np.std(best_val_losses):.4f}")
        
        if all_fold_results[0]['test_dice_scores']:
            final_test_dice = [fold['test_dice_scores'][-1] for fold in all_fold_results]
            print(f"Mean final test dice across folds: {np.mean(final_test_dice):.4f} ± {np.std(final_test_dice):.4f}")


if __name__ == "__main__":
    train_from_prompt()
