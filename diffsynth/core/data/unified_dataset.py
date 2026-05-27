from .operators import *
import torch, json, pandas, random


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        image_to_train_prob=0.2,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.image_to_train_prob = image_to_train_prob
        self.load_metadata(metadata_path)
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
        use_aux_video=False,
        num_overlap_frames=5,
        replace_first_frame_with_anchor=False,
    ):
        # Create a shared ImageCropAndResize instance with use_first_size=True
        # This ensures all videos in a batch use the same computed dimensions
        shared_frame_processor = ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor, use_first_size=True)
        
        # Choose video loader based on use_aux_video flag
        if use_aux_video:
            video_loader = LoadVideoAnchorContext(
                num_frames, time_division_factor, time_division_remainder,
                frame_processor=shared_frame_processor,
                num_overlap_frames=num_overlap_frames,
                replace_first_frame_with_anchor=replace_first_frame_with_anchor,
            )    
        else:
            video_loader = LoadVideo(
                num_frames, time_division_factor, time_division_remainder,
                frame_processor=shared_frame_processor,
            )
        
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> shared_frame_processor >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=shared_frame_processor,
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), video_loader),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            # Filter out rows with NaN values in critical columns
            for key in self.data_file_keys:
                if key in metadata.columns:
                    len_before = len(metadata)
                    metadata = metadata.dropna(subset=[key])
                    len_after = len(metadata)
                    if len_before != len_after:
                        print(f"Dropped {len_before - len_after} rows with NaN in column '{key}'.")
            
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        try:
            if self.load_from_cache:
                data = self.cached_data[data_id % len(self.cached_data)]
                data = self.cached_data_operator(data)
            else:
                data = self.data[data_id % len(self.data)].copy()
                for key in self.data_file_keys:
                    if key in data:
                        if key in self.special_operator_map:
                            result = self.special_operator_map[key](data[key])
                        elif key in self.data_file_keys:
                            result = self.main_data_operator(data[key])
                            
                        # Special handling for animate_pose_video: extract video from LoadVideoAnchorContext result
                        if key == 'animate_pose_video' and isinstance(result, dict) and 'video' in result:
                            data['animate_pose_video'] = result['video'] 
                            data['animate_pose_anchor'] = result.get('anchor', None)  
                        elif key == 'segmentation_masks':
                            # Handle both dict (from LoadVideoAnchorContext) and list (from LoadVideo)
                            if isinstance(result, dict) and 'video' in result:
                                data['segmentation_masks'] = result['video']
                            else:
                                data['segmentation_masks'] = result
                        
                        # Handle dict results from operators like LoadVideoAnchorContext
                        elif isinstance(result, dict):
                            # Merge all keys from result into data
                            for sub_key, sub_value in result.items():
                                data[sub_key] = sub_value
                            # Keep the original key pointing to the main content
                            # For 'video' key, keep result['video'] as data['video']
                            # Original key is already overwritten by the loop above if it exists in result
                        else:
                            data[key] = result
            return data
        except Exception as e:
            print(f"Error loading data {data_id}: {e}, trying next data point.")
            return self.__getitem__((data_id + 1) % len(self))

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
