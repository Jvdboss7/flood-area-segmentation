from dataclasses import dataclass

# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    root_dir: str
    image_data_file_path: str
    masks_data_file_path: str
    metadata_file_path: str

# Model trainer artifacts    
@dataclass 
class ModelTrainerArtifacts:
    trained_model_path: str
    test_dataset: 'tensorflow.python.data.ops.dataset_ops.PrefetchDataset'

# Model evaluation artifacts
@dataclass
class ModelEvaluationArtifacts:
    is_model_accepted: bool    

@dataclass
class ModelPusherArtifacts:
    bucket_name: str