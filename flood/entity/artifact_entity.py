from dataclasses import dataclass

# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    root_dir: str
    image_data_file_path: str
    masks_data_file_path: str
    metadata_file_path: str

@dataclass
class DataTransformationArtifacts:
    train_data_path: str
    test_data_path: str 