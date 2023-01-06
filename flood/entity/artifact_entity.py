from dataclasses import dataclass

# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    image_data_file_path: str
    masks_data_file_path: str
    metadata_file_path: str
    