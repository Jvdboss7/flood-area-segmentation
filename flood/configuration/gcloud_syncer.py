import os

class GCloudSync:

    def sync_folder_to_gcloud(self, gcp_bucket_url, foldername):

        command = f"gsutil cp -r {foldername} gs://{gcp_bucket_url}/"
        # command = f"gcloud storage cp {filepath}/{filename} gs://{gcp_bucket_url}/"
        os.system(command)

    def sync_folder_from_gcloud(self, gcp_bucket_url, filename, destination):

        command = f"gsutil cp gs://{gcp_bucket_url}/{filename} {destination}/{filename}"
        # command = f"gcloud storage cp gs://{gcp_bucket_url}/{filename} {destination}/{filename}"
        os.system(command)

    def sync_model_from_gcloud(self, gcp_bucket_url, foldername, destination):

        command = f"gsutil cp -r gs://{gcp_bucket_url}/{foldername} {destination}/"
        # command = f"gcloud storage cp gs://{gcp_bucket_url}/{filename} {destination}/{filename}"
        os.system(command)