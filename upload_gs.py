import os
from google.cloud import storage
# from config import bucketName, localFolder, bucketFolder

# Google Cloud Storage
bucketName = 'vcm-data-playground'
bucketFolder = 'xml-raw/'

# Local Data
localFolder = 'D:/XML_Data/'

storage_client = storage.Client()
bucket = storage_client.get_bucket(bucketName)

def upload_files(bucketName):
    """Upload files to GCP bucket."""
    files = [f for f in os.listdir(localFolder) if f.endswith('.xml')]
    for file in files:
        localFile = localFolder + file
        print(localFile)
        blob = bucket.blob(bucketFolder + file)
        blob.upload_from_filename(localFile)
    return f'Uploaded {files} to "{bucketName}" bucket.'

if __name__ == '__main__':
    print(upload_files(bucketName))
