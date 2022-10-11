import io
import json
import boto3

# Setting hyperparameters
with open('config.json') as config:
    hyperparams = json.load(config)

boto3.setup_default_session(profile_name=hyperparams['AWS_PROFILE'])

# def download_files(bucket_name, download_path, save_as=None):
#     s3_client = boto3.client('s3')
#     s3_client.download_file(bucket_name, download_path, save_as)

def download_s3(bucket_name, path_to_file):
    '''Function to test connection with s3'''
    s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket=bucket_name, Key=path_to_file)
    data = io.BytesIO(obj["Body"].read())
    return data

#path_to_bucket = "s3://crystal-gazers/archive/ca-all/ca.wiki.test.tokens"
#download_files("arn:aws:s3:::crystal-gazers", path_to_bucket, "test.tokens")
df = download_s3(hyperparams['BUCKET_NAME'], hyperparams['TEST_ROOT'])
print(df)
