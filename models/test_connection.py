import boto3
import io
import pandas as pd

# boto3.setup_default_session(profile_name='cgmiquel')

# def download_files(bucket_name, download_path, save_as=None):
#     s3_client = boto3.client('s3')
#     s3_client.download_file(bucket_name, download_path, save_as)

def download_s3(bucket_name, path_to_file):
    s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket=bucket_name, Key=path_to_file)
    return obj

#download_files("arn:aws:s3:::crystal-gazers", "s3://crystal-gazers/archive/ca-all/ca.wiki.test.tokens", "test.tokens")
df = download_s3("crystal-gazers", "archive/ca-all/ca.wiki.test.tokens")
print(df)