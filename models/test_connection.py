import boto3

boto3.setup_default_session(profile_name='cgmiquel')

def download_files(bucket_name, download_path, save_as=None):
    s3_client = boto3.client('s3')
    s3_client.download_file(bucket_name, download_path, save_as)

download_files("arn:aws:s3:::crystal-gazers", "s3://crystal-gazers/archive/ca-all/ca.wiki.test.tokens", "test.tokens")