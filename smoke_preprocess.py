import os
import boto3
import logging
import random
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """ Parse command line arguments """
    parser = ArgumentParser(description="Process S3 bucket and folder for data preparation.")
    parser.add_argument('--s3-bucket', type=str, required=True, help='S3 bucket name')
    parser.add_argument('--s3-folder', type=str, required=True, help='S3 folder path')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Ratio of data to be used for training')
    return parser.parse_args()

def fetch_s3_file_list(bucket, prefix):
    """ List all files in specified S3 bucket and prefix """
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [item['Key'] for item in response.get('Contents', []) if item['Key'] != prefix + '/']

def download_files_to_local(files, local_path, bucket):
    """ Download list of files from S3 to a local directory """
    os.makedirs(local_path, exist_ok=True)
    s3_client = boto3.client('s3')
    for file_path in files:
        local_file_path = os.path.join(local_path, os.path.basename(file_path))
        s3_client.download_file(bucket, file_path, local_file_path)

def main():
    args = parse_arguments()
    files = fetch_s3_file_list(args.s3_bucket, args.s3_folder)

    jpg_files = [f for f in files if f.endswith('.jpg')]
    paired_files = [(f, f.replace('.jpg', '.txt')) for f in jpg_files]

    random.shuffle(paired_files)
    total_files = len(paired_files)
    train_end = int(total_files * args.train_ratio)
    test_val_split = int((total_files - train_end) / 2) + train_end

    train_files = [file for pair in paired_files[:train_end] for file in pair]
    val_files = [file for pair in paired_files[train_end:test_val_split] for file in pair]
    test_files = [file for pair in paired_files[test_val_split:] for file in pair]

    download_files_to_local(train_files, '/opt/ml/processing/train', args.s3_bucket)
    download_files_to_local(val_files, '/opt/ml/processing/val', args.s3_bucket)
    download_files_to_local(test_files, '/opt/ml/processing/test', args.s3_bucket)

    logging.info(f"Training files: {len(train_files)}")
    logging.info(f"Validation files: {len(val_files)}")
    logging.info(f"Testing files: {len(test_files)}")

if __name__ == '__main__':
    main()