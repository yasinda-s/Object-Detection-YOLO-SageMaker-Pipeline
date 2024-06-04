import os
import sys
import random
import boto3
import logging
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--s3-bucket', type=str, required=True)
    parser.add_argument('--s3-folder', type=str, required=True)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    return parser.parse_args()

def list_files(bucket, prefix):
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [item['Key'] for item in response['Contents'] if item['Key'] != prefix + '/']

def save_files_locally(file_list, local_path, args):
    os.makedirs(local_path, exist_ok=True)
    s3_client = boto3.client('s3')
    for pair in file_list:
        for file in pair:
            local_file_path = os.path.join(local_path, os.path.basename(file))
            s3_client.download_file(args.s3_bucket, file, local_file_path)

def main():
    args = parse_args()

    files = list_files(args.s3_bucket, args.s3_folder)
    paired_files = [(f, f.replace('.jpg', '.txt')) for f in files if f.endswith('.jpg')]
    unpaired_txt_files = [f for f in files if f.endswith('.txt') and f.replace('.txt', '.jpg') not in files]

    random.shuffle(paired_files)
    split_index = int(len(paired_files) * args.train_ratio)
    train_files = paired_files[:split_index] + [(txt,) for txt in unpaired_txt_files]
    test_files = paired_files[split_index:]

    save_files_locally(train_files, '/opt/ml/processing/train', args)
    save_files_locally(test_files, '/opt/ml/processing/test', args)

    logging.info(f"Training files: {len(train_files)}")
    logging.info(f"Testing files: {len(test_files)}")

if __name__ == '__main__':
    main()
