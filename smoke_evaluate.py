import json
import yaml
import tarfile
from pathlib import Path
import subprocess
import sys
import os
import boto3
import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def install_packages():
    """ Install necessary Python packages. This step can be avoided by providing a docker image with ultralytics installed. """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        logging.info("Packages installed successfully.")
    except subprocess.CalledProcessError:
        logging.error("Failed to install packages.")
        sys.exit(1)

def log_directory_contents(path):
    try:
        entries = os.listdir(path)
        logging.info(f"Contents of {path}: {entries}")
    except Exception as e:
        logging.error(f"Failed to list contents of {path}: {e}")

def extract_model(tar_path, extract_to):
    """ Extract model tar.gz file to a specified directory. """
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
        logging.info(f"Extracted {tar_path} to {extract_to}")
    except tarfile.TarError:
        logging.error(f"Failed to extract {tar_path}")
        sys.exit(1)

def upload_directory_to_s3(directory, bucket, s3_folder):
    """ Upload a directory to an S3 bucket. """
    s3_client = boto3.client('s3')
    for root, dirs, files in os.walk(directory):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, directory)
            s3_path = os.path.join(s3_folder, relative_path)

            try:
                s3_client.upload_file(local_path, bucket, s3_path)
                logging.info(f"Uploaded {local_path} to s3://{bucket}/{s3_path}")
            except boto3.exceptions.S3UploadFailedError as e:
                logging.error(f"Failed to upload {local_path} to S3: {e}")
                sys.exit(1)

def configure_and_run_evaluation():
    """ Configures and runs the model evaluation. """
    from ultralytics import YOLO  # Importing here after installation
    
    model_tar_path = '/opt/ml/processing/model/model.tar.gz'
    extract_to = '/opt/ml/processing/model/'
    extract_model(model_tar_path, extract_to)

    model_path = Path(extract_to) / 'model.pt'
    logging.info(f'Loading model from {model_path}...')

    model = YOLO(str(model_path))
    logging.info("Model loaded successfully.")
    
    eval_output_dir = '/opt/ml/processing/evaluation'
    os.makedirs(eval_output_dir, exist_ok=True)
    
    data_config_path = '/opt/ml/processing/input/code/data.yaml'
    with open(data_config_path, 'w') as fp:
        data_conf = {
            'train': '/opt/ml/processing/input',
            'val': '/opt/ml/processing/input',
            'test': '/opt/ml/processing/input',
            'names': ['smoke']
        }
        yaml.dump(data_conf, fp)
        logging.info(f'Updated data configuration: {json.dumps(data_conf, indent=2)}')
        
    metrics = model.val(data=str(data_config_path), project=eval_output_dir, name="val-results", split="test")
    logging.info("Metrics received.")
    
    metrics_dict = {
        'mAP': metrics.box.map,
        'mAP50': metrics.box.map50,
        'mAP75': metrics.box.map75,
        'mAP_list': metrics.box.maps.tolist(),
        'precision': metrics.box.mp,
        'recall': metrics.box.mr
    }
    
    metrics_json_path = Path(eval_output_dir) / 'metrics.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_dict, f)
        
    logging.info(f"Evaluation complete with metrics: {metrics_dict}")

if __name__ == '__main__':
    install_packages()
    configure_and_run_evaluation()

