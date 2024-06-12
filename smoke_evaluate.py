import json
import yaml
import tarfile
from pathlib import Path
import subprocess
import sys
import os
import boto3
import datetime

def install_packages():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        
def extract_model(tar_path, extract_to):
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    print(f"Extracted {tar_path} to {extract_to}")
        
def upload_directory_to_s3(directory, bucket, s3_folder):
    s3_client = boto3.client('s3')
    for root, dirs, files in os.walk(directory):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, directory)
            s3_path = os.path.join(s3_folder, relative_path)

            try:
                s3_client.upload_file(local_path, bucket, s3_path)
                print(f"Uploaded {local_path} to s3://{bucket}/{s3_path}")
            except NoCredentialsError:
                print("Credentials not available")
    
if __name__ == '__main__':
    
    install_packages()
    from ultralytics import YOLO
    
    model_tar_path = '/opt/ml/processing/model/model.tar.gz'
    extract_to = '/opt/ml/processing/model/'
    extract_model(model_tar_path, extract_to)

    model_path = extract_to + 'model.pt'
    print(f'Loading model from {model_path}...')

    model = YOLO(model_path)
    print("Model loaded!")
    
    eval_output_dir = '/opt/ml/processing/evaluation'
    os.makedirs(eval_output_dir, exist_ok=True)
    
    with open('/opt/ml/processing/input/code/data.yaml', 'w') as fp:
        data_conf = {
            'train': '/opt/ml/processing/input',
            'val': '/opt/ml/processing/input',
            'test': '/opt/ml/processing/input',
            'names': {
                '0': 'forklift',
                '1': 'pallet jack',
                '2': 'worker'
            }
        }
        yaml.dump(data_conf, fp)
        print(f'Updated data conf: {json.dumps(data_conf, indent=2)}')
        
    metrics = model.val(data="/opt/ml/processing/input/code/data.yaml", 
                        project=eval_output_dir, 
                        name="val-results")
    
    print("Metrics receieved!")

    metrics_dict = {
        'mAP': metrics.box.map,
        'mAP50': metrics.box.map50,
        'mAP75': metrics.box.map75,
        'mAP_list': metrics.box.maps.tolist(),
        'precision': metrics.box.mp,
        'recall': metrics.box.mr
    }
        
    metrics_json_path = os.path.join(eval_output_dir, 'metrics.json')
    
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_dict, f)
        
    print("Evaluation Complete.")
    
    print(f"[METRICS] mAP={metrics_dict['mAP']}, mAP50={metrics_dict['mAP50']}, mAP75={metrics_dict['mAP75']}, precision={metrics_dict['precision']}, recall={metrics_dict['recall']}")
