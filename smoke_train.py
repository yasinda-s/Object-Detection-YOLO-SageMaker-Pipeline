import argparse
import os
import sys
import subprocess
import datetime
import shutil
import logging

subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])

import boto3
from ultralytics import YOLO

MODEL_DIR = '/opt/ml/model'
DATETIME_STRING = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
LOCAL_WEIGHTS_SAVE_DIR = f"cv_weights/{DATETIME_STRING}"
BEST_MODEL_PATH = os.path.join(LOCAL_WEIGHTS_SAVE_DIR, 'train/weights', 'best.pt')
SAGEMAKER_MODEL_PATH = os.path.join(MODEL_DIR, 'model.pt')
S3_FOLDER_NAME = f"yolov8smokeweights-{DATETIME_STRING}"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def upload_directory_to_s3(bucket, local_directory, s3_prefix):
    s3_client = boto3.client('s3')
    for root, dirs, files in os.walk(local_directory):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(s3_prefix, relative_path)

            try:
                s3_client.upload_file(local_path, bucket, s3_path)
                logging.info("YOLO weights + metrics + artifacts uploaded to S3.")
            except boto3.exceptions.S3UploadFailedError as e:
                logging.error(f"Failed to upload {local_path} to S3: {e}")
                return False
    return True

def train(args):
    try:
        logging.info("Model training begun")
        model = YOLO(args.model)
        model.train(
            data="smoke_config.yaml", 
            epochs=args.epochs, 
            batch=args.batch,
            patience=args.patience,
            optimizer=args.optimizer,
            lr0 = args.initial_learning_rate,
            lrf = args.final_learning_rate,
            project=LOCAL_WEIGHTS_SAVE_DIR
        )
        logging.info("Model training completed")

        if not os.path.exists(BEST_MODEL_PATH):
            raise Exception(f"Best model not found at {BEST_MODEL_PATH}")

        shutil.copyfile(BEST_MODEL_PATH, SAGEMAKER_MODEL_PATH)
        logging.info("Best model copied to SageMaker model directory.")

        upload_directory_to_s3('smoke-detection-model-registry', LOCAL_WEIGHTS_SAVE_DIR, S3_FOLDER_NAME)

    except Exception as e:
        logging.error(f"An error occurred during training or uploading: {e}")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default="yolov8n.yaml")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='auto')
    parser.add_argument('--initial_learning_rate', type=float, default=0.01)
    parser.add_argument('--final_learning_rate', type=float, default=0.01)
    
    args = parser.parse_args()

    try:
        train(args)
    except Exception as e:
        logging.error(f"Error in the training process: {e}")
        sys.exit(1)
