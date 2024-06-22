import argparse
import os
import sys
import subprocess
import datetime
import shutil
import logging
import boto3
import json

MODEL_DIR = '/opt/ml/model'
DATETIME_STRING = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
LOCAL_WEIGHTS_SAVE_DIR = f"cv_weights/{DATETIME_STRING}"
BEST_MODEL_PATH = os.path.join(LOCAL_WEIGHTS_SAVE_DIR, 'train/weights', 'best.pt')
SAGEMAKER_MODEL_PATH = os.path.join(MODEL_DIR, 'model.pt')
S3_FOLDER_NAME = f"yolov8smokeweights-{DATETIME_STRING}"
BUCKET_NAME = 'smoke-detection-model-registry'
eval_output_dir = '/opt/ml/processing/evaluation'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def install_packages():
    """ Install necessary Python packages. """
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])

def upload_directory_to_s3(local_directory, s3_prefix):
    """ Uploads a directory to an S3 bucket. """
    s3_client = boto3.client('s3')
    for root, dirs, files in os.walk(local_directory):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(s3_prefix, relative_path)

            try:
                s3_client.upload_file(local_path, BUCKET_NAME, s3_path)
                logging.info(f"Uploaded {filename} to S3 at {s3_path}")
            except Exception as e:
                logging.error(f"Failed to upload {local_path} to S3: {e}")
                return False
    return True

def upload_file_to_s3(file_path, bucket, s3_key):
    """ Uploads a single file to an S3 bucket. """
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_path, bucket, s3_key)
        logging.info(f"Successfully uploaded {file_path} to s3://{bucket}/{s3_key}")
    except boto3.exceptions.S3UploadFailedError as e:
        logging.error(f"Failed to upload {file_path} to S3: {e}")
        raise

def train_and_evaluate(args):
    try:
        logging.info("Model training started.")
        from ultralytics import YOLO  # Importing here to ensure packages are installed first.
        model = YOLO(args.model)
        model.train(
            data="smoke_config.yaml", 
            epochs=args.epochs, 
            batch=args.batch,
            patience=args.patience,
            optimizer=args.optimizer,
            lr0=args.initial_learning_rate,
            lrf=args.final_learning_rate,
            project=LOCAL_WEIGHTS_SAVE_DIR
        )
        logging.info("Model training completed.")

        if not os.path.exists(BEST_MODEL_PATH):
            logging.error(f"Best model not found at {BEST_MODEL_PATH}")
            return

        shutil.copyfile(BEST_MODEL_PATH, SAGEMAKER_MODEL_PATH)
        logging.info("Best model copied to SageMaker model directory.")

        if not upload_directory_to_s3(LOCAL_WEIGHTS_SAVE_DIR, S3_FOLDER_NAME):
            logging.error("Failed to upload some files to S3.")

        if args.evaluate:
            logging.info("Starting evaluation on test data")
            eval_metrics = model.val(data="smoke_config.yaml", project=eval_output_dir, name="val-results", split='test')
            metrics_dict = {
                'mAP': eval_metrics.box.map,
                'mAP50': eval_metrics.box.map50,
                'mAP75': eval_metrics.box.map75,
                'precision': eval_metrics.box.mp,
                'recall': eval_metrics.box.mr
            }
            metrics_json_path = os.path.join(eval_output_dir, 'metrics.json')
            with open(metrics_json_path, 'w') as f:
                json.dump(metrics_dict, f)
            logging.info(f"Saved evaluation metrics to {metrics_json_path}")
        
            upload_directory_to_s3(eval_output_dir, 'smoke-detection-eval-metrics', args.folder_name)
            upload_file_to_s3(metrics_json_path, 'smoke-detection-eval-metrics', args.folder_name)
    except Exception as e:
        logging.error(f"Error during training and evaluation: {e}")
        sys.exit(1)

def main():
    """ Main function to handle workflow logic. """
    install_packages()
    
    parser = argparse.ArgumentParser(description="Train a YOLO model for smoke detection.")
    parser.add_argument('--model', type=str, default="yolov8n.yaml")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='auto')
    parser.add_argument('--initial_learning_rate', type=float, default=0.01)
    parser.add_argument('--final_learning_rate', type=float, default=0.01)
    parser.add_argument('--folder_name', type=str, required=True, help="Folder name for storing results in S3.")
    
    args = parser.parse_args()
    train_and_evaluate(args)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(f"Unhandled error: {e}")
        sys.exit(1)
