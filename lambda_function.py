import json
import boto3
from urllib.parse import urlparse

s3 = boto3.client('s3')

def lambda_handler(event, context):
    s3_uri = event['s3_uri']
    parsed_uri = urlparse(s3_uri)
    bucket = parsed_uri.netloc
    key_prefix = parsed_uri.path.lstrip('/')

    # Thresholds from pipeline
    thresholds = {
        'mAP': event['mAPThreshold'],
        'mAP50': event['mAP50Threshold'],
        'mAP75': event['mAP75Threshold'],
        'precision': event['precisionThreshold'],
        'recall': event['recallThreshold']
    }

    # Construct the full path to the metrics file
    full_key = f'{key_prefix}/metrics.json'

    # Download the metrics file
    local_path = '/tmp/metrics.json'
    s3.download_file(bucket, full_key, local_path)

    # Read and evaluate the metrics
    with open(local_path, 'r') as f:
        metrics = json.load(f)

    # Check each metric against its threshold
    meets_criteria = True
    for key, threshold in thresholds.items():
        if key in metrics and metrics[key] < threshold:
            meets_criteria = False
            break

    mAP = metrics['mAP']
    mAP50 = metrics['mAP50']
    mAP75 = metrics['mAP75']
    precision = metrics['precision']
    recall = metrics['recall']

    print(f"Metrics: {metrics}")
    return {'result':meets_criteria, 'mAP':mAP, 'mAP50':mAP50, 'mAP75':mAP75, 'precision':precision, 'recall':recall}