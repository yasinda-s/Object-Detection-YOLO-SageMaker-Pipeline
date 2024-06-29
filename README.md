# YOLOv10-Sagemaker-Pipeline-Smoke-Detection
 Creating a custom smoke detection YOLOv8model using SageMaker pipelines with following steps (shown in "Pipelines" UI):

 - Preprocessing (splitting data in S3 bucket into train, validation, & test).
 - Training (taking in user model configurations and training with yolov10x model and storing training results).
 - Evaluation (benchmark of model evaluation using lambda step in workflow).
 - Conditional model registry push (ability to push into model registry if approved by user).
 - Model weights (.tar.gz file or .pt file) both stored in S3 URI and model registry for versioning/R&D.

Completed pipeline looks as follows - 

![image](https://github.com/yasinda-s/YOLO-Sagemaker-Pipeline-Smoke-Detection/assets/60426941/d1859d82-38de-4f47-91bd-d2f57c57d514)


Model Results - 

![val_batch2_pred](https://github.com/yasinda-s/Object-Detection-YOLO-SageMaker-Pipeline/assets/60426941/216c13fe-e162-4cc1-81df-e5c15dc706c7)

