# YOLOv10-Sagemaker-Pipeline-Smoke-Detection

Creating a custom smoke detection YOLOv10model using end-to-end, configurable SageMaker pipelines with following steps.

 - Preprocessing (splitting data in S3 bucket into train, validation, & test).
 - Training (taking in user model configurations and training with yolov10x model and storing training results).
 - Evaluation (benchmark of model evaluation using lambda step in workflow).
 - Conditional model registry push (ability to push into model registry if model metrics surpass the benchmarks assigned by user).

Model weights (.tar.gz file or .pt file) both stored in S3 URI and model registry for versioning/R&D.

Completed pipeline looks as follows - 

![image](https://github.com/user-attachments/assets/39612fe9-1657-427d-a107-23f723f4d260)

Model Results - 

![val_batch2_pred](https://github.com/yasinda-s/Object-Detection-YOLO-SageMaker-Pipeline/assets/60426941/216c13fe-e162-4cc1-81df-e5c15dc706c7)
