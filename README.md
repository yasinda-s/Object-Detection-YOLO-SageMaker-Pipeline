# YOLO-Sagemaker-Pipeline-Smoke-Detection
 Creating a custom smoke detection YOLOv8model using SageMaker pipelines with following steps (shown in "Pipelines" UI):

 - Preprocessing (splitting data in S3 bucket into train, test).
 - Training (taking in user model configurations and training with yolov8x model and storing training results).
 - Evaluation (benchmark of model evaluation using lambda step in workflow).
 - Conditional model registry push (ability to push into model registry if approved by user).
 - Model weights (.tar.gz file or .pt file) both stored in S3 URI and model registry for versioning/R&D.

![image](https://github.com/yasinda-s/YOLO-Sagemaker-Pipeline-Smoke-Detection/assets/60426941/d1859d82-38de-4f47-91bd-d2f57c57d514)

