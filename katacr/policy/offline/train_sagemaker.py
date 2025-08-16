from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig


def start_training():
    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path="s3://clashroyaleshawn/tensorboard", ## location to save the tensorboard result in aws
        container_local_output_path="/opt/ml/output/tensorboard" ## Local dir in aws
    )

    estimator = PyTorch(
        entry_point="policy/offline/torch_train.py",
        source_dir="katacr",
        role="arn:aws:iam::061039778161:role/clash-royale-training-role",
        framework_version="2.4",
        py_version="py311",
        instance_count=1,
        instance_type="ml.g5.xlarge",
        output_path="s3://clashroyaleshawn/model_output",
        hyperparameters={
            "batch-size": 32,
            "epochs": 60,
            "data-path": "/opt/ml/input/data/training"
        },
        tensorboard_config=tensorboard_config
    )

    # Start training
    estimator.fit({
        "training": "s3://clashroyaleshawn/offline_preprocess/train",
        "validation": "s3://clashroyaleshawn/offline_preprocess/validation"
    })


if __name__ == '__main__':
  start_training()