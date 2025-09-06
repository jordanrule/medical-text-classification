"""
Google Cloud Platform Configuration for Medical Text Classification Training

This module provides configuration and utilities for training on GCP.
"""

import os
from typing import Dict, List


class GCPConfig:
    """
    Configuration for GCP training deployment.
    """

    def __init__(self,
                 project_id: str,
                 region: str = "us-central1",
                 bucket_name: str = None):
        """
        Initialize GCP configuration.

        Args:
            project_id: Google Cloud project ID
            region: GCP region for training
            bucket_name: GCS bucket for storing data and models
        """
        self.project_id = project_id
        self.region = region
        self.bucket_name = bucket_name or f"{project_id}-medical-text-classification"

    def get_training_config(self) -> Dict:
        """
        Get Vertex AI training configuration.
        """
        return {
            "project": self.project_id,
            "region": self.region,
            "staging_bucket": f"gs://{self.bucket_name}",
            "display_name": "medical-text-classification-training",
            "worker_pool_specs": [
                {
                    "machine_spec": {
                        "machine_type": "n1-standard-8",
                        "accelerator_type": "NVIDIA_TESLA_T4",
                        "accelerator_count": 1,
                    },
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": f"gcr.io/{self.project_id}/medical-text-classifier:latest",
                        "command": ["python", "train.py"],
                        "args": ["--gcp", "--output-dir", "/app/outputs"],
                    },
                }
            ],
        }

    def get_container_config(self) -> Dict:
        """
        Get container build configuration.
        """
        return {
            "image_name": f"gcr.io/{self.project_id}/medical-text-classifier",
            "dockerfile_path": "Dockerfile",
            "context_path": ".",
            "build_args": {},
        }

    def get_hyperparameter_tuning_config(self) -> Dict:
        """
        Get hyperparameter tuning configuration.
        """
        return {
            "display_name": "medical-text-hpt",
            "trial_job_spec": self.get_training_config(),
            "study_spec": {
                "metrics": [
                    {
                        "metric_id": "validation_f1",
                        "goal": "MAXIMIZE",
                    }
                ],
                "parameters": [
                    {
                        "parameter_id": "learning_rate",
                        "discrete_value_spec": {"values": [0.001, 0.0001, 0.00001]},
                    },
                    {
                        "parameter_id": "batch_size",
                        "discrete_value_spec": {"values": [16, 32, 64]},
                    },
                    {
                        "parameter_id": "embedding_dim",
                        "discrete_value_spec": {"values": [384, 768, 1024]},
                    },
                ],
                "max_trial_count": 20,
                "parallel_trial_count": 3,
            },
        }


def create_gcp_deployment_script(config: GCPConfig) -> str:
    """
    Create GCP deployment script.
    """
    script = f'''#!/bin/bash
# GCP Deployment Script for Medical Text Classification

set -e

PROJECT_ID="{config.project_id}"
REGION="{config.region}"
BUCKET_NAME="{config.bucket_name}"
IMAGE_NAME="gcr.io/$PROJECT_ID/medical-text-classifier"

echo "Deploying Medical Text Classification to GCP"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Bucket: $BUCKET_NAME"

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com

# Create GCS bucket
echo "Creating GCS bucket..."
gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME

# Build and push Docker image
echo "Building Docker image..."
gcloud builds submit --tag $IMAGE_NAME .

# Upload data to GCS
echo "Uploading data to GCS..."
gsutil -m cp -r data/* gs://$BUCKET_NAME/data/

# Submit training job
echo "Submitting training job..."
gcloud ai custom-jobs create \\
  --region=$REGION \\
  --display-name=medical-text-classification-training \\
  --worker-pool-spec=machine-type=n1-standard-8,replica-count=1,container-image-uri=$IMAGE_NAME \\
  --args="--train-data,/app/data/medical_tc_train.csv,--test-data,/app/data/medical_tc_test.csv,--output-dir,/app/outputs,--gcp"

echo "Training job submitted successfully!"
echo "Monitor at: https://console.cloud.google.com/ai/custom-jobs?project=$PROJECT_ID"
'''

    return script


def create_vertex_ai_pipeline_config(config: GCPConfig) -> Dict:
    """
    Create Vertex AI Pipeline configuration for automated training.
    """
    return {
        "display_name": "medical-text-training-pipeline",
        "pipeline_spec": {
            "components": {
                "data_preparation": {
                    "executor_label": "data_prep",
                    "input_definitions": {
                        "parameters": {
                            "input_data": {"type": "STRING"},
                            "output_data": {"type": "STRING"},
                        }
                    }
                },
                "model_training": {
                    "executor_label": "trainer",
                    "input_definitions": {
                        "parameters": {
                            "training_data": {"type": "STRING"},
                            "validation_data": {"type": "STRING"},
                            "model_config": {"type": "STRING"},
                        }
                    }
                },
                "model_evaluation": {
                    "executor_label": "evaluator",
                    "input_definitions": {
                        "parameters": {
                            "model": {"type": "MODEL"},
                            "test_data": {"type": "STRING"},
                        }
                    }
                },
            },
            "deployment_spec": {
                "executors": {
                    "data_prep": {
                        "container": {
                            "image": f"gcr.io/{config.project_id}/medical-text-classifier",
                            "command": ["python", "prepare_data.py"],
                        }
                    },
                    "trainer": {
                        "container": {
                            "image": f"gcr.io/{config.project_id}/medical-text-classifier",
                            "command": ["python", "train.py"],
                        }
                    },
                    "evaluator": {
                        "container": {
                            "image": f"gcr.io/{config.project_id}/medical-text-classifier",
                            "command": ["python", "evaluate.py"],
                        }
                    },
                }
            },
        },
    }


def get_gcp_cost_estimate(config: GCPConfig, training_hours: float = 2.0) -> Dict:
    """
    Estimate GCP training costs.
    """
    # Rough cost estimates (as of 2024, prices may vary)
    costs = {
        "n1-standard-8": 0.38,  # per hour
        "nvidia-tesla-t4": 0.35,  # per hour
        "storage": 0.026,  # per GB per month
        "data_transfer": 0.12,  # per GB
    }

    compute_cost = training_hours * (costs["n1-standard-8"] + costs["nvidia-tesla-t4"])
    storage_cost = 10 * costs["storage"]  # Assuming 10GB storage
    data_cost = 5 * costs["data_transfer"]  # Assuming 5GB data transfer

    total_cost = compute_cost + storage_cost + data_cost

    return {
        "compute_cost": compute_cost,
        "storage_cost": storage_cost,
        "data_cost": data_cost,
        "total_estimated_cost": total_cost,
        "currency": "USD",
        "notes": "Estimates are approximate and may vary based on actual usage"
    }


# Example usage and configuration
def create_example_config():
    """
    Create example GCP configuration.
    """
    config = GCPConfig(
        project_id="your-project-id",
        region="us-central1",
        bucket_name="your-bucket-name"
    )

    print("GCP Configuration Created:")
    print(f"Project ID: {config.project_id}")
    print(f"Region: {config.region}")
    print(f"Bucket: {config.bucket_name}")

    # Generate deployment script
    deployment_script = create_gcp_deployment_script(config)

    with open("deploy_gcp.sh", "w") as f:
        f.write(deployment_script)

    print("Deployment script saved to: deploy_gcp.sh")
    print("Make it executable with: chmod +x deploy_gcp.sh")

    return config


if __name__ == "__main__":
    # This would typically be called with actual project details
    print("GCP Configuration Module")
    print("To use this module:")
    print("1. Set your Google Cloud Project ID")
    print("2. Run: python gcp_config.py")
    print("3. Execute: ./deploy_gcp.sh")
