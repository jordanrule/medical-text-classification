#!/bin/bash
# GCP Deployment Script for Medical Text Classification
# Run this script after setting your GCP_PROJECT_ID environment variable

set -e

# Configuration - Update these values
GCP_PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="us-central1"
BUCKET_NAME="${BUCKET_NAME:-$GCP_PROJECT_ID-medical-text-classification}"
IMAGE_NAME="gcr.io/$GCP_PROJECT_ID/medical-text-classifier"

echo "Deploying Medical Text Classification to GCP"
echo "Project: $GCP_PROJECT_ID"
echo "Region: $REGION"
echo "Bucket: $BUCKET_NAME"

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1 > /dev/null; then
    echo "Please authenticate with Google Cloud:"
    echo "gcloud auth login"
    exit 1
fi

# Set project
gcloud config set project $GCP_PROJECT_ID

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Create GCS bucket
echo "Creating GCS bucket..."
if ! gsutil ls -b gs://$BUCKET_NAME > /dev/null 2>&1; then
    gsutil mb -p $GCP_PROJECT_ID -l $REGION gs://$BUCKET_NAME
    echo "Created bucket: gs://$BUCKET_NAME"
else
    echo "Bucket already exists: gs://$BUCKET_NAME"
fi

# Upload data to GCS
echo "Uploading data to GCS..."
gsutil -m rsync -r data gs://$BUCKET_NAME/data/

# Build and push Docker image
echo "Building Docker image..."
gcloud builds submit --tag $IMAGE_NAME .

# Submit training job
echo "Submitting training job..."
JOB_NAME="medical-text-training-$(date +%Y%m%d-%H%M%S)"

gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name="Medical Text Classification Training" \
  --worker-pool-spec=machine-type=n1-standard-8,replica-count=1,container-image-uri=$IMAGE_NAME \
  --args="--gcp,--experiment-name,$JOB_NAME,--output-dir,gs://$BUCKET_NAME/outputs"

echo "Training job submitted successfully!"
echo "Job name: $JOB_NAME"
echo ""
echo "Monitor training at:"
echo "https://console.cloud.google.com/ai/custom-jobs?project=$GCP_PROJECT_ID"
echo ""
echo "View logs:"
echo "gcloud ai custom-jobs describe $JOB_NAME --region=$REGION"
echo ""
echo "Download results:"
echo "gsutil -m cp -r gs://$BUCKET_NAME/outputs ./results"
