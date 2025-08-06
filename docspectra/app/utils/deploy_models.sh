#!/bin/bash
# deploy_models.sh - Complete deployment script

set -e  # Exit on any error

echo "🚀 Starting Marker models deployment to S3..."

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS CLI is not configured. Please run 'aws configure' first."
    exit 1
fi

# Check if bucket exists
if ! aws s3 ls s3://markerbucket69 &> /dev/null; then
    echo "❌ S3 bucket 'markerbucket69' does not exist or is not accessible."
    exit 1
fi

echo "✅ AWS CLI configured and bucket accessible"

# Activate virtual environment if it exists
if [ -d "marker-env" ]; then
    echo "📦 Activating marker-env virtual environment..."
    source marker-env/bin/activate
fi

# Install/upgrade required packages
echo "📦 Installing required packages..."
pip install --upgrade marker-pdf[full] boto3

# Set environment variables
export AWS_DEFAULT_REGION=us-east-1
export TORCH_DEVICE=cuda  # Change to 'cpu' if no GPU

# Run model setup
echo "🔧 Setting up models..."
python setup_models.py

# Verify S3 contents
echo "📋 Verifying S3 bucket contents..."
aws s3 ls s3://markerbucket69/ --recursive --human-readable --summarize

echo "✅ Deployment completed successfully!"
echo ""
echo "📌 Next steps:"
echo "1. Your models are now cached in S3 bucket 'markerbucket69'"
echo "2. EC2 instances will automatically download models from S3 on first run"
echo "3. Test your setup by running: python -c 'from app.utils.marker_handler import MarkerHandler; h = MarkerHandler(); print(h.health_check())'"
