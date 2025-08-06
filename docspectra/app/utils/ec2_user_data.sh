#!/bin/bash
# EC2 User Data Script for automatic Marker setup
# This script runs when EC2 instance starts

# Log everything
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
echo "Starting EC2 User Data script..."

# Update system
apt-get update -y
apt-get install -y python3-pip python3-venv git awscli

# Set environment variables
export AWS_DEFAULT_REGION=us-east-1
export TORCH_DEVICE=cuda  # Change to 'cpu' if no GPU instance

# Create application user (optional, for security)
useradd -m -s /bin/bash appuser || true

# Create directories
mkdir -p /opt/marker_cache
mkdir -p /home/ubuntu/.cache
chown -R ubuntu:ubuntu /home/ubuntu/.cache
chown -R ubuntu:ubuntu /opt/marker_cache

# Function to download models from S3
download_models_from_s3() {
    echo "Downloading models from S3..."
    
    # Create cache directories
    sudo -u ubuntu mkdir -p /home/ubuntu/.cache/surya
    sudo -u ubuntu mkdir -p /home/ubuntu/.cache/texify
    sudo -u ubuntu mkdir -p /home/ubuntu/.cache/marker
    
    # Download models
    sudo -u ubuntu aws s3 sync s3://markerbucket69/surya/ /home/ubuntu/.cache/surya/ --region us-east-1 || echo "Surya models not found in S3"
    sudo -u ubuntu aws s3 sync s3://markerbucket69/texify/ /home/ubuntu/.cache/texify/ --region us-east-1 || echo "Texify models not found in S3"
    sudo -u ubuntu aws s3 sync s3://markerbucket69/marker/ /home/ubuntu/.cache/marker/ --region us-east-1 || echo "Marker models not found in S3"
    
    echo "Model download completed"
}

# Download models on startup
download_models_from_s3

# Create a systemd service for automatic model sync (optional)
cat > /etc/systemd/system/marker-model-sync.service << 'EOF'
[Unit]
Description=Marker Model S3 Sync
After=network.target

[Service]
Type=oneshot
User=ubuntu
ExecStart=/usr/bin/aws s3 sync s3://markerbucket69/surya/ /home/ubuntu/.cache/surya/ --region us-east-1
ExecStart=/usr/bin/aws s3 sync s3://markerbucket69/texify/ /home/ubuntu/.cache/texify/ --region us-east-1
ExecStart=/usr/bin/aws s3 sync s3://markerbucket69/marker/ /home/ubuntu/.cache/marker/ --region us-east-1

[Install]
WantedBy=multi-user.target
EOF

# Create timer for periodic sync (optional)
cat > /etc/systemd/system/marker-model-sync.timer << 'EOF'
[Unit]
Description=Sync Marker models from S3 daily
Requires=marker-model-sync.service

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Enable and start services
systemctl enable marker-model-sync.timer
systemctl start marker-model-sync.timer

echo "EC2 setup completed. Models will be synced from S3 on startup."

# Create a health check script
cat > /home/ubuntu/check_models.sh << 'EOF'
#!/bin/bash
echo "Checking model availability..."

directories=("$HOME/.cache/surya" "$HOME/.cache/texify" "$HOME/.cache/marker")

for dir in "${directories[@]}"; do
    if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        echo "✅ $dir - Available"
        echo "   Files: $(find "$dir" -type f | wc -l)"
        echo "   Size: $(du -sh "$dir" 2>/dev/null | cut -f1)"
    else
        echo "❌ $dir - Not available or empty"
    fi
done
EOF

chmod +x /home/ubuntu/check_models.sh
chown ubuntu:ubuntu /home/ubuntu/check_models.sh

echo "User data script completed successfully!"
