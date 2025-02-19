#!/bin/bash

# Set variables
REPO_URL="https://github.com/deflagg/ai_workbench.git"
PROJECT_DIR="/ai_workbench"

# Update package list
echo "Updating package list..."
sudo apt update

# Install required packages
echo "Installing Git and Python..."
sudo apt install -y git python3 python3-pip

# Create project directory if it doesn't exist
echo "Setting up project directory..."
mkdir -p $PROJECT_DIR

# Navigate to project directory
cd $PROJECT_DIR

# Clone the GitHub repository
if [ ! -d "$(basename $REPO_URL .git)" ]; then
    echo "Cloning repository..."
    git clone $REPO_URL
else
    echo "Repository already exists, pulling latest changes..."
    cd "$(basename $REPO_URL .git)"
    git pull
fi

# Navigate to the repository directory
cd "$(basename $REPO_URL .git)"

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
else
    echo "No requirements.txt found, skipping dependency installation."
fi

# Print success message
echo "Setup complete! You can now navigate to $PROJECT_DIR and start working."

