#!/bin/bash

# GitHub Repository Setup Script for Fieldwork Forecasting
# This script helps set up the repository on berkeley-yeast GitHub account

echo "Setting up Fieldwork Forecasting Repository"
echo "=========================================="

# Repository details
GITHUB_USERNAME="berkeley-yeast"
REPO_NAME="fieldwork_forecast"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

# Check if GitHub token is provided
if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GITHUB_TOKEN environment variable is not set!"
    echo "Please set it before running this script:"
    echo "export GITHUB_TOKEN=your_github_token_here"
    echo "./setup_github_repo.sh"
    exit 1
fi

echo "Repository: ${GITHUB_USERNAME}/${REPO_NAME}"
echo ""

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
    echo "Git repository initialized."
else
    echo "Git repository already exists."
fi

# Add all files
echo "Adding files to Git..."
git add .

# Commit files
echo "Committing files..."
git commit -m "Initial commit: Fieldwork forecasting system with skforecast and LightGBM"

# Add remote origin
echo "Setting up remote origin..."
git remote remove origin 2>/dev/null || true
git remote add origin https://${GITHUB_TOKEN}@github.com/${GITHUB_USERNAME}/${REPO_NAME}.git

# Create main branch and push
echo "Pushing to GitHub..."
git branch -M main
git push -u origin main

echo ""
echo "Repository setup complete!"
echo ""
echo "Next steps:"
echo "1. Go to https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
echo "2. Go to Settings > Secrets and variables > Actions"
echo "3. Add the GOOGLE_SHEETS_CREDENTIALS secret with your GCP service account JSON"
echo "4. The workflow will run automatically every Monday at 8:00 AM UTC"
echo "5. You can also trigger it manually from the Actions tab"
echo ""
echo "Repository URL: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"

