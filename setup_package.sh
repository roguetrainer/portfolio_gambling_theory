#!/bin/bash

# This script sets up the complete portfolio gambling theory package

echo "Setting up Portfolio Gambling Theory package..."

# Create all necessary directories
mkdir -p src/portfolio_gambling notebooks docs tests examples data

echo "Package structure created successfully!"
echo "Directory tree:"
tree -L 2 || ls -R
