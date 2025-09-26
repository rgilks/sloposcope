#!/bin/bash

# Sloposcope Fly.io Deployment Script
set -e

echo "🚀 Deploying Sloposcope to Fly.io..."

# Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    echo "❌ flyctl not found. Please install it first:"
    echo "curl -L https://fly.io/install.sh | sh"
    exit 1
fi

# Check if logged in to Fly.io
if ! flyctl auth whoami &> /dev/null; then
    echo "❌ Not logged in to Fly.io. Please run:"
    echo "flyctl auth login"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Deploy to Fly.io
echo "📦 Deploying to Fly.io..."
flyctl deploy

if [ $? -eq 0 ]; then
    echo "✅ Deployment successful!"
    echo ""
    echo "🔗 Your application is now live at:"
    echo "   https://sloposcope.fly.dev"
    echo ""
    echo "📝 Next steps:"
    echo "   1. Test the application"
    echo "   2. Set up a custom domain if desired"
    echo "   3. Monitor performance in Fly.io dashboard"
else
    echo "❌ Deployment failed"
    exit 1
fi
