#!/bin/bash

# Sloposcope Cloudflare Deployment Script
set -e

echo "🚀 Starting Sloposcope deployment to Cloudflare..."

# Check if wrangler is installed
if ! command -v wrangler &> /dev/null; then
    echo "❌ Wrangler CLI not found. Please install it first:"
    echo "npm install -g wrangler"
    exit 1
fi

# Check if logged in to Cloudflare
if ! wrangler whoami &> /dev/null; then
    echo "❌ Not logged in to Cloudflare. Please run:"
    echo "wrangler login"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Deploy the Python Worker (Backend)
echo "📦 Deploying Python Worker..."
wrangler deploy --env production

if [ $? -eq 0 ]; then
    echo "✅ Python Worker deployed successfully"
else
    echo "❌ Python Worker deployment failed"
    exit 1
fi

# Deploy the Next.js Frontend
echo "🌐 Deploying Next.js Frontend..."
cd web

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

# Build the frontend
echo "🔨 Building frontend..."
npm run build

# Deploy to Cloudflare Pages
echo "🚀 Deploying to Cloudflare Pages..."
npx wrangler pages deploy out --project-name sloposcope-web

if [ $? -eq 0 ]; then
    echo "✅ Frontend deployed successfully"
else
    echo "❌ Frontend deployment failed"
    exit 1
fi

cd ..

echo "🎉 Deployment completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Check your Cloudflare dashboard for the deployed worker and pages"
echo "2. Set up a custom domain if desired"
echo "3. Test the application"
echo ""
echo "🔗 Your application should be available at:"
echo "   Worker: https://sloposcope.rob-gilks-gmail-coms-account.workers.dev"
echo "   Pages: https://sloposcope-web.pages.dev"
