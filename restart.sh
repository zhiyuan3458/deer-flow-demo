#!/bin/bash

# Restart DeerFlow services
# This script stops all running services and starts them fresh

echo "🔄 Restarting DeerFlow services..."
echo ""

# Stop existing services
./stop.sh

echo ""
echo "⏳ Waiting for cleanup..."
sleep 2

# Start services
echo ""
echo "🚀 Starting DeerFlow..."
echo ""

./bootstrap.sh -d
