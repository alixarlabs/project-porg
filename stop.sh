#!/bin/bash
# Project Porg - Stop All Services
#
# Stops all running containers for the project

echo "Stopping Project Porg services..."

# Stop voice container
docker stop porg-voice 2>/dev/null && echo "  Stopped porg-voice" || true

# Stop compose services
docker compose down 2>/dev/null && echo "  Stopped porg-vllm, porg-agent" || true

echo "All services stopped."
