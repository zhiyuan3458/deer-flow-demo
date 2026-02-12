#!/bin/bash

# Stop DeerFlow services by killing processes on ports 3000, 3001, and 8089
# This script safely stops all DeerFlow backend and frontend processes

echo "🛑 Stopping DeerFlow services..."

# Function to kill process on a specific port
kill_port() {
  local port=$1
  local pids=$(lsof -ti :$port 2>/dev/null)
  
  if [ -n "$pids" ]; then
    echo "  ├─ Stopping processes on port $port (PIDs: $pids)"
    echo "$pids" | xargs kill -15 2>/dev/null
    sleep 1
    
    # Force kill if still running
    local remaining=$(lsof -ti :$port 2>/dev/null)
    if [ -n "$remaining" ]; then
      echo "  ├─ Force stopping remaining processes on port $port"
      echo "$remaining" | xargs kill -9 2>/dev/null
    fi
    echo "  └─ Port $port cleared ✓"
  else
    echo "  └─ No processes on port $port"
  fi
}

# Stop backend (8089)
kill_port 8089

# Stop frontend (3000 and 3001)
kill_port 3000
kill_port 3001

# Also kill any remaining server.py or next dev processes
echo ""
echo "🔍 Checking for remaining DeerFlow processes..."

# Kill server.py processes
server_pids=$(ps aux | grep -E "[s]erver\.py|[u]v run server" | awk '{print $2}')
if [ -n "$server_pids" ]; then
  echo "  ├─ Stopping server.py processes (PIDs: $server_pids)"
  echo "$server_pids" | xargs kill -15 2>/dev/null
  sleep 1
fi

# Kill next dev processes
next_pids=$(ps aux | grep -E "[n]ext dev|[p]npm dev" | awk '{print $2}')
if [ -n "$next_pids" ]; then
  echo "  ├─ Stopping next dev processes (PIDs: $next_pids)"
  echo "$next_pids" | xargs kill -15 2>/dev/null
  sleep 1
fi

echo ""
echo "✅ DeerFlow services stopped"
echo ""
echo "Verify with: lsof -i :3000 -i :8089"
