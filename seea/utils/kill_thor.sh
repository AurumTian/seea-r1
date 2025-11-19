#!/bin/bash

echo "Looking for AI2Thor processes..."
THOR_PROCESSES=$(ps aux | grep "thor-201909061227-Linux" | grep -v grep | awk '{print $2}')

if [ -z "$THOR_PROCESSES" ]; then
    echo "No AI2Thor processes found"
    exit 0
fi

echo "Found the following AI2Thor processes:"
ps aux | grep "thor-201909061227-Linux" | grep -v grep

echo "Terminating AI2Thor processes..."
for pid in $THOR_PROCESSES; do
    echo "Terminating process $pid"
    kill -9 $pid
done

echo "Checking for remaining processes..."
REMAINING=$(ps aux | grep "thor-201909061227-Linux" | grep -v grep | wc -l)

if [ "$REMAINING" -gt 0 ]; then
    echo "Warning: $REMAINING AI2Thor processes were not terminated"
    ps aux | grep "thor-201909061227-Linux" | grep -v grep
else
    echo "All AI2Thor processes have been successfully terminated"
fi

# Clean up Unity-related processes
echo "Cleaning up Unity-related processes..."
pkill -f Unity
pkill -f unity

echo "Cleanup complete"