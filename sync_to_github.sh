#!/bin/bash

# Navigate to repository's directory
cd nn_softp_interface

# Check for changes in the CSV file
if git diff --exit-code --quiet data.csv; then
    echo "No changes in data.csv"
else
    # Add the updated CSV file to the staging area
    git add data.csv

    # Commit the changes
    git commit -m "Update data.csv with new entries"

    # Push the changes to GitHub
    git push origin main
fi
