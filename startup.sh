#!/bin/bash
set -e

# Create data directory if it doesn't exist
mkdir -p /data

# Download spaCy model if not present
if [ ! -d "/data/en_core_web_sm" ]; then
    echo "Downloading spaCy model to volume..."
    /app/.venv/bin/python -c "
import sys
import os
import tempfile
import shutil
sys.path.insert(0, '/app/.venv/lib/python3.11/site-packages')
try:
    import spacy
    # Create temp directory for download
    temp_dir = '/tmp/spacy_download'
    os.makedirs(temp_dir, exist_ok=True)
    os.environ['SPACY_DATA_PATH'] = temp_dir

    # Download model to temp directory
    spacy.cli.download('en_core_web_sm')

    # Move to volume
    src_path = os.path.join(temp_dir, 'en_core_web_sm')
    dest_path = '/data/en_core_web_sm'

    if os.path.exists(src_path):
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        shutil.move(src_path, dest_path)
        print('spaCy model moved to volume successfully')
    else:
        print('spaCy model downloaded successfully')

except ImportError:
    echo 'spaCy not available - will use fallback mode'
except Exception as e:
    print(f'Warning: Could not download spaCy model: {e}')
    # Create a marker file so we don't try again
    os.makedirs('/data/en_core_web_sm', exist_ok=True)
"
fi

# Set the spaCy data path to use the volume
export SPACY_DATA_PATH=/data

# Start the application
exec "$@"
