#!/bin/bash

# PPF Python Bindings Installation Script

set -e

echo "PPF Python Bindings Installation"
echo "================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if we're in the right directory
if [ ! -f "ppf_python.cpp" ]; then
    echo "Error: Please run this script from the python directory"
    exit 1
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create build directory
echo "Creating build directory..."
rm -rf build
mkdir build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DPYTHON_EXECUTABLE=$(which python)

# Build
echo "Building..."
make -j$(nproc)

# Copy the built module
echo "Installing module..."
cd ..
cp build/ppf.so .

# Test the installation
echo "Testing installation..."
python -c "import ppf; print('✓ PPF module imported successfully!')"

echo ""
echo "Installation completed successfully!"
echo ""
echo "You can now use the PPF library in Python:"
echo "  python simple_example.py"
echo ""
echo "Or run the test suite:"
echo "  python test_binding.py"