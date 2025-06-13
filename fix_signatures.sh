#!/bin/bash
echo "Searching for and approving all compiled libraries in your virtual environment..."
# Find all .so files within the 'lib' directory and run xattr on them
find ./lib -name "*.so" -exec xattr -d com.apple.quarantine {} \;
find ./lib -name "*.dylib" -exec xattr -d com.apple.quarantine {} \;
echo "Done. All found .so files have been approved."