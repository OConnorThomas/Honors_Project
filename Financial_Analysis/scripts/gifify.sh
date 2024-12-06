#!/bin/bash

# Output GIF file name
output_gif="temp.gif"

# Check if there are any PNG files in the directory
png_files=(*.png)
if [ ${#png_files[@]} -eq 0 ]; then
  echo "No PNG files found in the directory."
  exit 1
fi

# Convert PNG images to a single GIF with a 100 ms delay between frames
# The `-delay 10` option sets the delay between frames in hundredths of a second (100 ms = 10)
convert -delay 10 *.png "$output_gif"

# Check if the conversion was successful
if [ $? -eq 0 ]; then
  echo "GIF created successfully: $output_gif"
else
  echo "Failed to create GIF."
  exit 1
fi
