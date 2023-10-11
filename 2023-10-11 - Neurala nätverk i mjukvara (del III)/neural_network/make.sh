#!/bin/bash

################################################################################
# @brief Script for building and running the neural network.
#
# @note Run the script in this directory by using the source command:
#       $ source make.sh
################################################################################

cd build # Directs to the build subdirectory.
make     # Builds the project.
cd ..    # Redirects to the base directory.
./run    # Runs the program (runs the "run" executable).