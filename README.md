# FastGraphCut

Final Project for **How to Write Fast Code II (18-646)** at **Carnegie Mellon University**.

This project implements a parallelized version of the GrabCut image segmentation algorithm, aiming to improve runtime performance over a sequential baseline.

## Requirements

- [OpenCV](https://opencv.org/) (version 4.x recommended)
- CMake (version 3.10 or higher)
- A C++17 compatible compiler (e.g., `g++`, `clang++`)

## Build

- mkdir build
- cd build
- cmake ..

## Running sequential version

- cd build
- make
- ./SlowGrabCut <image_path> x1 y1 x2 y2

Where image_path is the path to your image, and x1, y1, x2, y2 are the coordinates of the bounding box.

e.g. ./SlowGrabCut ../dataset/large/flower.jpg 531 300 3383 2101

## Running parallelized version

- cd build
- make
- ./FastGrabCut <image_path> x1 y1 x2 y2

Where image_path is the path to your image, and x1, y1, x2, y2 are the coordinates of the bounding box.

e.g. ./FastGrabCut ../dataset/large/flower.jpg 531 300 3383 2101
