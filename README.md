# Centroid-Based-Object-Pose-Estimation-with-Ground-Plane-Normal-Vector-Analysis

This project focuses on processing depth data from Intel RealSense cameras to detect and segment objects in real-time. The primary goal is to enhance the capabilities of robotic systems by providing them with a better understanding of their environment using depth data.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [DetailedProjectBreakdown](#Detailed Project Breakdown)
- [Prerequisites](#prerequisites)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction
Ghost Robotics aims to revolutionize the robotics industry with cutting-edge technologies. This project aligns with that vision by leveraging depth data for object detection and segmentation, which can be crucial for tasks like navigation, obstacle avoidance, and object manipulation.

## Features
- **Real-time Depth Data Processing**: Extracts depth data from Intel RealSense cameras in real-time.
- **Object Detection**: Identifies objects in the environment based on depth data.
- **Object Segmentation**: Segments the detected objects for a more detailed analysis.
- **RANSAC Plane Fitting**: Implements the RANSAC algorithm for plane detection.
- **Region Growing Segmentation**: Uses region growing methods to segment the point cloud data.
- **Multithreading**: Utilizes multithreading to enhance performance and processing speed.
- **Visualization**: Provides a 3D visualization of the processed data.


## Detailed Project Breakdown

### 1. Depth Data Acquisition from Intel RealSense Cameras
- **Camera Configuration**: Configured the Intel RealSense cameras to capture depth data at optimal settings for object detection and segmentation.
- **Data Streaming**: Established a real-time data streaming pipeline to fetch depth data from the camera to the processing unit.

### 2. Data Pre-processing
- **Noise Reduction**: Implemented filters to reduce noise in the captured depth data, enhancing the clarity and accuracy of the point cloud.
- **Data Transformation**: Converted the depth data into a structured point cloud format suitable for further processing.

### 3. Object Detection
- **Depth Thresholding**: Used depth thresholding techniques to identify objects based on their distance from the camera.
- **Bounding Box Generation**: Generated bounding boxes around detected objects to highlight areas of interest.

### 4. Object Segmentation
- **RANSAC Plane Fitting**: Applied the RANSAC algorithm to detect planes in the point cloud, which helps in segmenting objects from the background.
- **Region Growing Segmentation**: Used region growing methods to further segment the point cloud data, isolating individual objects for detailed analysis.

### 5. Visualization
- **3D Visualization**: Developed a 3D visualization tool to display the processed point cloud data, allowing users to interactively explore the segmented objects.
- **GUI Integration**: Integrated a graphical user interface (GUI) to facilitate user interaction, enabling users to select points of interest in the real-time depth image.

### 6. Performance Optimization
- **Multithreading**: Implemented multithreading to parallelize data processing tasks, significantly enhancing the processing speed and overall system performance.
- **Algorithm Optimization**: Refined and optimized the core algorithms to ensure efficient and accurate processing of depth data.

### 7. Documentation and Testing
- **Code Documentation**: Thoroughly documented the codebase, ensuring clarity and ease of understanding for future developers and collaborators.
- **Unit Testing**: Developed unit tests for critical components of the system, ensuring the reliability and robustness of the solution.

## Prerequisites
- Intel RealSense SDK 2.0
- OpenCV 4.x
- Point Cloud Library (PCL) 1.11
- Eigen Library
- Boost Library
- C++ Compiler with C++11 support

## Dependencies
- **C++17**: The project uses C++17 features.
- **Eigen3**: Required for matrix and linear algebra operations.
- **ryml**: YAML parser library.
- **PCL (Point Cloud Library)**: Used for point cloud processing.
- **Intel RealSense SDK**: For interfacing with Intel RealSense cameras.
- **OpenCV**: Used for image processing tasks.
- **loguru**: Logging library (ensure the library is in the specified path).
- **Threads**: For multithreading capabilities.

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/realsense-pointcloud-processing.git

# Navigate to the project directory
cd realsense-pointcloud-processing

# Create a build directory and navigate to it
mkdir build && cd build

# Compile the project
cmake ..
make
```
## Usage

```bash
# Run the main executable
./grasp_synthesis
```

- Use the GUI to select a point of interest in the real-time depth image. The program will process the depth data, detect and segment objects, and display the results in a 3D visualization window.

## Contributing
- Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
- This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
- Ghost Robotics for providing the opportunity to work on this innovative project.
- Intel for the RealSense technology.
- The open-source community for the libraries and tools.