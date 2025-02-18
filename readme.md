# PointCloudProcessing

PointCloudProcessing is a project that uses OpenCV, PCL, and Boost libraries to generate point clouds from reference and real data images, then performs preprocessing, notch detection, registration, and postprocessing. The processing pipeline varies depending on the panel location of the real data (REAL_SIDE, REAL_TOP, or REAL_BOTTOM).

> **Note:**  
> The reference image is pre-aligned to the origin by applying a `refTransform`, so it is not included in the overall transformation from the origin to the real data.

## Project Overview

The project processes two types of data:

- **Reference Data**  
  Reference images (e.g., `MASTER_TOP.tiff`, `MASTER_BOTTOM.tiff`, `MASTER_SIDE.tiff`) are loaded and transformed with a specific `refTransform` to align them with the origin coordinate system. The resulting reference point cloud is then downsampled and subjected to notch detection.

- **Real Data**  
  Real data images (e.g., `REAL_SIDE.tiff`, `REAL_TOP.tiff`, `REAL_BOTTOM.tiff`) are loaded and processed using different pipelines according to the panel type:
  - **REAL_SIDE:**  
    - Visualize downsampled reference data, notch detection on the reference, original real data, reference notch clustering by X-axis and by Z-axis (after X-axis), real data preprocessing (removing small areas), real data notch detection, initial rotation (using reference notch and real notch), second rotation, real notch clustering (using X-axis after Z-axis clustering), and final notch alignment.
  - **REAL_TOP:**  
    - In addition to the above, the Top panel includes extra preprocessing steps: filtering using X-axis clustering and Y-axis filtering to remove noise.
  - **REAL_BOTTOM:**  
    - Similar to the Top panel, the Bottom panel includes extra preprocessing steps (X-axis and Y-axis filtering) before proceeding to the notch detection and final alignment.

After registration, the overall transformation from the origin to the real data is computed by combining the registration matrices. For example, for REAL_SIDE:

```math
T_{align} = translationMat_{side} * SecondTransformationMatrix * InitialTransformationMatrix
```

```math
T_{final} = (T_{align})^{(-1)}
```

Similar calculations are performed for REAL_TOP and REAL_BOTTOM.

## Project Structure
```css
ProjectRoot/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── Alignment.hpp
│   ├── DataLoader.hpp
│   ├── DataPreprocessing.hpp
│   ├── NotionDetection.hpp
│   ├── Postprocessing.hpp
│   └── Visualization.hpp
└── src/
    ├── Alignment.cpp
    ├── DataLoader.cpp
    ├── DataPreprocessing.cpp
    ├── NotionDetection.cpp
    ├── Postprocessing.cpp
    ├── Visualization.cpp
    └── main.cpp

```

```bash
./PointCloudProcessing <reference_path> <real_path> [<vis1> ... <vis11>]
```

- **reference_path:** Path to the reference image.
- **real_path:** Path to the real data image.
- **vis1 ~ vis11:** (Optional) Visualization flags ("true" or "false").  
  If not fully provided, the defaults are: vis1–vis10 = false, and vis11 = true.

### Visualization Steps by Panel

#### For REAL_SIDE:
1. **Visualization1:** Downsampled data (Reference)
2. **Visualization2:** Reference Notch Detection
3. **Visualization3:** Real Data (Original)
4. **Visualization4:** Reference Notch Clustering By X-axis
5. **Visualization5:** Reference Notch Clustering By Z-axis After X-axis
6. **Visualization6:** Real Data Preprocessing (Remove Small Area)
7. **Visualization7:** Real Data Notch Detection
8. **Visualization8:** Initial Rotation using Reference Notch and Real Notch
9. **Visualization9:** Second Rotation using Reference Notch and Real Notch
10. **Visualization10:** Real Notch Clustering using X-axis After Z-axis clustering
11. **Visualization11:** Final Notch Alignment

#### For REAL_TOP:
1. **Visualization1:** Downsampled data (Reference)
2. **Visualization2:** Reference Notch Detection
3. **Visualization3:** Real Data (Original)
4. **Visualization4:** Preprocessing Step 1 – Filtering using X-axis clustering (Real Data Original)
5. **Visualization5:** Preprocessing Step 2 – Filtering using Y-axis to remove noise for disconnected Y
6. **Visualization6:** Preprocessing Step 3 – Real Data Preprocessing (Remove Small Area)
7. **Visualization7:** Real Data Notch Detection
8. **Visualization8:** Initial Rotation using Reference Notch and Real Notch
9. **Visualization9:** Second Rotation using Reference Notch and Real Notch
10. **Visualization10:** Real Notch Clustering for Reference and Real
11. **Visualization11:** Final Notch Alignment

#### For REAL_BOTTOM:
1. **Visualization1:** Downsampled data (Reference)
2. **Visualization2:** Reference Notch Detection
3. **Visualization3:** Real Data (Original)
4. **Visualization4:** Preprocessing Step 1 – Filtering using X-axis clustering (Real Data Original)
5. **Visualization5:** Preprocessing Step 2 – Filtering using Y-axis to remove noise for disconnected Y
6. **Visualization6:** Preprocessing Step 3 – Real Data Preprocessing (Remove Small Area)
7. **Visualization7:** Real Data Notch Detection
8. **Visualization8:** Initial Rotation using Reference Notch and Real Notch
9. **Visualization9:** Second Rotation using Reference Notch and Real Notch
10. **Visualization10:** Final Notch Alignment

## Overall Transformation Calculation

After registration, the overall alignment transformation from the origin to the real data is computed by combining the transformation matrices from the registration steps. For example, in the REAL_SIDE branch:
```math
T_{align} = translationMat_{side} * SecondTransformationMatrix * InitialTransformationMatrix 
```

```math
T_{final} = (T_{align})^{(-1)}
```

The final transformation `T_final` represents the relationship between the origin and the real data coordinate system.

## Build and Run Instructions

### Requirements

- CMake (version 3.10 or higher)
- A C++17 (or later) compatible compiler
- OpenCV, PCL, and Boost libraries

### Dependency
Recommended Versions \
OpenCV: 4.4.0 \
PCL: 1.10.0 \
Boost: (Compatible with PCL 1.10.0) \
Operating System: Ubuntu 20.04

### 1. Basic Development Tools Installation
First, update your system and install the essential development tools:

  ```bash 
  # Update and upgrade your system
  sudo apt update
  sudo apt upgrade
  
  # Install basic development tools
  sudo apt install -y build-essential cmake git pkg-config wget
  ```
### 2. OpenCV 4.4.0 Installation
Step 1: Install Dependencies
  
  ```bash
  sudo apt install -y build-essential cmake pkg-config
  sudo apt install -y libjpeg-dev libpng-dev libtiff-dev
  sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev libavresample-dev
  sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
  sudo apt install -y libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev
  sudo apt install -y libfaac-dev libmp3lame-dev libvorbis-dev
  sudo apt install -y libopencore-amrnb-dev libopencore-amrwb-dev
  sudo apt install -y libgtk-3-dev
  sudo apt install -y python3-dev python3-numpy
  sudo apt install -y libtbb-dev
  sudo apt install -y libatlas-base-dev gfortran
  sudo apt install -y libprotobuf-dev protobuf-compiler
  sudo apt install -y libgoogle-glog-dev libgflags-dev
  sudo apt install -y libgphoto2-dev libeigen3-dev libhdf5-dev doxygen
  ```

Step 2: Download and Build OpenCV
  ```bash
  # Download OpenCV 4.4.0 and opencv_contrib modules
  wget -O opencv.zip https://github.com/opencv/opencv/archive/4.4.0.zip
  wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.4.0.zip
  unzip opencv.zip
  unzip opencv_contrib.zip
  
  # Create a build directory and configure the build
  cd opencv-4.4.0
  mkdir build && cd build
  cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D INSTALL_C_EXAMPLES=OFF \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.4.0/modules \
        -D PYTHON_EXECUTABLE=/usr/bin/python3 \
        -D BUILD_EXAMPLES=OFF ..
  
  # Build and install OpenCV
  make -j$(nproc)
  sudo make install
  sudo ldconfig
  
  ```
### 3. PCL 1.10.0 and Dependencies Installation
Step 1: Install PCL 1.10.0 Packages
  ```bash
  sudo apt install -y libpcl-dev=1.10.0* \
                      libpcl-apps1.10=1.10.0* \
                      libpcl-common1.10=1.10.0* \
                      libpcl-features1.10=1.10.0* \
                      libpcl-filters1.10=1.10.0* \
                      libpcl-io1.10=1.10.0* \
                      libpcl-kdtree1.10=1.10.0* \
                      libpcl-keypoints1.10=1.10.0* \
                      libpcl-ml1.10=1.10.0* \
                      libpcl-octree1.10=1.10.0* \
                      libpcl-outofcore1.10=1.10.0* \
                      libpcl-people1.10=1.10.0* \
                      libpcl-recognition1.10=1.10.0* \
                      libpcl-registration1.10=1.10.0* \
                      libpcl-sample-consensus1.10=1.10.0* \
                      libpcl-search1.10=1.10.0* \
                      libpcl-segmentation1.10=1.10.0* \
                      libpcl-stereo1.10=1.10.0* \
                      libpcl-surface1.10=1.10.0* \
                      libpcl-tracking1.10=1.10.0* \
                      libpcl-visualization1.10=1.10.0*
  
  ```
Step 2: Install Additional PCL Dependencies
  ```bash
  sudo apt install -y libvtk7-dev libvtk7-qt-dev
  sudo apt install -y libflann-dev
  sudo apt install -y libqhull-dev
  sudo apt install -y libboost-all-dev
  sudo apt install -y libeigen3-dev
  
  ```
### 4. Verify Installation


#### OpenCV Version:
  ```bash
  pkg-config --modversion opencv4
  ```
#### PCL Version:
  ```bash
  dpkg -l | grep libpcl
  ```
#### Boost Version:
  ``` bash
  dpkg -l | grep libboost
  
  ```





### Building

1. Clone the repository:
   ```bash
   git clone https://github.com/WD4715/PointcloudRegistration.git
   cd PointCloudProcessing

2. Create a build directory and run CMake:
    ```bash
    mkdir build
    cd build
    cmake ..
    ```
3. Build the project:
    ```bash
    make
    ```
### Running
Provide at least a reference image path and a real data image path, along with optional visualization flags:

  ```bash
  ./PointCloudProcessing <reference_path> <real_path> [<vis1> ... <vis11>]
  
  ```
Example:

  ```bash
  ./PointCloudProcessing /path/to/MASTER_SIDE.tiff /path/to/REAL_SIDE.tiff true false true false false false false false false false true
  
  ```
If the visualization flags are not fully provided, the defaults are: vis1–vis10 = false and vis11 = true.
