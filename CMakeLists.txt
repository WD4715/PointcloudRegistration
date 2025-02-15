cmake_minimum_required(VERSION 3.10)
project(PointCloudProcessing)

set(CMAKE_CXX_STANDARD 17)

# Find Library
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(Boost REQUIRED)

# include Directory
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

# add src
add_executable(PointCloudProcessing 
    src/main.cpp
    src/DataLoader.cpp
    src/DataPreprocessing.cpp
    src/Visualization.cpp
    src/NotionDetection.cpp
    src/Alignment.cpp
    src/Postprocessing.cpp
)

# library link
target_link_libraries(PointCloudProcessing 
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} 
    ${Boost_LIBRARIES}
)
