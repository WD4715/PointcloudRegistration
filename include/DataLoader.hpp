#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include <string>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

// Reference Data loader and Translation
void transformAndColorPointCloud(const cv::Mat& image, 
                                 pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, 
                                 const cv::Mat& transformation);

// Voxel Grid Downsampling for the Reference data
void downsamplePointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, 
                          pcl::PointCloud<pcl::PointXYZRGB>::Ptr& downsampledCloud, 
                          float leafSize);

// Real Data Loader
pcl::PointCloud<pcl::PointXYZRGB>::Ptr loadRealPointCloud(const std::string& filePath);

#endif // DATA_LOADER_HPP
