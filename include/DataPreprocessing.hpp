#ifndef DATA_PREPROCESSING_HPP
#define DATA_PREPROCESSING_HPP

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>


// Clustering using X-Axis
pcl::PointCloud<pcl::PointXYZRGB>::Ptr filterSmallXClusters(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud);

// Using contingent Data following Y-Axis
std::vector<float> detectYBreaks(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, float threshold = 5.0);

// Selection for the High Y 
pcl::PointCloud<pcl::PointXYZRGB>::Ptr filterTop2YSegments(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, float threshold = 10.0);

// Remove the small area
pcl::PointCloud<pcl::PointXYZRGB>::Ptr removeSmallComponents(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
    float min_height_ratio = 0.3,
    float min_area_ratio = 0.2
);

#endif // DATA_PREPROCESSING_HPP
