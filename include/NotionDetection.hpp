#ifndef NOTION_DETECTION_HPP
#define NOTION_DETECTION_HPP

#include <string>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

// Reference Notch Detection
pcl::PointCloud<pcl::PointXYZRGB>::Ptr detectNotches(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, const std::string& filePath);

// Real Notch Detection
pcl::PointCloud<pcl::PointXYZRGB>::Ptr RealdetectNotches(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, const std::string& filePath);

#endif // NOTION_DETECTION_HPP
