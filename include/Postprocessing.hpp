#ifndef POSTPROCESSING_HPP
#define POSTPROCESSING_HPP

#include <utility>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Core>

// X-Clustering using X-Axis
std::pair<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr>
splitReferenceNotchesByX(pcl::PointCloud<pcl::PointXYZRGB>::Ptr referenceNotches);

// Clustering using Z-Axis
std::pair<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr>
splitClusterByZ(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster);

// Clustering using XZ-Plane
std::pair<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr>
splitReferenceNotchesByXZ(pcl::PointCloud<pcl::PointXYZRGB>::Ptr referenceNotches);

// Selection for the Contingent Regions following Y-Axis
std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> splitContingentRegionsByY(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, float threshold = 10);

// Transformation
pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformPointCloud(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, 
    const Eigen::Matrix4f& transformationMatrix);

// Normal vector calculation using PCA
Eigen::Vector3f computePlaneNormal(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

// Second Alignment
Eigen::Matrix4f SecondAlignment(pcl::PointCloud<pcl::PointXYZRGB>::Ptr RclusterZ2, 
                                 pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusterZ2);

// Transformation for Side Data
Eigen::Matrix4f computeTranslationMatrix(pcl::PointCloud<pcl::PointXYZRGB>::Ptr RclusterZ2, 
                                          pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusterZ2);

// Transformation for Bottom Data
Eigen::Matrix4f computeTranslationMatrixBottom(pcl::PointCloud<pcl::PointXYZRGB>::Ptr RclusterZ2, 
                                                pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusterZ2);

// Transformation for Top Data
Eigen::Matrix4f computeTranslationMatrixTop(pcl::PointCloud<pcl::PointXYZRGB>::Ptr RclusterZ2, 
                                             pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusterZ2);

#endif // POSTPROCESSING_HPP
