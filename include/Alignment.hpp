#ifndef ALIGNMENT_HPP
#define ALIGNMENT_HPP

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Core>

// Initial Alignment using PCA
Eigen::Matrix4f InitialAlignment(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source,
                                 pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source_original,
                                 pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target,
                                 pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target_original);

#endif // ALIGNMENT_HPP
