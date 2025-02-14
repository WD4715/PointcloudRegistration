#ifndef VISUALIZATION_HPP
#define VISUALIZATION_HPP

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Core>

// Basic Visualization
void visualizePointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud);

// Visualization for the registration output
void visualizeRegistration(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &originalCloud,
                             const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &toTransformCloud,
                             const Eigen::Matrix4f &transformationMatrix);

// Visualizatio for the Clustering output 
void visualizeClusters(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster1, 
                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster2);

// Visualization for the final output
void visualizeFinal(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &originalCloud,
                    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &toTransformCloud);

#endif // VISUALIZATION_HPP
