#include "Alignment.hpp"
#include <pcl/features/normal_3d.h>
#include <pcl/common/centroid.h>
#include <Eigen/Eigenvalues>

Eigen::Matrix4f InitialAlignment(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source,
                                 pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source_original,
                                 pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target,
                                 pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target_original) {
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_est;
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
    
    pcl::PointCloud<pcl::Normal>::Ptr source_normal(new pcl::PointCloud<pcl::Normal>());
    pcl::PointCloud<pcl::Normal>::Ptr target_normal(new pcl::PointCloud<pcl::Normal>());
    
    normal_est.setInputCloud(source);
    normal_est.setSearchMethod(tree);
    normal_est.setKSearch(30);
    normal_est.compute(*source_normal);
    
    normal_est.setInputCloud(target);
    normal_est.compute(*target_normal);
    
    Eigen::Vector4f source_centroid, target_centroid;
    pcl::compute3DCentroid(*source, source_centroid);
    pcl::compute3DCentroid(*target, target_centroid);
    
    Eigen::Matrix3f source_covariance, target_covariance;
    pcl::computeCovarianceMatrixNormalized(*source, source_centroid, source_covariance);
    pcl::computeCovarianceMatrixNormalized(*target, target_centroid, target_covariance);
    
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> source_eigen(source_covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> target_eigen(target_covariance);
    
    Eigen::Matrix4f initial_transform = Eigen::Matrix4f::Identity();
    initial_transform.block<3,3>(0,0) = target_eigen.eigenvectors() * source_eigen.eigenvectors().transpose();
    initial_transform.block<3,1>(0,3) = target_centroid.head<3>() - (initial_transform.block<3,3>(0,0) * source_centroid.head<3>());
    
    return initial_transform;
}
