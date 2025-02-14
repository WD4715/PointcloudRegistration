#include "Postprocessing.hpp"
#include <pcl/common/common.h>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <map>
#include <limits>
#include <cfloat>
#include <pcl/common/transforms.h>
#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>
#include <Eigen/Geometry>

// splitReferenceNotchesByX
std::pair<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr>
splitReferenceNotchesByX(pcl::PointCloud<pcl::PointXYZRGB>::Ptr referenceNotches) {
    std::sort(referenceNotches->points.begin(), referenceNotches->points.end(),
              [](const pcl::PointXYZRGB& a, const pcl::PointXYZRGB& b) {
                  return a.x < b.x;
              });
    size_t splitIndex = 0;
    float maxGap = 0.0;
    for (size_t i = 1; i < referenceNotches->points.size(); ++i) {
        float gap = referenceNotches->points[i].x - referenceNotches->points[i - 1].x;
        if (gap > maxGap) {
            maxGap = gap;
            splitIndex = i;
        }
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster1(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster2(new pcl::PointCloud<pcl::PointXYZRGB>);
    float xMid = (referenceNotches->points[splitIndex - 1].x + referenceNotches->points[splitIndex].x) / 2.0;
    for (size_t i = 0; i < splitIndex; ++i) {
        pcl::PointXYZRGB p = referenceNotches->points[i];
        p.r = 0;
        p.g = 0;
        p.b = 255;
        cluster1->push_back(p);
    }
    for (size_t i = splitIndex; i < referenceNotches->points.size(); ++i) {
        pcl::PointXYZRGB p = referenceNotches->points[i];
        p.r = 255;
        p.g = 0;
        p.b = 0;
        cluster2->push_back(p);
    }
    return {cluster1, cluster2};
}

// splitClusterByZ
std::pair<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr>
splitClusterByZ(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster) {
    std::sort(cluster->points.begin(), cluster->points.end(),
              [](const pcl::PointXYZRGB& a, const pcl::PointXYZRGB& b) {
                  return a.z < b.z;
              });
    size_t splitIndex = 0;
    float maxGap = 0.0;
    for (size_t i = 1; i < cluster->points.size(); ++i) {
        float gap = cluster->points[i].z - cluster->points[i - 1].z;
        if (gap > maxGap) {
            maxGap = gap;
            splitIndex = i;
        }
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr lowZCluster(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr highZCluster(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (size_t i = 0; i < splitIndex; ++i) {
        pcl::PointXYZRGB p = cluster->points[i];
        p.r = 255;
        p.g = 255;
        p.b = 0;
        lowZCluster->push_back(p);
    }
    for (size_t i = splitIndex; i < cluster->points.size(); ++i) {
        pcl::PointXYZRGB p = cluster->points[i];
        p.r = 0;
        p.g = 255;
        p.b = 0;
        highZCluster->push_back(p);
    }
    return {lowZCluster, highZCluster};
}

// splitReferenceNotchesByXZ
std::pair<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr>
splitReferenceNotchesByXZ(pcl::PointCloud<pcl::PointXYZRGB>::Ptr referenceNotches) {
    if (referenceNotches->points.empty()) {
        std::cerr << "Error: Input cloud is empty!" << std::endl;
        return {nullptr, nullptr};
    }
    std::sort(referenceNotches->points.begin(), referenceNotches->points.end(),
              [](const pcl::PointXYZRGB& a, const pcl::PointXYZRGB& b) {
                  return (a.x == b.x) ? (a.z < b.z) : (a.x < b.x);
              });
    size_t splitIndex = 0;
    float maxDistance = 0.0;
    for (size_t i = 1; i < referenceNotches->points.size(); ++i) {
        float dx = referenceNotches->points[i].x - referenceNotches->points[i - 1].x;
        float dz = referenceNotches->points[i].z - referenceNotches->points[i - 1].z;
        float distance = std::sqrt(dx * dx + dz * dz);
        if (distance > maxDistance) {
            maxDistance = distance;
            splitIndex = i;
        }
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster1(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster2(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (size_t i = 0; i < splitIndex; ++i) {
        pcl::PointXYZRGB p = referenceNotches->points[i];
        p.r = 0;
        p.g = 0;
        p.b = 255;
        cluster1->push_back(p);
    }
    for (size_t i = splitIndex; i < referenceNotches->points.size(); ++i) {
        pcl::PointXYZRGB p = referenceNotches->points[i];
        p.r = 255;
        p.g = 0;
        p.b = 0;
        cluster2->push_back(p);
    }
    return {cluster1, cluster2};
}

// splitContingentRegionsByY
std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> splitContingentRegionsByY(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, float threshold) {
    
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> regions;
    if (cloud->empty()) return regions;
    std::sort(cloud->points.begin(), cloud->points.end(), 
              [](const pcl::PointXYZRGB& a, const pcl::PointXYZRGB& b) {
                  return a.y < b.y;
              });
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr currentRegion(new pcl::PointCloud<pcl::PointXYZRGB>);
    currentRegion->push_back(cloud->points[0]);
    for (size_t i = 1; i < cloud->size(); ++i) {
        float deltaY = std::abs(cloud->points[i].y - cloud->points[i - 1].y);
        if (deltaY > threshold) {
            regions.push_back(currentRegion);
            currentRegion.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        }
        currentRegion->push_back(cloud->points[i]);
    }
    if (!currentRegion->empty()) {
        regions.push_back(currentRegion);
    }
    std::cout << "Total Contingent Regions Found: " << regions.size() << std::endl;
    return regions;
}


// transformPointCloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformPointCloud(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, 
    const Eigen::Matrix4f& transformationMatrix) 
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformedCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::transformPointCloud(*cloud, *transformedCloud, transformationMatrix);
    return transformedCloud;
}

// computePlaneNormal using PCA
Eigen::Vector3f computePlaneNormal(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
    pcl::PCA<pcl::PointXYZRGB> pca;
    pca.setInputCloud(cloud);
    Eigen::Matrix3f eigenVectors = pca.getEigenVectors();
    return eigenVectors.col(2);
}

// SecondAlignment
Eigen::Matrix4f SecondAlignment(pcl::PointCloud<pcl::PointXYZRGB>::Ptr RclusterZ2, pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusterZ2) {
    Eigen::Vector3f normalR = computePlaneNormal(RclusterZ2);
    Eigen::Vector3f normalC = computePlaneNormal(clusterZ2);
    Eigen::Vector3f rotationAxis = normalR.cross(normalC);
    rotationAxis.normalize();
    float angle = std::acos(normalR.dot(normalC));
    Eigen::AngleAxisf rotation(angle, rotationAxis);
    Eigen::Matrix3f rotationMatrix = rotation.toRotationMatrix();
    Eigen::Matrix4f transformationMatrix = Eigen::Matrix4f::Identity();
    transformationMatrix.block<3,3>(0,0) = rotationMatrix;
    return transformationMatrix;
}

// computeTranslationMatrixSide
Eigen::Matrix4f computeTranslationMatrix(pcl::PointCloud<pcl::PointXYZRGB>::Ptr RclusterZ2, pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusterZ2) {
    Eigen::Vector4f centroidR, centroidC;
    pcl::compute3DCentroid(*RclusterZ2, centroidR);
    pcl::compute3DCentroid(*clusterZ2, centroidC);
    Eigen::Vector3f translation = centroidC.head<3>() - centroidR.head<3>();
    Eigen::Matrix4f translationMatrix = Eigen::Matrix4f::Identity();
    translationMatrix.block<3,1>(0,3) = translation;
    return translationMatrix;
}


Eigen::Matrix4f computeTranslationMatrixBottom(pcl::PointCloud<pcl::PointXYZRGB>::Ptr RclusterZ2, 
                                              pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusterZ2) {
    pcl::PointXYZRGB minPtR, maxPtR;
    pcl::getMinMax3D(*RclusterZ2, minPtR, maxPtR);

    float maxX_R = maxPtR.x * 3.0;
    float minX_R = minPtR.x * 3.0;
    float minY_R = minPtR.y * 0.901;
    float maxY_R = maxPtR.y * 1.017;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredClusterZ2(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (const auto& p : clusterZ2->points) {
        if (p.x < maxX_R && p.x >= minX_R && p.y >= minY_R && p.y <= maxY_R) {
            filteredClusterZ2->points.push_back(p);
        }
    }
    if (filteredClusterZ2->empty()) {
        std::cerr << "Error: No valid points in clusterZ2 after filtering" << std::endl;
        return Eigen::Matrix4f::Identity();
    }
    Eigen::Vector4f centroidR, centroidC;
    pcl::compute3DCentroid(*RclusterZ2, centroidR);
    pcl::compute3DCentroid(*filteredClusterZ2, centroidC);
    Eigen::Vector3f translation = centroidC.head<3>() - centroidR.head<3>();
    Eigen::Matrix4f translationMatrix = Eigen::Matrix4f::Identity();
    translationMatrix.block<3,1>(0,3) = translation;
    return translationMatrix;
}


// computeTranslationMatrixTop 
Eigen::Matrix4f computeTranslationMatrixTop(pcl::PointCloud<pcl::PointXYZRGB>::Ptr RclusterZ2, 
                                         pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusterZ2) {
    pcl::PointXYZRGB minPtR, maxPtR;
    pcl::getMinMax3D(*RclusterZ2, minPtR, maxPtR);
    float minX_R = minPtR.x * 0.9;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredClusterZ2(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (const auto& p : clusterZ2->points) {
        if (p.x >= minX_R) {
            filteredClusterZ2->points.push_back(p);
        }
    }
    if (filteredClusterZ2->empty()) {
        std::cerr << "Error: No valid points in clusterZ2 after filtering" << std::endl;
        return Eigen::Matrix4f::Identity();
    }
    Eigen::Vector4f centroidR, centroidC;
    pcl::compute3DCentroid(*RclusterZ2, centroidR);
    pcl::compute3DCentroid(*filteredClusterZ2, centroidC);
    Eigen::Vector3f translation = centroidC.head<3>() - centroidR.head<3>();
    Eigen::Matrix4f translationMatrix = Eigen::Matrix4f::Identity();
    translationMatrix.block<3,1>(0,3) = translation;
    return translationMatrix;
}
