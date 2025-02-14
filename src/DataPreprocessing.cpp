#include "DataPreprocessing.hpp"
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/common.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <algorithm>
#include <iostream>

pcl::PointCloud<pcl::PointXYZRGB>::Ptr filterSmallXClusters(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
    pcl::PointXYZRGB minPt, maxPt;
    pcl::getMinMax3D(*cloud, minPt, maxPt);
    float totalWidth = maxPt.x - minPt.x;
    float thresholdWidth = 0.3 * totalWidth;

    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance(0.02);
    ec.setMinClusterSize(50);
    ec.setMaxClusterSize(100000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(clusterIndices);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (const auto& indices : clusterIndices) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (int idx : indices.indices) {
            cluster->push_back(cloud->points[idx]);
        }
        pcl::PointXYZRGB minC, maxC;
        pcl::getMinMax3D(*cluster, minC, maxC);
        float clusterWidth = maxC.x - minC.x;
        if (clusterWidth >= thresholdWidth) {
            *filteredCloud += *cluster;
        }
    }
    return filteredCloud;
}

std::vector<float> detectYBreaks(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, float threshold) {
    std::vector<float> y_values;
    for (const auto& point : cloud->points) {
        y_values.push_back(point.y);
    }
    std::sort(y_values.begin(), y_values.end());
    std::vector<std::pair<float, float>> y_diffs;
    for (size_t i = 1; i < y_values.size(); i++) {
        float diff = y_values[i] - y_values[i - 1];
        y_diffs.push_back({y_values[i], diff});
    }
    std::vector<float> breakPoints;
    for (const auto& [y, diff] : y_diffs) {
        if (diff > threshold) {
            breakPoints.push_back(y);
        }
    }
    return breakPoints;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr filterTop2YSegments(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, float threshold) {
    std::vector<float> breakPoints = detectYBreaks(cloud, threshold);
    std::vector<std::pair<float, float>> y_segments;
    if (breakPoints.empty()) {
        pcl::PointXYZRGB minPt, maxPt;
        pcl::getMinMax3D(*cloud, minPt, maxPt);
        y_segments.push_back({minPt.y, maxPt.y});
    } else {
        pcl::PointXYZRGB minPt, maxPt;
        pcl::getMinMax3D(*cloud, minPt, maxPt);
        y_segments.push_back({minPt.y, breakPoints[0]});
        for (size_t i = 0; i < breakPoints.size() - 1; i++) {
            y_segments.push_back({breakPoints[i], breakPoints[i + 1]});
        }
        y_segments.push_back({breakPoints.back(), maxPt.y});
    }
    std::vector<std::pair<float, std::pair<float, float>>> segment_lengths;
    for (const auto& segment : y_segments) {
        float length = segment.second - segment.first;
        segment_lengths.push_back({length, segment});
    }
    std::sort(segment_lengths.begin(), segment_lengths.end(), [](auto& a, auto& b) {
        return a.first > b.first;
    });
    if (segment_lengths.size() > 2) {
        segment_lengths.resize(2);
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (const auto& [length, segment] : segment_lengths) {
        float y_min = segment.first;
        float y_max = segment.second;
        for (const auto& point : cloud->points) {
            if (point.y >= y_min && point.y <= y_max) {
                filteredCloud->push_back(point);
            }
        }
    }
    return filteredCloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr removeSmallComponents(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
    float min_height_ratio,
    float min_area_ratio
) {
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(cloud);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance(0.02);
    ec.setMinClusterSize(100);
    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    std::vector<std::pair<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, float>> clusters_with_metrics;
    float max_height = 0;
    float max_area = 0;
    for (const auto& indices : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (const auto& idx : indices.indices) {
            cluster->push_back(cloud->points[idx]);
        }
        pcl::PointXYZRGB min_pt, max_pt;
        pcl::getMinMax3D(*cluster, min_pt, max_pt);
        float height = max_pt.y - min_pt.y;
        float area = (max_pt.x - min_pt.x) * (max_pt.z - min_pt.z);
        max_height = std::max(max_height, height);
        max_area = std::max(max_area, area);
        clusters_with_metrics.push_back({cluster, height});
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (const auto& [cluster, height] : clusters_with_metrics) {
        if (height >= max_height * min_height_ratio) {
            *filtered_cloud += *cluster;
        }
    }
    return filtered_cloud;
}
