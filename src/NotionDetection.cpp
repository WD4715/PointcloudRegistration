#include "NotionDetection.hpp"
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <tuple>
#include <vector>

pcl::PointCloud<pcl::PointXYZRGB>::Ptr detectNotches(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, const std::string& filePath) {
   if (filePath.find("MASTER_SIDE.tiff") != std::string::npos) {
       std::sort(cloud->points.begin(), cloud->points.end(), [](const pcl::PointXYZRGB& a, const pcl::PointXYZRGB& b) {
           return a.x < b.x;
       });

       size_t splitIndex = 0;
       float maxGap = 0.0;
       for (size_t i = 1; i < cloud->points.size(); ++i) {
           float gap = cloud->points[i].x - cloud->points[i - 1].x;
           if (gap > maxGap) {
               maxGap = gap;
               splitIndex = i;
           }
       }

       pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster1(new pcl::PointCloud<pcl::PointXYZRGB>);
       pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster2(new pcl::PointCloud<pcl::PointXYZRGB>);
       cluster1->points.insert(cluster1->points.end(), cloud->points.begin(), cloud->points.begin() + splitIndex);
       cluster2->points.insert(cluster2->points.end(), cloud->points.begin() + splitIndex, cloud->points.end());

       double sumZ1 = 0, sumZ2 = 0;
       for (const auto& p : *cluster1) sumZ1 += p.z;
       for (const auto& p : *cluster2) sumZ2 += p.z;
       double avgZ1 = sumZ1 / cluster1->size();
       double avgZ2 = sumZ2 / cluster2->size();

       pcl::PointCloud<pcl::PointXYZRGB>::Ptr targetCluster = (avgZ1 < avgZ2) ? cluster1 : cluster2;
       pcl::PointCloud<pcl::PointXYZRGB>::Ptr nonTargetCluster = (avgZ1 < avgZ2) ? cluster2 : cluster1;

       pcl::PointXYZRGB minPt, maxPt;
       pcl::getMinMax3D(*targetCluster, minPt, maxPt);

       double radius = std::sqrt(std::pow(maxPt.x - minPt.x, 2) + std::pow(maxPt.z - minPt.z, 2)) / 2.0 * 4.0;
       double center_x = (minPt.x + maxPt.x) / 2.0;
       double center_z = (minPt.z + maxPt.z) / 2.0;

       pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
       kdtree.setInputCloud(targetCluster);
       std::vector<int> pointIdxRadiusSearch;
       std::vector<float> pointRadiusSquaredDistance;

       std::vector<std::pair<double, std::vector<pcl::PointXYZRGB>>> y_segments;
       
       for (double y = minPt.y; y <= maxPt.y; y += radius) {
           pcl::PointXYZRGB searchPoint;
           searchPoint.y = y;
           searchPoint.x = center_x;
           searchPoint.z = center_z;

           if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
               std::vector<pcl::PointXYZRGB> segment_points;
               for (int idx : pointIdxRadiusSearch) {
                   segment_points.push_back(targetCluster->points[idx]);
               }
               double avg_z = 0.0;
               for (const auto& p : segment_points) {
                   avg_z += p.z;
               }
               avg_z /= segment_points.size();
               y_segments.push_back({avg_z, segment_points});
           }
       }

       std::sort(y_segments.begin(), y_segments.end(),
                 [](const auto& a, const auto& b) { return a.first > b.first; });

       pcl::PointCloud<pcl::PointXYZRGB>::Ptr notchCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
       const std::vector<std::tuple<int, int, int>> colors = {
           {255, 0, 0},
           {0, 255, 0},
           {0, 0, 255}
       };

       for (size_t i = 0; i < std::min(size_t(3), y_segments.size()); ++i) {
           double y_center = y_segments[i].second[0].y;
           for (const auto& point : targetCluster->points) {
               if (std::abs(point.y - y_center) <= radius) {
                   pcl::PointXYZRGB colored_point = point;
                   colored_point.r = std::get<0>(colors[i]);
                   colored_point.g = std::get<1>(colors[i]);
                   colored_point.b = std::get<2>(colors[i]);
                   notchCloud->points.push_back(colored_point);
               }
           }
           for (const auto& point : nonTargetCluster->points) {
               if (std::abs(point.y - y_center) <= radius) {
                   pcl::PointXYZRGB colored_point = point;
                   colored_point.r = std::get<0>(colors[i]);
                   colored_point.g = std::get<1>(colors[i]);
                   colored_point.b = std::get<2>(colors[i]);
                   notchCloud->points.push_back(colored_point);
               }
           }
       }
       return notchCloud;
   } else {
       pcl::PointXYZRGB minPt, maxPt;
       pcl::getMinMax3D(*cloud, minPt, maxPt);
       double radius = (maxPt.x - minPt.x) / 2.0 * 7.0;
       double center_x = minPt.x + radius;
       double center_z = (minPt.z + maxPt.z) / 2.0;
       std::vector<std::pair<double, int>> y_counts;
       pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
       kdtree.setInputCloud(cloud);
       std::vector<int> pointIdxRadiusSearch;
       std::vector<float> pointRadiusSquaredDistance;
       pcl::PointCloud<pcl::PointXYZRGB>::Ptr notchCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
       for (double y = minPt.y; y <= maxPt.y; y += radius) {
           int count = 0;
           pcl::PointXYZRGB searchPoint;
           searchPoint.y = y;
           searchPoint.x = center_x;
           searchPoint.z = center_z;
           if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
               count = pointIdxRadiusSearch.size();
           }
           y_counts.push_back({y, count});
       }
       if (y_counts.size() > 2) {
           y_counts.erase(y_counts.begin());
           y_counts.pop_back();
       }
       std::sort(y_counts.begin(), y_counts.end(), [](auto& a, auto& b) { return a.second < b.second; });
       std::vector<std::pair<double, pcl::RGB>> notch_colors = {
           {y_counts[0].first, {255, 0, 0}},
           {y_counts[1].first, {0, 255, 0}},
           {y_counts[2].first, {0, 0, 255}},
       };
       for (auto& [y, color] : notch_colors) {
           for (auto& point : cloud->points) {
               if (point.x >= center_x - radius && point.x <= center_x + radius &&
                   point.y >= y - radius && point.y <= y + radius) {
                   point.r = color.r;
                   point.g = color.g;
                   point.b = color.b;
                   notchCloud->push_back(point);
               }
           }
       }
       return notchCloud;
   }
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr RealdetectNotches(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, const std::string& filePath) {
    if (filePath.find("REAL_SIDE.tiff") != std::string::npos) {
        // 첫 번째 코드 실행
        std::sort(cloud->points.begin(), cloud->points.end(), [](const pcl::PointXYZRGB& a, const pcl::PointXYZRGB& b) {
            return a.z < b.z;
        });

        size_t splitIndex = 0;
        float maxGap = 0.0;
        for (size_t i = 1; i < cloud->points.size(); ++i) {
            float gap = cloud->points[i].z - cloud->points[i - 1].z;
            if (gap > maxGap) {
                maxGap = gap;
                splitIndex = i;
            }
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster1(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster2(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (size_t i = 0; i < splitIndex; i++) {
            cluster1->push_back(cloud->points[i]);
        }
        for (size_t i = splitIndex; i < cloud->points.size(); i++) {
            cluster2->push_back(cloud->points[i]);
        }

        double avgZ1 = std::accumulate(cluster1->begin(), cluster1->end(), 0.0, 
            [](double sum, const pcl::PointXYZRGB& p) { return sum + p.z; }) / cluster1->size();
        double avgZ2 = std::accumulate(cluster2->begin(), cluster2->end(), 0.0, 
            [](double sum, const pcl::PointXYZRGB& p) { return sum + p.z; }) / cluster2->size();

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr targetCluster = (avgZ1 > avgZ2) ? cluster1 : cluster2;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr nonTargetCluster = (avgZ1 > avgZ2) ? cluster2 : cluster1;

        std::sort(targetCluster->points.begin(), targetCluster->points.end(), 
            [](const pcl::PointXYZRGB& a, const pcl::PointXYZRGB& b) {
                return a.y < b.y;
            });

        std::vector<size_t> gaps;
        for (size_t i = 1; i < targetCluster->size(); ++i) {
            if ((targetCluster->points[i].y - targetCluster->points[i - 1].y) > maxGap) {
                gaps.push_back(i);
            }
        }

        if (gaps.size() < 2) {
            gaps.clear();
            size_t thirdSize = targetCluster->size() / 3;
            gaps.push_back(thirdSize);
            gaps.push_back(2 * thirdSize);
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr segment1(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr segment2(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr segment3(new pcl::PointCloud<pcl::PointXYZRGB>);

        segment1->points.insert(segment1->points.end(), targetCluster->points.begin(), 
                            targetCluster->points.begin() + gaps[0]);
        segment2->points.insert(segment2->points.end(), targetCluster->points.begin() + gaps[0], 
                            targetCluster->points.begin() + gaps[1]);
        segment3->points.insert(segment3->points.end(), targetCluster->points.begin() + gaps[1], 
                            targetCluster->points.end());

        auto processSegmentWithExtendedRange = [&nonTargetCluster](
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr segment, 
            const std::vector<float>& colors) {
            
            float min_y = FLT_MAX;
            float max_y = -FLT_MAX;
            for (const auto& point : segment->points) {
                min_y = std::min(min_y, point.y);
                max_y = std::max(max_y, point.y);
            }

            float y_distance = max_y - min_y;
            float extended_min_y = min_y - (y_distance * 0.3f);
            float extended_max_y = max_y + (y_distance * 0.3f);

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr extended_segment(new pcl::PointCloud<pcl::PointXYZRGB>);
            
            for (auto& point : segment->points) {
                pcl::PointXYZRGB colored_point = point;
                colored_point.r = colors[0];
                colored_point.g = colors[1];
                colored_point.b = colors[2];
                extended_segment->points.push_back(colored_point);
            }

            for (const auto& point : nonTargetCluster->points) {
                if (point.y >= extended_min_y && point.y <= extended_max_y) {
                    pcl::PointXYZRGB colored_point = point;
                    colored_point.r = colors[0];
                    colored_point.g = colors[1];
                    colored_point.b = colors[2];
                    extended_segment->points.push_back(colored_point);
                }
            }

            return extended_segment;
        };

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr extended_segment1 = 
            processSegmentWithExtendedRange(segment1, {255, 0, 0});
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr extended_segment2 = 
            processSegmentWithExtendedRange(segment2, {0, 255, 0});
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr extended_segment3 = 
            processSegmentWithExtendedRange(segment3, {0, 0, 255});

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr notchCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        *notchCloud += *extended_segment1;
        *notchCloud += *extended_segment2;
        *notchCloud += *extended_segment3;

        return notchCloud;
    } else {
        // 두 번째 코드 실행
        auto findContinuousRegions = [](const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, double minGap) {
            std::vector<double> y_values;
            for (const auto& point : cloud->points) {
                y_values.push_back(point.y);
            }
            std::sort(y_values.begin(), y_values.end());
            std::vector<std::pair<double, double>> regions;
            if (y_values.empty()) return regions;
            double region_start = y_values.front();
            double prev_y = y_values.front();
            for (size_t i = 1; i < y_values.size(); ++i) {
                if (y_values[i] - prev_y > minGap) {
                    regions.push_back({region_start, prev_y});
                    region_start = y_values[i];
                }
                prev_y = y_values[i];
            }
            regions.push_back({region_start, prev_y});
            return regions;
        };

        pcl::PointXYZRGB minPt, maxPt;
        pcl::getMinMax3D(*cloud, minPt, maxPt);

        double radius = (maxPt.x - minPt.x) / 2.0 * 7.0;
        double center_x = minPt.x + radius;
        double center_z = (minPt.z + maxPt.z) / 2.0;

        double minGap = radius * 0.5;
        auto continuous_regions = findContinuousRegions(cloud, minGap);

        std::cout << "Found " << continuous_regions.size() << " continuous regions:" << std::endl;
        for (const auto& region : continuous_regions) {
            std::cout << "Region: [" << region.first << ", " << region.second << "]" << std::endl;
        }

        continuous_regions.erase(
            std::remove_if(continuous_regions.begin(), continuous_regions.end(),
                [](const auto& region) { 
                    return (region.second - region.first) < 100.0; 
                }),
            continuous_regions.end()
        );

        std::cout << "After filtering, " << continuous_regions.size() << " regions remain:" << std::endl;
        for (const auto& region : continuous_regions) {
            std::cout << "Region: [" << region.first << ", " << region.second << "]" << std::endl;
        }

        pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
        kdtree.setInputCloud(cloud);

        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr notchCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        notchCloud->width = 0;
        notchCloud->height = 1;
        notchCloud->is_dense = false;

        std::vector<std::pair<double, pcl::RGB>> notch_colors;

        if (continuous_regions.size() == 1) {
            std::vector<std::pair<double, int>> y_counts;
            const auto& region = continuous_regions[0];
            for (double y = region.first; y <= region.second; y += radius) {
                int count = 0;
                pcl::PointXYZRGB searchPoint;
                searchPoint.y = y;
                searchPoint.x = center_x;
                searchPoint.z = center_z;

                if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
                    count = pointIdxRadiusSearch.size();
                }
                y_counts.push_back({y, count});
            }

            if (y_counts.size() > 2) {  
                y_counts.erase(y_counts.begin());  
                y_counts.pop_back();               
            }
            std::sort(y_counts.begin(), y_counts.end(), [](auto& a, auto& b) { return a.second < b.second; });
            notch_colors = {
                {y_counts[0].first, {255, 0, 0}},
                {y_counts[1].first, {0, 255, 0}},
                {y_counts[2].first, {255, 255, 0}},
            };
        } else if (continuous_regions.size() == 2) {
            std::vector<std::vector<std::pair<double, int>>> region_y_counts(2);
            for (size_t region_idx = 0; region_idx < 2; ++region_idx) {
                const auto& region = continuous_regions[region_idx];
                for (double y = region.first; y <= region.second; y += radius) {
                    int count = 0;
                    pcl::PointXYZRGB searchPoint;
                    searchPoint.y = y;
                    searchPoint.x = center_x;
                    searchPoint.z = center_z;

                    if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
                        count = pointIdxRadiusSearch.size();
                        std::cout << "Region " << region_idx + 1 << " - Points at y=" << y << ": " << count << std::endl;
                    }
                    region_y_counts[region_idx].push_back({y, count});
                }
            }
            const std::vector<pcl::RGB> colors = {{255, 0, 0}, {0, 255, 0}};
            for (size_t region_idx = 0; region_idx < 2; ++region_idx) {
                auto& counts = region_y_counts[region_idx];
                if (!counts.empty()) {
                    std::sort(counts.begin(), counts.end(), 
                        [](const auto& a, const auto& b) { return a.second < b.second; });
                    counts.erase(counts.begin());
                    notch_colors.push_back({counts[0].first, colors[region_idx]});
                    std::cout << "Region " << region_idx + 1 << " minimum at y=" << counts[0].first 
                              << " with " << counts[0].second << " points" << std::endl;
                }
            }
        }

        std::vector<int> notch_point_counts;
        for (const auto& [y, color] : notch_colors) {
            int current_notch_count = 0;
            for (auto& point : cloud->points) {
                if (point.x >= center_x - radius && point.x <= center_x + radius &&
                    point.y >= y - radius && point.y <= y + radius) {
                    pcl::PointXYZRGB colored_point = point;
                    colored_point.r = color.r;
                    colored_point.g = color.g;
                    colored_point.b = color.b;
                    notchCloud->points.push_back(colored_point);
                    notchCloud->width++;
                    current_notch_count++;
                }
            }
            notch_point_counts.push_back(current_notch_count);
            std::cout << "Notch at y=" << y << " contains " << current_notch_count << " points" << std::endl;
        }
        std::cout << "Total extracted points: " << notchCloud->points.size() << std::endl;
        return notchCloud;
    }  
} 
