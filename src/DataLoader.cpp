#include "DataLoader.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <iostream>

pcl::PointCloud<pcl::PointXYZRGB>::Ptr loadRealPointCloud(const std::string& filePath) {
    const float NULLVALUE = -999, SCALEX = 0.006, SCALEY = 0.1;
    
    // Real Data image load
    auto realDepthMap = cv::imread(filePath, cv::IMREAD_UNCHANGED);
    if (realDepthMap.empty()){
        std::cerr << "Error loading image: " << filePath << std::endl;
        return pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    }
    int width = realDepthMap.cols;
    int height = realDepthMap.rows;
    auto imgPtr = realDepthMap.ptr<float>(0);
    
    // PointCloud 
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr realCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    for (int j = 0; j < height; ++j) {
        int step = width * j;
        for (int i = 0; i < width; ++i) {
            if (imgPtr[step + i] != NULLVALUE) {
                pcl::PointXYZRGB pclPoint;
                pclPoint.x = SCALEX * i;
                pclPoint.y = SCALEY * j;
                pclPoint.z = -imgPtr[step + i];
                pclPoint.r = 255;
                pclPoint.g = 255;
                pclPoint.b = 255;
                realCloud->push_back(pclPoint);
            }
        }
    }
    return realCloud;
}

void transformAndColorPointCloud(const cv::Mat& image, 
                                 pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, 
                                 const cv::Mat& transformation) {
    if (image.type() == CV_32FC3) {
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                cv::Vec3f point = image.at<cv::Vec3f>(y, x);
                if (point[2] != -999) {
                    cv::Mat srcPt = (cv::Mat_<double>(4, 1) << point[0], point[1], point[2], 1);
                    cv::Mat dstPt = transformation * srcPt;
                    pcl::PointXYZRGB pclPoint;
                    pclPoint.x = dstPt.at<double>(2);
                    pclPoint.y = dstPt.at<double>(1);
                    pclPoint.z = dstPt.at<double>(0);
                    pclPoint.r = 255; // 초기 색상: 흰색
                    pclPoint.g = 255;
                    pclPoint.b = 255;
                    cloud->push_back(pclPoint);
                }
            }
        }
    }
}

void downsamplePointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, 
                          pcl::PointCloud<pcl::PointXYZRGB>::Ptr& downsampledCloud, 
                          float leafSize) {
    pcl::VoxelGrid<pcl::PointXYZRGB> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(leafSize, leafSize, leafSize);
    vg.filter(*downsampledCloud);
}
