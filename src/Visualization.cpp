#include "Visualization.hpp"
#include "DataLoader.hpp" 
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>

void visualizePointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
    pcl::visualization::PCLVisualizer viewer("Point Cloud Viewer");
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer.addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "cloud");
    // XYZ 축 추가
    viewer.addLine(pcl::PointXYZ(0, 0, 0), pcl::PointXYZ(100, 0, 0), 255, 0, 0, "X-Axis");
    viewer.addLine(pcl::PointXYZ(0, 0, 0), pcl::PointXYZ(0, 100, 0), 0, 255, 0, "Y-Axis");
    viewer.addLine(pcl::PointXYZ(0, 0, 0), pcl::PointXYZ(0, 0, 100), 0, 0, 255, "Z-Axis");
    viewer.addCoordinateSystem(1.0);
    viewer.setBackgroundColor(0, 0, 0);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
}

void visualizeRegistration(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &originalCloud,
                             const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &toTransformCloud,
                             const Eigen::Matrix4f &transformationMatrix) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampledOriginal(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampledTransform(new pcl::PointCloud<pcl::PointXYZRGB>());
    downsamplePointCloud(originalCloud, downsampledOriginal, 0.03f);
    downsamplePointCloud(toTransformCloud, downsampledTransform, 0.03f);
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformedCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::transformPointCloud(*downsampledTransform, *transformedCloud, transformationMatrix);
    
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> redHandler(downsampledOriginal, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZRGB>(downsampledOriginal, redHandler, "originalCloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "originalCloud");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> blueHandler(transformedCloud, 0, 0, 255);
    viewer->addPointCloud<pcl::PointXYZRGB>(transformedCloud, blueHandler, "transformedCloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "transformedCloud");
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
}

void visualizeClusters(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster1, 
                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster2) {
    pcl::visualization::PCLVisualizer viewer("Cluster Visualization");
    viewer.addLine(pcl::PointXYZ(0,0,0), pcl::PointXYZ(100,0,0), 255, 0, 0, "X-Axis");
    viewer.addLine(pcl::PointXYZ(0,0,0), pcl::PointXYZ(0,100,0), 0, 255, 0, "Y-Axis");
    viewer.addLine(pcl::PointXYZ(0,0,0), pcl::PointXYZ(0,0,100), 0, 0, 255, "Z-Axis");
    viewer.addCoordinateSystem(1.0);
    viewer.setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> blue(cluster1);
    viewer.addPointCloud<pcl::PointXYZRGB>(cluster1, blue, "Cluster 1 (Blue)");
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> red(cluster2);
    viewer.addPointCloud<pcl::PointXYZRGB>(cluster2, red, "Cluster 2 (Red)");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Cluster 1 (Blue)");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Cluster 2 (Red)");
    viewer.addCoordinateSystem(1.0);
    viewer.initCameraParameters();
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
}

void visualizeFinal(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &originalCloud,
                    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &toTransformCloud) {
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> redHandler(originalCloud, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZRGB>(originalCloud, redHandler, "originalCloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "originalCloud");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> blueHandler(toTransformCloud, 0, 0, 255);
    viewer->addPointCloud<pcl::PointXYZRGB>(toTransformCloud, blueHandler, "transformedCloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "transformedCloud");
    viewer->addCoordinateSystem(100.0);
    viewer->addLine(pcl::PointXYZ(0,0,0), pcl::PointXYZ(100,0,0), 255, 0, 0, "X-Axis");
    viewer->addLine(pcl::PointXYZ(0,0,0), pcl::PointXYZ(0,100,0), 0, 255, 0, "Y-Axis");
    viewer->addLine(pcl::PointXYZ(0,0,0), pcl::PointXYZ(0,0,100), 0, 0, 255, "Z-Axis");
    viewer->initCameraParameters();
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
}
