#include <iostream>
#include <string>
#include "DataLoader.hpp"
#include "DataPreprocessing.hpp"
#include "Visualization.hpp"
#include "NotionDetection.hpp"
#include "Alignment.hpp"
#include "Postprocessing.hpp"

int main(int argc, char** argv) {
    // Code expect at least 3 arguments: program, reference_path, real_path.
    // If visualization flags are not provided, default: vis1-vis10 = false and vis11 = true.
    std::string refPath, realDataPath;
    bool vis1, vis2, vis3, vis4, vis5, vis6, vis7, vis8, vis9, vis10, vis11;
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <reference_path> <real_path> [<vis1> ... <vis11>]" << std::endl;
        return -1;
    }
    
    refPath = argv[1];
    realDataPath = argv[2];
    
    if (argc < 14) {
        // Default: all visualization flags false except Final Notch Alignment (vis11)
        vis1 = vis2 = vis3 = vis4 = vis5 = vis6 = vis7 = vis8 = vis9 = vis10 = false;
        vis11 = true;
        std::cout << "Visualization flags not fully provided. Defaulting: Only Final Notch Alignment (vis11) = true." << std::endl;
    } else {
        vis1  = (std::string(argv[3])  == "true");
        vis2  = (std::string(argv[4])  == "true");
        vis3  = (std::string(argv[5])  == "true");
        vis4  = (std::string(argv[6])  == "true");
        vis5  = (std::string(argv[7])  == "true");
        vis6  = (std::string(argv[8])  == "true");
        vis7  = (std::string(argv[9])  == "true");
        vis8  = (std::string(argv[10]) == "true");
        vis9  = (std::string(argv[11]) == "true");
        vis10 = (std::string(argv[12]) == "true");
        vis11 = (std::string(argv[13]) == "true");
    }
    
    // =======================================================
    // Process Reference Data
    // =======================================================
    std::cout << "****** Processing Reference Data ******" << std::endl;
    
    // Load the reference image using OpenCV
    cv::Mat refImage = cv::imread(refPath, cv::IMREAD_UNCHANGED);
    if(refImage.empty()){
        std::cerr << "Error: Could not load reference image from " << refPath << std::endl;
        return -1;
    }
    // Create a point cloud for the reference data
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr refCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    // Set the transformation matrix based on the reference type
    cv::Mat refTransform;
    if (refPath.find("MASTER_BOTTOM.tiff") != std::string::npos) {
        // For Bottom reference
        refTransform = (cv::Mat_<double>(4, 4) <<
                     1, 0, 0, 0,
                     0, -1, 0, 537,
                     0, 0, -1, 2,
                     0, 0, 0, 1);
        std::cout << "****** [Reference - BOTTOM] Using Bottom transformation." << std::endl;
    }
    else if (refPath.find("MASTER_SIDE.tiff") != std::string::npos) {
        // For Side reference
        refTransform = (cv::Mat_<double>(4, 4) <<
                     0, 0, 1, 0,
                     0, -1, 0, 537,
                     1, 0, 0, -3,
                     0, 0, 0, 1);
        std::cout << "****** [Reference - SIDE] Using Side transformation." << std::endl;
    }
    else if (refPath.find("MASTER_TOP.tiff") != std::string::npos) {
        // For Top reference
        refTransform = (cv::Mat_<double>(4, 4) <<
                     -1, 0, 0, 7,
                     0, -1, 0, 537,
                     0, 0, 1, 0,
                     0, 0, 0, 1);
        std::cout << "****** [Reference - TOP] Using Top transformation." << std::endl;
    }
    else {
        // Default transformation
        refTransform = cv::Mat::eye(4, 4, CV_64F);
        std::cout << "****** [Reference] Using default transformation." << std::endl;
    }
    
    std::cout << "Reference Transformation" << std::endl;
    transformAndColorPointCloud(refImage, refCloud, refTransform);
    
    std::cout << "Reference Down Sampling" << std::endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampledCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    downsamplePointCloud(refCloud, downsampledCloud, 0.05);
    
    // Visualization1. Downsampled data (Reference)
    if (vis1) {
        std::cout << "[Vis1] Displaying Downsampled Reference Data" << std::endl;
        visualizePointCloud(downsampledCloud);
    }
    
    std::cout << "Reference Detection for Notch" << std::endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr referenceNotches = detectNotches(downsampledCloud, refPath);
    // Visualization2. Reference Notch Detection
    if (vis2) {
        std::cout << "[Vis2] Displaying Reference Notch Detection" << std::endl;
        visualizePointCloud(referenceNotches);
    }
    
    // =======================================================
    // Process Real Data
    // =======================================================
    std::cout << "****** Real Data Loader" << std::endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr realCloud = loadRealPointCloud(realDataPath);
    if(realCloud->empty()){
        std::cerr << "Error: Could not load real data from " << realDataPath << std::endl;
        return -1;
    }
    // Visualization3. Real Data (Original)
    if (vis3) {
        std::cout << "[Vis3] Displaying Original Real Data" << std::endl;
        visualizePointCloud(realCloud);
    }
    
    std::cout << "Postprocessing" << std::endl;
    // Process real data differently based on the file type
    if (realDataPath.find("REAL_SIDE.tiff") != std::string::npos) {
        // ----------------------- REAL SIDE processing
        std::cout << "[REAL SIDE] Processing REAL SIDE data..." << std::endl;
        std::cout << "Visualization for the ReferenceNotch - X Clustering" << std::endl;
        auto [clusterX1, clusterX2] = splitReferenceNotchesByX(referenceNotches);
        
        // Visualization4. Reference Notch Clustering By X-axis
        if (vis4) {
            std::cout << "[Vis4] (SIDE) Displaying Reference Notch Clustering by X-Axis." << std::endl;
            visualizeClusters(clusterX1, clusterX2);
        }
        auto [clusterZ1, clusterZ2] = splitClusterByZ(clusterX2);
        // Visualization5. Reference Notch Clustering By Z-axis After X-Axis
        if (vis5) {
            std::cout << "[Vis5] (SIDE) Displaying Reference Notch Clustering by Z-Axis (after X-Axis)." << std::endl;
            visualizeClusters(clusterZ1, clusterZ2);
        }
        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr Postprocessing = removeSmallComponents(realCloud);
        // Visualization6. Real Data Preprocessing (Remove Small Area)
        if (vis6) {
            std::cout << "[Vis6] (SIDE) Displaying Postprocessed Real Data (small areas removed)." << std::endl;
            visualizePointCloud(Postprocessing);
        }
        std::cout << "Real Notch Detection" << std::endl;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr realNotch = RealdetectNotches(Postprocessing, realDataPath);
        std::cout << "Real Notch Size: " << realNotch->size() << std::endl;
        // Visualization7. Real Data Notch Detection
        if (vis7) {
            std::cout << "[Vis7] (SIDE) Displaying Detected Real Notches." << std::endl;
            visualizePointCloud(realNotch);
        }
        
        std::cout << "****** Matching - Initial Alignment (SIDE)" << std::endl;
        Eigen::Matrix4f InitialTransformationMatrix = InitialAlignment(realNotch, Postprocessing, referenceNotches, downsampledCloud);
        // Apply transformation for Notch
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr rotatedRealNotch = transformPointCloud(realNotch, InitialTransformationMatrix);
        // Visualization8. Initial Rotation using Reference Notch and Real Notch (SIDE)
        if (vis8) {
            std::cout << "[Vis8] (SIDE) Displaying Initial Rotation (Reference vs. Rotated Real Notch)." << std::endl;
            visualizeFinal(referenceNotches, rotatedRealNotch);
        }
        
        auto [RclusterZ1_side, RclusterZ2_side] = splitClusterByZ(rotatedRealNotch);
        // Visualization9. Second Rotation using Reference Notch and Real Notch (SIDE)
        if (vis9) {
            std::cout << "[Vis9] (SIDE) Displaying Clustering Second Rotation (using Z-Axis clustering)." << std::endl;
            visualizeClusters(RclusterZ1_side, RclusterZ2_side);
        }
        Eigen::Matrix4f SecondTransformationMatrix = SecondAlignment(RclusterZ2_side, clusterZ2);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr SecondrotatedRealNotch = transformPointCloud(rotatedRealNotch, SecondTransformationMatrix);
        if (vis9) {
            std::cout << "[Vis9] (SIDE) Displaying Second Rotation (Reference vs. Second Rotated Real Notch)." << std::endl;
            visualizeFinal(referenceNotches, SecondrotatedRealNotch);
        }
        
        std::cout << "*****************************" << std::endl;
        std::cout << "Final Rotation (SIDE)" << std::endl;
        auto [RclusterX1_side, RclusterX2_side] = splitReferenceNotchesByX(SecondrotatedRealNotch);
        std::cout << "X Clustering for Real Notch (SIDE)" << std::endl;
        // Visualization10. Real Notch Clustering using X-Axis After Z-Axis (SIDE)
        if (vis10) {
            std::cout << "[Vis10] (SIDE) Displaying X-Axis clustering after Z-Axis clustering." << std::endl;
            visualizeClusters(RclusterX1_side, RclusterX2_side);
        }
        
        Eigen::Matrix4f translationMat_side = computeTranslationMatrix(RclusterX2_side, clusterZ2);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr FinalRealNotch_side = transformPointCloud(SecondrotatedRealNotch, translationMat_side);
        std::cout << "*****************************" << std::endl;
        std::cout << "Translation (SIDE)" << std::endl;
        // Visualization11. Final Notch Alignment (SIDE)
        if (vis11) {
            std::cout << "[Vis11] (SIDE) Displaying Final Notch Alignment after translation." << std::endl;
            visualizeFinal(referenceNotches, FinalRealNotch_side);
        }
        
        // ---------------------------------------------------------
        // Compute and output the overall transformation between origin and real data.
        // The overall alignment transformation (mapping real data to origin) is:
        //     T_align = translationMat_side * SecondTransformationMatrix * InitialTransformationMatrix
        // Therefore, the transformation from origin to real data is:
        //     T_final = T_align^(-1)
        Eigen::Matrix4f overallTransformation = translationMat_side * SecondTransformationMatrix * InitialTransformationMatrix;
        Eigen::Matrix4f finalOriginToReal = overallTransformation.inverse();
        std::cout << "Final Transformation (Origin to Real Data) [SIDE]:" << std::endl;
        std::cout << finalOriginToReal << std::endl;
    }
    else if (realDataPath.find("REAL_TOP.tiff") != std::string::npos) {
        // ----------------------- REAL TOP processing
        std::cout << "[Real - TOP] Processing REAL TOP data..." << std::endl;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredRealCloud = filterSmallXClusters(realCloud);
        if (vis4) {
            std::cout << "[Vis4] (TOP) Preprocessing Step 1: Filtering using X-Axis clustering." << std::endl;
            visualizePointCloud(filteredRealCloud);
        }
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr PreprocessedData = filterTop2YSegments(filteredRealCloud, 1.0);
        if (vis5) {
            std::cout << "[Vis5] (TOP) Preprocessing Step 2: Filtering using Y-Axis." << std::endl;
            visualizePointCloud(PreprocessedData);
        }
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr Postprocessing = removeSmallComponents(PreprocessedData);
        if (vis6){
            std::cout << "[Vis6] (TOP) Remove Small Area." << std::endl;
            visualizePointCloud(Postprocessing);
        }
        
        std::cout << "Real Notch Detection" << std::endl;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr realNotch = RealdetectNotches(Postprocessing, realDataPath);

        if (vis7){
            std::cout << "[Vis7] (TOP) Real Notch Detection." << std::endl;
            visualizePointCloud(realNotch);
        }


        std::cout << "****** Matching - Initial Alignment (TOP)" << std::endl;
        Eigen::Matrix4f InitialTransformationMatrix = InitialAlignment(realNotch, Postprocessing, referenceNotches, downsampledCloud);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr rotatedRealNotch = transformPointCloud(realNotch, InitialTransformationMatrix);
        
        if (vis8) {
            std::cout << "[Vis8] (TOP) Displaying Initial Rotation (Reference vs. Rotated Real Notch)." << std::endl;
            visualizeFinal(referenceNotches, rotatedRealNotch);
        }

        std::cout << "[Real - TOP] Performing second alignment based on XZ clustering." << std::endl;        
        auto [clusterX1_top, clusterX2_top] = splitReferenceNotchesByX(referenceNotches);
        auto [RclusterXZ1, RclusterXZ2] = splitReferenceNotchesByXZ(rotatedRealNotch);

        Eigen::Matrix4f SecondTransformationMatrix = SecondAlignment(RclusterXZ2, clusterX2_top);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr SecondrotatedRealNotch = transformPointCloud(rotatedRealNotch, SecondTransformationMatrix);
        if (vis9){
            std::cout << "[Vis9] (TOP) Displaying Second Rotation (Reference vs. Rotated Real Notch)." << std::endl;
            visualizeFinal(referenceNotches, SecondrotatedRealNotch);
        }
        
        // ********************************************************************
        // ********************************************************************
        std::cout << "Final Rotation - Cluster " <<std::endl;
        auto [RclusterXZ_rotation1, RclusterXZ_rotation2] = splitReferenceNotchesByXZ(SecondrotatedRealNotch);
        
        std::cout << "[Real - TOP] Performing final translation." << std::endl;
        Eigen::Matrix4f translationMat = computeTranslationMatrixTop(RclusterXZ_rotation2, clusterX2_top);
        if (vis10){
            std::cout << "[Vis10] (TOP) Real Notch clustering for Reference and Real ." << std::endl;
            visualizeClusters(RclusterXZ_rotation1, RclusterXZ_rotation2);
            visualizeClusters(clusterX1_top, clusterX2_top);
        
        }
        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr FinalRealNotch = transformPointCloud(SecondrotatedRealNotch, translationMat);
        if (vis11) {
            std::cout << "[Vis11] (TOP) Displaying Final Notch Alignment after translation." << std::endl;
            visualizeFinal(referenceNotches, FinalRealNotch);
        }
        
        // Compute overall transformation for TOP:
        Eigen::Matrix4f overallTransformation = translationMat * SecondTransformationMatrix * InitialTransformationMatrix;
        Eigen::Matrix4f finalOriginToReal = overallTransformation.inverse();
        std::cout << "Final Transformation (Origin to Real Data) [TOP]:" << std::endl;
        std::cout << finalOriginToReal << std::endl;
        
    }
    else {
        // ----------------------- REAL BOTTOM processing (default branch)
        std::cout << "[Real - BOTTOM] Processing REAL BOTTOM data..." << std::endl;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredRealCloud = filterSmallXClusters(realCloud);
        if (vis4) {
            std::cout << "[Vis4] (BOTTOM) Displaying Filtered Real Data." << std::endl;
            visualizePointCloud(filteredRealCloud);
        }
        std::cout << "Preprocessing 2" << std::endl;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr PreprocessedData = filterTop2YSegments(filteredRealCloud, 1.0);
        
        if (vis5) {
            std::cout << "[Vis5] (BOTTOM) Preprocessing Step 2: Filtering using Y-Axis." << std::endl;
            visualizePointCloud(PreprocessedData);
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr Postprocessing = removeSmallComponents(PreprocessedData);
        if (vis6) {
            std::cout << "[Vis6] (BOTTOM) Rmove the small area" << std::endl;
            visualizePointCloud(Postprocessing);
        }
        std::cout << "Real Notch Detection (BOTTOM)" << std::endl;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr realNotch = RealdetectNotches(Postprocessing, realDataPath);
        if (vis7) {
            std::cout << "[Vis7] (BOTTOM) Real Notch Detection." << std::endl;
            visualizePointCloud(realNotch);
        }
        
        std::cout << "****** Matching - Initial Alignment (BOTTOM)" << std::endl;
        Eigen::Matrix4f InitialTransformationMatrix = InitialAlignment(realNotch, Postprocessing, referenceNotches, downsampledCloud);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr rotatedRealNotch = transformPointCloud(realNotch, InitialTransformationMatrix);
        
        if (vis8) {
            std::cout << "[Vis8] (BOTTOM) Displaying Initial Rotation (Reference vs. Rotated Real Notch" << std::endl;
            visualizeFinal(referenceNotches, rotatedRealNotch);
        }
        std::cout << "*****************************" << std::endl;
        std::cout << "Final Rotation (BOTTOM)" << std::endl;
        Eigen::Matrix4f SecondTransformationMatrix = Eigen::Matrix4f::Identity();
        SecondTransformationMatrix << 
                                    -1,  0,  0, 0,   // X-axis inversion
                                     0,  1,  0, 0,   // Y-axis remains
                                     0,  0,  1, 0,   // Z-axis remains
                                     0,  0,  0, 1;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr SecondrotatedRealNotch = transformPointCloud(rotatedRealNotch, SecondTransformationMatrix);
        
        if (vis9) {
            std::cout << "[Vis9] (BOTTOM) Displaying Second Rotation (Reference vs. Rotated Real Notch" << std::endl;
            visualizeFinal(referenceNotches, SecondrotatedRealNotch);
        }
        std::cout << "*****************************" << std::endl;
        std::cout << "Translation (BOTTOM)" << std::endl;
        Eigen::Matrix4f translationMat = computeTranslationMatrixBottom(SecondrotatedRealNotch, referenceNotches);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr FinalRealNotch = transformPointCloud(SecondrotatedRealNotch, translationMat);
        
        if (vis11) {
            std::cout << "[Vis11] (BOTTOM) Executing custom visualization step 11." << std::endl;
            visualizeFinal(referenceNotches, FinalRealNotch);
        }
        
        // Compute overall transformation for BOTTOM:
        Eigen::Matrix4f overallTransformation = translationMat * SecondTransformationMatrix * InitialTransformationMatrix;
        Eigen::Matrix4f finalOriginToReal = overallTransformation.inverse();
        std::cout << "Final Transformation (Origin to Real Data) [BOTTOM]:" << std::endl;
        std::cout << finalOriginToReal << std::endl;
    }
    
    return 0;
}
