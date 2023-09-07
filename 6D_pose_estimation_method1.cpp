#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable:4996)
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <omp.h>

// RealSense headers
#include <librealsense2/rs.hpp>

// OpenCV headers
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

// PCL headers
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/crop_box.h>
#include <pcl/features/normal_3d.h>

// Eigen headers (might be already included with PCL, but just in case)
#define EIGEN_DONT_VECTORIZE 
#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT
#include <Eigen/Dense>

//Boost Headers
#include <boost/thread/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

// Custom Header file
#include "FEC.h"


const int NUM_THREADS = std::thread::hardware_concurrency();

struct CameraIntrinsics {
    double fx, fy, cx, cy;
};

float depth_scale = 0.001;

cv::Point clicked_point;
bool is_point_selected = false;

void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        clicked_point = cv::Point(x, y);
        is_point_selected = true;
    }
}

void drawCentroidOnImage(cv::Mat& image, const cv::Point3f& point) {
    cv::circle(image, cv::Point(point.x, point.y), 5, cv::Scalar(0, 0, 255), -1); // Red circle
    cv::putText(image, "Centroid", cv::Point(point.x + 10, point.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr createPointCloud(const cv::Mat& depth_img, const CameraIntrinsics& intrinsics) {
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

#pragma omp parallel for
for (int v = 0; v < depth_img.rows; ++v) {
    for (int u = 0; u < depth_img.cols; ++u) {
        float z = depth_img.at<ushort>(v, u) * depth_scale;
        if (z > 0) {
            float x = (u - intrinsics.cx) * z / intrinsics.fx;
            float y = (v - intrinsics.cy) * z / intrinsics.fy;
            cloud->push_back(pcl::PointXYZ(x, y, z));
        }
    }
}

    return cloud;
}

// RANSAC-related structures and functions

struct PlaneFitResult {
    Eigen::Vector4f equation;
    std::vector<int> inliers;
};
struct Result {
    Eigen::Vector4f eq;
    std::vector<int> inliers;
};
pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);

std::mutex mtx;

void ransac_thread(int iterations, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float thresh, Result& bestResult) {
    std::srand((unsigned)time(0));  // Seed the random number generator
    std::vector<Eigen::Vector3f> pt_samples(3);
    Eigen::Vector3f vecA, vecB, vecC;
    Eigen::MatrixXf points = cloud->getMatrixXfMap(3, 4, 0);  // Moved out of the loop

    for (int it = 0; it < iterations; ++it) {
        for (int s = 0; s < 3; ++s) {
            int idx = std::rand() % cloud->points.size();
            pcl::PointXYZ sampled_point = cloud->points[idx];
            pt_samples[s] = Eigen::Vector3f(sampled_point.x, sampled_point.y, sampled_point.z);  // Direct assignment
        }

        vecA = pt_samples[1] - pt_samples[0];
        vecB = pt_samples[2] - pt_samples[0];
        vecC = vecA.cross(vecB);  // No need to normalize here

        float k = -vecC.dot(pt_samples[1]);
        Eigen::Vector4f plane_eq(vecC[0], vecC[1], vecC[2], k);

        Eigen::Vector3f normal(plane_eq[0], plane_eq[1], plane_eq[2]);
        float d = plane_eq[3];

        Eigen::ArrayXf dists = (points.transpose() * normal).array() + d;
        dists = dists.abs() / normal.norm();

        std::vector<int> inliers;
        inliers.reserve(points.cols());  // Reserve memory

        for (int i = 0; i < dists.size(); ++i) {
            if (dists[i] <= thresh) {
                inliers.push_back(i);
            }
        }

        if (inliers.size() > bestResult.inliers.size()) {
            mtx.lock();
            bestResult.eq = plane_eq;
            bestResult.inliers = inliers;
            mtx.unlock();
        }
    }
}

PlaneFitResult fit(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float thresh = 1e-2, int maxIteration = 150) {
    std::vector<std::thread> threads;
    std::vector<Result> results(NUM_THREADS, {Eigen::Vector4f::Zero(), {}});

    int iterationsPerThread = maxIteration / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(ransac_thread, iterationsPerThread, cloud, thresh, std::ref(results[i]));
    }

    for (auto& t : threads) {
        t.join();
    }

    // Retrieve the best result
    Result best = results[0];
    for (int i = 1; i < NUM_THREADS; ++i) {
        if (results[i].inliers.size() > best.inliers.size()) {
            best = results[i];
        }
    }

    PlaneFitResult result;
    result.equation = best.eq;
    result.inliers = best.inliers;

    return result;
}

// generate cluster color randomly
int* rand_rgb() {
    int* rgb = new int[3];
    rgb[0] = rand() % 255;
    rgb[1] = rand() % 255;
    rgb[2] = rand() % 255;
    return rgb;
}

int main() 
{
    // Check if a RealSense device is connected
    rs2::context ctx;
    if (ctx.query_devices().size() == 0) {
        std::cerr << "No RealSense devices were found!" << std::endl;
        return -1;
    }

    rs2::pipeline pipe;
    rs2::config cfg;

    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);

    rs2::pipeline_profile profile = pipe.start(cfg);

    // Retrieve the intrinsic parameters for the depth stream
    rs2::video_stream_profile depth_stream_profile = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    const rs2_intrinsics& intrinsics_struct = depth_stream_profile.get_intrinsics();

    CameraIntrinsics intrinsics;
    intrinsics.fx = intrinsics_struct.fx;
    intrinsics.fy = intrinsics_struct.fy;
    intrinsics.cx = intrinsics_struct.ppx;
    intrinsics.cy = intrinsics_struct.ppy;

    rs2::align align_to(RS2_STREAM_COLOR);

    rs2::spatial_filter spat_filter;
    rs2::temporal_filter temp_filter;
    rs2::disparity_transform depth_to_disparity(true);
    rs2::disparity_transform disparity_to_depth(false);

    cv::namedWindow("Color Image", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("Color Image", onMouse, NULL);

    cv::Mat depth_img;  // Declare depth_img here
    
    // Declare the pointcloud object and points object
    rs2::pointcloud pc;
    rs2::points points;

    
    rs2::decimation_filter dec_filter; // Create the decimation filter
    dec_filter.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2);

    while (!is_point_selected) {
        rs2::frameset data = pipe.wait_for_frames();
        rs2::frameset aligned_data = align_to.process(data);

        rs2::frame depth_filtered = aligned_data.get_depth_frame();
        
        depth_filtered = dec_filter.process(depth_filtered);  // Apply the decimation filter here

        depth_filtered = depth_to_disparity.process(depth_filtered);
        depth_filtered = spat_filter.process(depth_filtered);
        depth_filtered = temp_filter.process(depth_filtered);
        depth_filtered = disparity_to_depth.process(depth_filtered);

        points = pc.calculate(depth_filtered);

        rs2::video_frame color = aligned_data.get_color_frame();

        cv::Mat color_img(cv::Size(color.get_width(), color.get_height()), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
        depth_img = cv::Mat(cv::Size(depth_filtered.as<rs2::depth_frame>().get_width(), depth_filtered.as<rs2::depth_frame>().get_height()), CV_16UC1, (void*)depth_filtered.get_data(), cv::Mat::AUTO_STEP);

        color_img.convertTo(color_img, -1, 1, 0); 
        cv::imshow("Color Image", color_img);

        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    auto start1 = std::chrono::high_resolution_clock::now();

    if (is_point_selected) {
        std::cout << "Selected point on color image: (" << clicked_point.x << ", " << clicked_point.y << ")" << std::endl;
       
        // Convert the RealSense SDK point cloud to PCL point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        auto vertices = points.get_vertices();
        for (int i = 0; i < points.size(); ++i) {
            if (vertices[i].z) {  // Check if the depth is valid
                cloud->push_back(pcl::PointXYZ(vertices[i].x, vertices[i].y, vertices[i].z));
            }
        }

        // Convert the clicked point to a 3D point
        float clicked_z = depth_img.at<ushort>(clicked_point.y, clicked_point.x) * depth_scale;
        float clicked_x = (clicked_point.x - intrinsics.cx) * clicked_z / intrinsics.fx;
        float clicked_y = (clicked_point.y - intrinsics.cy) * clicked_z / intrinsics.fy;
        pcl::PointXYZ clicked_3d_point(clicked_x, clicked_y, clicked_z);


        // auto start_time1 = std::chrono::high_resolution_clock::now();

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        
        auto start_time2 = std::chrono::high_resolution_clock::now();

        // Create a PassThrough filter instance
                pcl::PassThrough<pcl::PointXYZ> pass;
                pass.setInputCloud(cloud);
                pass.setFilterFieldName("z");
                pass.setFilterLimits(clicked_z - 0.30, clicked_z + 0.30); // PassThrough filter between the clicked_z (converted to meters) and clicked_z + 0.10 m.
                pass.filter(*cloud_filtered);

        auto end_time2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time2 - start_time2);
        std::cout << "[Pass Through Filter]: " << duration2.count() << "ms" << std::endl;


        auto start_time3 = std::chrono::high_resolution_clock::now();

        PlaneFitResult result = fit(cloud_filtered, 0.004, 600);
        Eigen::Vector4f bestPlane = result.equation;

        auto end_time3 = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time3 - start_time3);
        std::cout << "[Ransac Plane Segmentation]: " << duration3.count() << "ms" << std::endl;

        // Create a pcl::PointIndices from the inliers
        pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices());
        inliers_plane->indices = result.inliers;

        // Use pcl::ExtractIndices to remove the inliers from the cloud
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_filtered);
        extract.setIndices(inliers_plane);
        extract.setNegative(true);  // true means we want to get the points that are NOT in the indices list
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_without_inliers(new pcl::PointCloud<pcl::PointXYZ>);
        extract.filter(*cloud_without_inliers);

                
        // Extract the inliers cloud
        extract.setNegative(false);  // false means we want to get the points that ARE in the indices list
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_inliers(new pcl::PointCloud<pcl::PointXYZ>());
        extract.filter(*cloud_inliers);

        // Save the point cloud of inliers
        pcl::io::savePCDFileASCII("cloud_inliers.pcd", *cloud_inliers);
        std::cout << "Saved " << cloud_inliers->points.size() << " inliers data points to cloud_inliers.pcd." << std::endl;

        // Save the point cloud with Pass through and Voxel grid filter
        pcl::io::savePCDFileASCII("cloud_filtered.pcd", *cloud_filtered);
        std::cout << "Saved " << cloud_filtered->points.size() << " data points to cloud_filtered.pcd." << std::endl;
        std::cout <<"hello"<<endl;

        // Save the point cloud without the inliers
        pcl::io::savePCDFileASCII("cloud_without_inliers.pcd", *cloud_without_inliers);
        std::cout << "Saved " << cloud_without_inliers->points.size() << " data points to cloud_without_inliers.pcd." << std::endl;

        // Filter the cloud to extract a cube region around the selected point
        float half_side_length = 0.3;
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::CropBox<pcl::PointXYZ> boxFilter;
        boxFilter.setMin(Eigen::Vector4f(clicked_x - half_side_length, clicked_y - half_side_length, clicked_z - half_side_length, 1.0));
        boxFilter.setMax(Eigen::Vector4f(clicked_x + half_side_length, clicked_y + half_side_length, clicked_z + half_side_length, 1.0));
        boxFilter.setInputCloud(cloud_without_inliers);
        boxFilter.filter(*filtered_cloud);

        auto start_time4 = std::chrono::high_resolution_clock::now();

        // Now perform Fast Euclidean Clustering on filtered_cloud
        std::vector<pcl::PointIndices> cluster_indices;
        cluster_indices = FEC(filtered_cloud, 20, 0.015, 50);

        // Prepare to color the clusters
        std::vector<unsigned char> color;
        for (int i_segment = 0; i_segment < cluster_indices.size(); i_segment++) {
            color.push_back(static_cast<unsigned char>(rand() % 256));
            color.push_back(static_cast<unsigned char>(rand() % 256));
            color.push_back(static_cast<unsigned char>(rand() % 256));
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_point(new pcl::PointCloud<pcl::PointXYZRGB>());

        #pragma omp parallel for
        for (size_t i = 0; i < cluster_indices.size(); i++) {
            std::vector<pcl::PointXYZRGB> local_points;
            for (size_t j = 0; j < cluster_indices[i].indices.size(); j++) {
                pcl::PointXYZRGB point;
                point.x = filtered_cloud->points[cluster_indices[i].indices[j]].x;
                point.y = filtered_cloud->points[cluster_indices[i].indices[j]].y;
                point.z = filtered_cloud->points[cluster_indices[i].indices[j]].z;
                point.r = color[int(3) * i];
                point.g = color[int(3) * i + 1];
                point.b = color[int(3) * i + 2];
                local_points.push_back(point);
            }
            #pragma omp critical
            {
                color_point->insert(color_point->end(), local_points.begin(), local_points.end());
            }
        }

        // Save the colored clusters to a file
        pcl::io::savePCDFileASCII("color_clusters.pcd", *color_point);
        std::cout << "Saved " << color_point->points.size() << " data points to color_clusters.pcd." << std::endl;

        // Compute the centroids of all clusters and find the one closest to the clicked point
        std::vector<Eigen::Vector4f> centroids(cluster_indices.size());
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters(cluster_indices.size());

        #pragma omp parallel for
        for (size_t i = 0; i < cluster_indices.size(); i++) {
            clusters[i] = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
            for (size_t j = 0; j < cluster_indices[i].indices.size(); j++) {
                clusters[i]->points.push_back(filtered_cloud->points[cluster_indices[i].indices[j]]);
            }
            pcl::compute3DCentroid(*clusters[i], centroids[i]);
        }

        // Find the closest centroid to the clicked point
        float min_dist = FLT_MAX;
        int closest_cluster_idx = -1;
        for (size_t i = 0; i < centroids.size(); i++) {
            pcl::PointXYZ p1 = clicked_3d_point;
            pcl::PointXYZ p2 = pcl::PointXYZ(centroids[i][0], centroids[i][1], centroids[i][2]);
            float dist = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));

            if (dist < min_dist) {
                min_dist = dist;
                closest_cluster_idx = i;
            }
        }

        // Print the cluster the clicked point belongs to and its centroid
        std::cout << "The clicked point belongs to cluster " << closest_cluster_idx << std::endl;
        if (closest_cluster_idx != -1) {
            std::cout << "Centroid of the selected cluster: (" << centroids[closest_cluster_idx][0] << ", "
                    << centroids[closest_cluster_idx][1] << ", " << centroids[closest_cluster_idx][2] << ")" << std::endl;
        } else {
            std::cout << "No cluster is found for the selected point." << std::endl;
        }
        auto end_time4 = std::chrono::high_resolution_clock::now();
        auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time4 - start_time4);
        std::cout << "[Fast Eucledian Clustering and Centroid Selection]: " << duration4.count() << "ms" << std::endl;

        // Apply Region Growing to the selected cluster
        if (closest_cluster_idx != -1) {

            auto start_time5 = std::chrono::high_resolution_clock::now();

            std::shared_ptr<pcl::search::Search<pcl::PointXYZ>> tree = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
            pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
            pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
            normal_estimator.setSearchMethod(tree);
            normal_estimator.setInputCloud(clusters[closest_cluster_idx]);
            normal_estimator.setKSearch(50);
            normal_estimator.compute(*normals);

            pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
            reg.setMinClusterSize(30);
            reg.setMaxClusterSize(100000);
            reg.setSearchMethod(tree);
            reg.setNumberOfNeighbours(30);
            reg.setInputCloud(clusters[closest_cluster_idx]);
            reg.setInputNormals(normals);
            reg.setSmoothnessThreshold(8.0 / 180.0 * M_PI);
            reg.setCurvatureThreshold(2.0);

            std::vector<pcl::PointIndices> clusters_indices;
            reg.extract(clusters_indices);

            
            // Compute the centroids of all region-grown clusters
            std::vector<Eigen::Vector4f> rg_centroids(clusters_indices.size());
            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> rg_clusters(clusters_indices.size());

            #pragma omp parallel for
            for (size_t i = 0; i < clusters_indices.size(); i++) {
                rg_clusters[i] = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
                for (size_t j = 0; j < clusters_indices[i].indices.size(); j++) {
                    rg_clusters[i]->points.push_back(clusters[closest_cluster_idx]->points[clusters_indices[i].indices[j]]);
                }
                pcl::compute3DCentroid(*rg_clusters[i], rg_centroids[i]);
            }
           // Find the closest region-grown centroid to the clicked point
            float rg_min_dist = FLT_MAX;
            int closest_rg_cluster_idx = -1;
            for (size_t i = 0; i < rg_centroids.size(); i++) {
                pcl::PointXYZ p1 = clicked_3d_point;
                pcl::PointXYZ p2 = pcl::PointXYZ(rg_centroids[i][0], rg_centroids[i][1], rg_centroids[i][2]);
                float dist = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));

                if (dist < rg_min_dist) {
                    rg_min_dist = dist;
                    closest_rg_cluster_idx = i;
                }
            }


            auto end_time5 = std::chrono::high_resolution_clock::now();
            auto duration5 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time5 - start_time5);
            std::cout << "[Region Growing method]: " << duration5.count() << "ms" << std::endl;

            // Print the centroid of the largest region-grown cluster
            if (closest_rg_cluster_idx != -1) {
                std::cout << "The largest region-grown cluster is cluster " << closest_rg_cluster_idx << std::endl;
                std::cout << "Centroid of the largest region-grown cluster: (" << rg_centroids[closest_rg_cluster_idx][0] << ", "
                        << rg_centroids[closest_rg_cluster_idx][1] << ", " << rg_centroids[closest_rg_cluster_idx][2] << ")" << std::endl;
                
            }
            
            else {
                std::cout << "No region-grown cluster is found for the selected point." << std::endl;
            }
 
            // Display the normal of the RANSAC plane
            Eigen::Vector3f plane_normal(bestPlane[0], bestPlane[1], bestPlane[2]);

            // Check the direction of the normal
            if (plane_normal[1] < 0) { // If y-component of the normal is positive
                // Reverse the normal's direction to make it point downwards
                plane_normal = -plane_normal;
            }

            // Scaling the normal for better visualization
            float scale_factor = 2; // Adjust this value as needed
            plane_normal[0] *= scale_factor;
            plane_normal[1] *= scale_factor;
            plane_normal[2] *= scale_factor;

            // Visualization
            boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
            viewer->setBackgroundColor(1, 1, 1);

            // Add the inliers to the viewer with blue color
            if (cloud_inliers->points.size() > 0) {
                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> inliers_color(cloud_inliers, 0, 0, 255);  // Blue color for inliers
                viewer->addPointCloud<pcl::PointXYZ>(cloud_inliers, inliers_color, "cloud_inliers");
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_inliers");
            }

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();
            if (colored_cloud->points.size() > 0) {
                viewer->addPointCloud<pcl::PointXYZRGB>(colored_cloud, "region_growing");
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "region_growing");
            }
            
            // 1. RANSAC Plane Fitting
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

            pcl::SACSegmentation<pcl::PointXYZ> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(0.05);
            seg.setInputCloud(clusters[closest_rg_cluster_idx]);
            seg.segment(*inliers, *coefficients);

            if (inliers->indices.size() == 0) {
                PCL_ERROR("Could not estimate a planar model for the given dataset.");
            }

            // 2. Get the Unit Normal
            Eigen::Vector3f normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
            normal.normalize();

            // 3. Push the Centroid by 2 cm in the Direction of the Unit Normal
            float push_distance = 0.02; // 2 cm
            Eigen::Vector4f pushed_centroid = rg_centroids[closest_rg_cluster_idx] + push_distance * Eigen::Vector4f(normal[0], normal[1], normal[2], 0);

            // 4. Visualize the Pushed Centroid
            viewer->addSphere(pcl::PointXYZ(pushed_centroid[0], pushed_centroid[1], pushed_centroid[2]), 0.01, 1.0, 1.0, 0.0, "pushed_centroid_sphere");

            // Add the cluster's point cloud to the viewer
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cluster_color(rg_clusters[closest_rg_cluster_idx], 0, 255, 0); // Green color for the cluster
            viewer->addPointCloud<pcl::PointXYZ>(rg_clusters[closest_rg_cluster_idx], cluster_color, "closest_cluster");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "closest_cluster");

            pcl::PointXYZ start(pushed_centroid[0], pushed_centroid[1], pushed_centroid[2]);

            // Computing the orthogonal basis vectors
            Eigen::Vector3f v1 = normal.normalized(); // Ensure it's a unit vector
            Eigen::Vector3f g = Eigen::Vector3f(plane_normal[0], plane_normal[1], plane_normal[2]).normalized(); // Ensure it's a unit vector
            Eigen::Vector3f v2 = v1.cross(g).normalized(); // Ensure it's a unit vector

            // Visualizing the basis vectors
            float arrow_length = 0.5;  // You can adjust this as needed
            pcl::PointXYZ end_v1(start.x + v1[0] * arrow_length, start.y + v1[1] * arrow_length, start.z + v1[2] * arrow_length);
            pcl::PointXYZ end_g(start.x + g[0] * arrow_length, start.y + g[1] * arrow_length, start.z + g[2] * arrow_length);
            pcl::PointXYZ end_v2(start.x + v2[0] * arrow_length, start.y + v2[1] * arrow_length, start.z + v2[2] * arrow_length);

            viewer->addArrow(end_v1, start, 1.0, 0.0, 0.0, false, "v1_arrow");  // v1 in red
            viewer->addArrow(end_g, start, 1.0, 1.0, 0.0, false, "g_arrow");   // g in green
            viewer->addArrow(end_v2, start, 0.0, 0.0, 1.0, false, "v2_arrow"); // v2 in blue

            // Use spinOnce in a loop for continuous update
            while (!viewer->wasStopped()) {
                viewer->spinOnce(100); // Update every 100 milliseconds
                boost::this_thread::sleep(boost::posix_time::milliseconds(100));
            }
           std::cout<< "Plane Normal"<< plane_normal<< endl;
        }
    }

// End time
auto stop1 = std::chrono::high_resolution_clock::now();
auto duration6 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
std::cout << "Time taken by function: " << duration6.count() << " microseconds" << std::endl;
cv::destroyAllWindows();
pipe.stop();
return 0;
}
