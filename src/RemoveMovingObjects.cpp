#include "lio_sam/utility.hpp"
#include "lio_sam/msg/cloud_info.hpp"

class RemoveMovingObjects : public ParamServer
{
private:
    cv::Mat currentImage_, previousImage_;
    Eigen::Affine3f sensorPose_ = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
    pcl::RangeImage::CoordinateFrame coordinate_frame_ = pcl::RangeImage::LASER_FRAME;


    
public:
    RemoveMovingObjects(const rclcpp::NodeOptions & options) :
            ParamServer("lio_sam_imageProjection", options)
    {
         
  

    }


void pointCloudHandler(const lio_sam::msg::CloudInfo::SharedPtr msgIn){

}

void convertPointCloudToRangeImage(pcl::PointCloud<pcl::PointXYZ> &pointCloud,
                                   cv::Mat& rangeImage_out)
{
    if (!rangeImage_out.empty())
        rangeImage_out.release();
    rangeImage_out.create(MAXHEIGHT, MAXWIDTH, CV_32F);
    // height is w 
    #pragma omp parallel for
    for (auto &&point : pointCloud)
    {
        float R, omega, alpha;
        R = sqrt(pow(point.x,2) + pow(point.y,2) + pow(point.z,2));
        omega = -(asin(point.z/R) - (15 * TORADAIAN));
        alpha = atan2(point.x,point.y);
        int y = omega / ANGULARRESOLUTION_Y, x = alpha/ANGULARRESOLUTION_X;
        rangeImage_out.at<float>(x, y) = R;
    }
    
}



void applyMedianFilter(const cv::Mat& inputImage, cv::Mat& outputImage, int kernelSize)
{
    // Apply median filter
    cv::medianBlur(inputImage, outputImage, kernelSize);
}

void estimateRangeFlow(cv::Mat& rangeImage1, cv::Mat& rangeImage2, vector<cv::Mat> rangeFlow)
{
    // Calculate spatial-temporal gradients
    cv::Mat dx, dy, dR;
    cv::Sobel(rangeImage1, dx, CV_32F, 1, 0, 1);
    cv::Sobel(rangeImage1, dy, CV_32F, 0, 1, 1);
    cv::subtract(rangeImage2, rangeImage1, dR);
    rangeFlow = {dx, dy, dR};
}



};



#include <pcl/range_image/range_image.h>

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor exec;

    auto IP = std::make_shared<RemoveMovingObjects>(options);
    exec.add_node(IP);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Remove Moving Objects Started.\033[0m");

    exec.spin();

    rclcpp::shutdown();
    return 0;
}


