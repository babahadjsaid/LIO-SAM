#include "lio_sam/utility.hpp"
#include "lio_sam/msg/cloud_info.hpp"

class RemoveMovingObjects : public ParamServer
{
private:
    cv::Mat currentImage_, previousImage_;
    Eigen::Affine3f sensorPose_ = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
    pcl::RangeImage::CoordinateFrame coordinate_frame_ = pcl::RangeImage::LASER_FRAME;
    rclcpp::Subscription<lio_sam::msg::CloudInfo>::SharedPtr subCloud;
    rclcpp::Publisher<lio_sam::msg::CloudInfo>::SharedPtr pubLaserCloudInfo;
    std::mutex mapMtx_;
public:
    RemoveMovingObjects(const rclcpp::NodeOptions & options) :
            ParamServer("lio_sam_RemoveMovingObject", options)
    {
         
        subCloud = create_subscription<lio_sam::msg::CloudInfo>(
            "lio_sam/deskew/cloud_info", qos,
            std::bind(&RemoveMovingObjects::pointCloudHandler, this, std::placeholders::_1));
        pubLaserCloudInfo = create_publisher<lio_sam::msg::CloudInfo>(
            "lio_sam/RemoveMovingObjects/cloud_info", qos);
    
    }


void pointCloudHandler(const lio_sam::msg::CloudInfo::SharedPtr msgIn){
    mapMtx_.lock();
    const auto StartMethod = std::chrono::system_clock::now();
    pcl::PointCloud<pcl::PointXYZI> pc;
    pcl::fromROSMsg(msgIn->cloud_deskewed,pc);

    convertPointCloudToRangeImage(pc,currentImage_);
    cv::Mat dst;
    cv::normalize(currentImage_, dst, 0, 2, cv::NORM_MINMAX);
    // cv::namedWindow("Hello world",cv::WINDOW_FULLSCREEN);
    cv::imshow("test", dst);
    const std::chrono::duration<double> duration = std::chrono::system_clock::now() - StartMethod;
    float durationms = 100 - 1000 * duration.count();
    std::cout <<"delay: "<< 100 - durationms <<std::endl;
    dst.release();
    currentImage_.release();
    cv::waitKey(50);
    mapMtx_.unlock();
}

void convertPointCloudToRangeImage(pcl::PointCloud<pcl::PointXYZI> &pointCloud,
                                   cv::Mat& rangeImage_out)
{
    if (!rangeImage_out.empty())
        rangeImage_out.release();
    rangeImage_out.create(MAXHEIGHT, MAXWIDTH, CV_32F);
    // height is w 
    #pragma omp parallel for num_threads(5)
    for (auto &&point : pointCloud)
    {
        float R, omega, alpha,x=point.y,y = -point.x;// +90 degree rotation yaw
        R = sqrt(pow(x,2) + pow(y,2) + pow(point.z,2));
        omega = -(asin(point.z/R) - (19.1 * TORADAIAN));
        alpha = atan2(y,x);
        if (alpha < 0) alpha +=2*M_PI;
        int i = alpha/ANGULARRESOLUTION_X , j = omega / ANGULARRESOLUTION_Y;
        if (i<0 || j <0 || i> MAXWIDTH || j > MAXHEIGHT) std::cout<<"i: "<<i<<" j: "<<j<<" alpha: "<<asin(point.z/R)*(180.0/M_PI)<<std::endl;
        #pragma omp critical
        {
            rangeImage_out.at<float>(j, i) = R;
        }
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


