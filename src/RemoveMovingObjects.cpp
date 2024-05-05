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
    cv::normalize(currentImage_, dst, 0, 3, cv::NORM_MINMAX);
    // cv::namedWindow("Hello world",cv::WINDOW_FULLSCREEN);
    cv::imshow("test", dst);
    cv::imwrite("image.png",dst);
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
    rangeImage_out.create(32, MAXWIDTH, CV_32F);
    // height is w 
    #pragma omp parallel for num_threads(5)
    for (auto &&point : pointCloud)
    {
        float R, omega, alpha;
        R = sqrt(pow(point.x,2) + pow(point.y,2) + pow(point.z,2));
        omega = asin(point.z/R);
        alpha = atan2(point.y,point.x);
        if (alpha < 0) alpha +=2*M_PI;
        int i = alpha/ANGULARRESOLUTION_X , j = checkRing(omega);
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



int checkRing(double angle) {
    if (angle >= (15 * TORADAIAN))                                   return 0;
    if (angle < (15 * TORADAIAN) && angle >= (13 * TORADAIAN))       return 1;
    if (angle < (13 * TORADAIAN) && angle >= (11 * TORADAIAN))       return 2;
    if (angle < (11 * TORADAIAN) && angle >= (9 * TORADAIAN))        return 3;
    if (angle < (9 * TORADAIAN) && angle  >= (7 * TORADAIAN))        return 4;
    if (angle < (7 * TORADAIAN) && angle  >= (5.5 * TORADAIAN))      return 5;
    if (angle < (5.5 * TORADAIAN) && angle >= (4 * TORADAIAN))       return 6;
    if (angle < (4 * TORADAIAN) && angle >= (2.67 * TORADAIAN))      return 7;
    if (angle < (2.67 * TORADAIAN) && angle >= (1.33 * TORADAIAN))   return 8;
    if (angle < (1.33 * TORADAIAN) && angle >= 0)                    return 9;
    if (angle < 0 && angle >= (-1.33 * TORADAIAN))                   return 10;
    if (angle < (-1.33 * TORADAIAN) && angle >= (-2.67 * TORADAIAN)) return 11;
    if (angle < (-2.67 * TORADAIAN) && angle >= (-4 * TORADAIAN))    return 12;
    if (angle < (-4 * TORADAIAN) && angle >= (-5.33 * TORADAIAN))    return 13;
    if (angle < (-5.33 * TORADAIAN) && angle >= (-6.67 * TORADAIAN)) return 14;
    if (angle < (-6.67 * TORADAIAN) && angle >= (-8 * TORADAIAN))    return 15;
    if (angle < (-8 * TORADAIAN) && angle >= (-10 * TORADAIAN))      return 16;
    if (angle < (-10 * TORADAIAN) && angle >= (-16 * TORADAIAN))     return 17;
    if (angle < (-16 * TORADAIAN) && angle >= (-13 * TORADAIAN))     return 18;
    if (angle < (-13 * TORADAIAN) && angle >= (-19 * TORADAIAN))     return 19;
    if (angle < (-19 * TORADAIAN) && angle >= (-22 * TORADAIAN))     return 20;
    if (angle < (-22 * TORADAIAN) && angle >= (-28 * TORADAIAN))     return 21;
    if (angle < (-28 * TORADAIAN) && angle >= (-25 * TORADAIAN))     return 22;
    if (angle < (-25 * TORADAIAN) && angle >= (-31 * TORADAIAN))     return 23;
    if (angle < (-31 * TORADAIAN) && angle >= (-34 * TORADAIAN))     return 24;
    if (angle < (-34 * TORADAIAN) && angle >= (-37 * TORADAIAN))     return 25;
    if (angle < (-37 * TORADAIAN) && angle >= (-40 * TORADAIAN))     return 26;
    if (angle < (-40 * TORADAIAN) && angle >= (-43 * TORADAIAN))     return 27;
    if (angle < (-43 * TORADAIAN) && angle >= (-46 * TORADAIAN))     return 28;
    if (angle < (-46 * TORADAIAN) && angle >= (-49 * TORADAIAN))     return 29;
    if (angle < (-49 * TORADAIAN) && angle >= (-52 * TORADAIAN))     return 30;
    if (angle < (-55 * TORADAIAN) )                                  return 31;
    return 0;
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


