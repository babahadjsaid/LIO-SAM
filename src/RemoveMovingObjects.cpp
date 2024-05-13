#include "lio_sam/utility.hpp"
#include "lio_sam/msg/cloud_info.hpp"

class RemoveMovingObjects : public ParamServer
{
public:
    vector<cv::Mat> frames_;
private:
    cv::Mat currentImage_, previousImage_;
    pcl::PointCloud<PointType> current_pc_, previous_pc_;
    Eigen::Affine3f CurrentT_,ToCurrentT_,previousT_;
    rclcpp::Subscription<lio_sam::msg::CloudInfo>::SharedPtr subCloud_;
    rclcpp::Publisher<lio_sam::msg::CloudInfo>::SharedPtr pubLaserCloudInfo_;
    std::mutex mapMtx_;
    
public:
    RemoveMovingObjects(const rclcpp::NodeOptions & options) :
            ParamServer("lio_sam_RemoveMovingObject", options),
            frames_({})
    {
         
        subCloud_ = create_subscription<lio_sam::msg::CloudInfo>(
            "lio_sam/deskew/cloud_info", qos,
            std::bind(&RemoveMovingObjects::pointCloudHandler, this, std::placeholders::_1));
        pubLaserCloudInfo_ = create_publisher<lio_sam::msg::CloudInfo>(
            "lio_sam/RemoveMovingObjects/cloud_info", qos);
    
    }


void pointCloudHandler(const lio_sam::msg::CloudInfo::SharedPtr msgIn){
    mapMtx_.lock();
    const auto StartMethod = std::chrono::system_clock::now();
    
    pcl::fromROSMsg(msgIn->cloud_deskewed,current_pc_);
    
    if (previous_pc_.empty())
    {
        previous_pc_ = current_pc_;
        previousT_ = pcl::getTransformation(msgIn->initial_guess_x, msgIn->initial_guess_y, msgIn->initial_guess_z, msgIn->initial_guess_roll, msgIn->initial_guess_pitch, msgIn->initial_guess_yaw);
        pubLaserCloudInfo_->publish(*msgIn);
        mapMtx_.unlock();
        return;
    }
    
    convertPointCloudToRangeImage(current_pc_,currentImage_);
    displayImage(currentImage_);
    CurrentT_ = pcl::getTransformation(msgIn->initial_guess_x, msgIn->initial_guess_y, msgIn->initial_guess_z, msgIn->initial_guess_roll, msgIn->initial_guess_pitch, msgIn->initial_guess_yaw);
    ToCurrentT_ = previousT_.inverse() * CurrentT_;
    
    transformPointCloud(previous_pc_,ToCurrentT_);
    convertPointCloudToRangeImage(previous_pc_,previousImage_);
    applyMedianFilter(currentImage_,previousImage_);
    vector<cv::Mat> rangeFlow;
    rangeFlow = estimateRangeFlow(currentImage_,previousImage_);

    // TODO: fix the problem of blank frames....


    previous_pc_ = current_pc_;
    previousT_ = CurrentT_;
    const std::chrono::duration<double> duration = std::chrono::system_clock::now() - StartMethod;
    float durationms = 100 - 1000 * duration.count();
    std::cout <<"delay: "<< 100 - durationms <<std::endl;
    pubLaserCloudInfo_->publish(*msgIn);
    mapMtx_.unlock();
    return;
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

void displayImage(cv::Mat &image){ // this is just pieces of code not yet implemented
    cv::Mat frame,image2;
    cv::normalize(image, frame, 0, 255, cv::NORM_MINMAX);
    frame.convertTo(image2, CV_8U);
    std::stringstream ss;
    ss << "image_" << frames_.size() << ".png";
    // cv::imwrite(ss.str(),image2);
    frames_.push_back(image2);
    cv::imshow("test", frame);
    cv::waitKey(50);
    frame.release();
}

void applyMedianFilter(cv::Mat& currentImage, cv::Mat& previousImage, int kernelSize=3)
{
    cv::medianBlur(currentImage, currentImage, kernelSize);
    cv::medianBlur(previousImage, previousImage, kernelSize);
}

vector<cv::Mat> estimateRangeFlow(cv::Mat& rangeImage1, cv::Mat& rangeImage2)
{
    // Calculate spatial-temporal gradients
    cv::Mat dx, dy, dR;
    cv::Sobel(rangeImage1, dx, CV_32F, 1, 0, 1);
    cv::Sobel(rangeImage1, dy, CV_32F, 0, 1, 1);
    cv::subtract(rangeImage2, rangeImage1, dR);
    return {dx, dy, dR};
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
    cout << "Hello"<<endl;
    cv::Size S = IP->frames_[0].size();    
    int fourcc = cv::VideoWriter::fourcc('H','2','6','4');
    cv::VideoWriter outputVideo;  // Open the output
    outputVideo.open("output.avi"  , fourcc, 20, S, false);  //30 for 30 fps

    if (!outputVideo.isOpened()){
        cout  << "Could not open the output video for write: "<< endl;
        return -1;
    }

    for(int i=0; i<IP->frames_.size(); i++){
        outputVideo << IP->frames_[i];
        
    }

    cout << "Finished writing" << endl;
    return 0;
}





    
// }
// void writeVideo(cv::Mat &image)
// {
//     cv::Mat frame;
//     cv::normalize(currentImage_, frame, 0, 255, cv::NORM_MINMAX);
    
//     cv::imwrite("image.png",frame);
// }
