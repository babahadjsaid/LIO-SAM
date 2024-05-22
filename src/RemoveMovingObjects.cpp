#include "lio_sam/utility.hpp"
#include "lio_sam/msg/cloud_info.hpp"


struct xi {
    float R0;
    float omega0;
    float alpha0;
};
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
    float k = 0.5;
    
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
    frames_.push_back(currentImage_);
    CurrentT_ = pcl::getTransformation(msgIn->initial_guess_x, msgIn->initial_guess_y, msgIn->initial_guess_z, msgIn->initial_guess_roll, msgIn->initial_guess_pitch, msgIn->initial_guess_yaw);
    ToCurrentT_ = previousT_.inverse() * CurrentT_;
    
    transformPointCloud(previous_pc_,ToCurrentT_);
    convertPointCloudToRangeImage(previous_pc_,previousImage_);
    displayImage(previousImage_);
    applyMedianFilter(currentImage_,previousImage_);
    // TODO: fix the problem of blank frames.... Currently i think it is not a problem from my part math part..
    vector<cv::Mat> rangeFlow;
    rangeFlow = estimateRangeFlow(currentImage_,previousImage_);
    xi tmp = estimateRangeFlow(currentImage_,rangeFlow);
    std::cout << "alpha: "<< tmp.alpha0<<" omega: " << tmp.omega0<< " R_t: "<<tmp.R0 << std::endl;
    // TODO: look if this is the same with the received vel??
    previous_pc_ = current_pc_;
    previousT_ = CurrentT_;
    const std::chrono::duration<double> duration = std::chrono::system_clock::now() - StartMethod;
    float durationms = 100 - 1000 * duration.count();
    //std::cout <<"delay: "<< 100 - durationms <<std::endl;
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
    cv::normalize(image, frame, 0, 1, cv::NORM_MINMAX);
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
    cv::Sobel(rangeImage1, dx, CV_32F, 1, 0, 1);//alpha
    cv::Sobel(rangeImage1, dy, CV_32F, 0, 1, 1);//omega
    cv::subtract(rangeImage2, rangeImage1, dR);
    return {dx, dy, dR};
}

float geometricConstraintResidual(const xi& rf, float R_w, float R_a, float R_t) {
    return R_w * rf.omega0 + R_a * rf.alpha0 + R_t - rf.R0;
}

float robustFunction(float rho) {
    return (pow(k,2) / 2) * log(1 + pow((rho / k),2));
}
float weightFunction(float rho) {
    return 1.0 / (1.0 + k * rho * rho);
}

// Function to estimate range flow
xi estimateRangeFlow(cv::Mat& rangeImage,vector<cv::Mat>& rangeflow) {
    xi estimatedFlow = {0, 0, 0};

    // Initialize variables for optimization
    float sumWeights = 0;
    Eigen::Vector3f sumResiduals(0, 0, 0);
     
    // Iterate over each point
    for(int i = 0; i < rangeImage.rows; i++)
    {
        for(int j = 0; j < rangeImage.cols; j++) {
        
        float R     = rangeImage.at<float>(i,j);
        float omega = rangeflow[0].at<float>(i,j);
        float alpha = rangeflow[1].at<float>(i,j);
        float R_t   = rangeflow[2].at<float>(i,j);

        float rho = geometricConstraintResidual(estimatedFlow, omega, alpha,R_t);
        float weight = weightFunction(rho);
        sumWeights += weight;
        sumResiduals += weight * Eigen::Vector3f(R * omega, R * alpha, R * R_t);
    }}

    // Estimate the range flow variables
    if (sumWeights > 0) {
        estimatedFlow.R0 = sumResiduals[0] / sumWeights;
        estimatedFlow.omega0 = sumResiduals[1] / sumWeights;
        estimatedFlow.alpha0 = sumResiduals[2] / sumWeights;
    }

    return estimatedFlow;
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
    for (size_t i = 0; i < IP->frames_.size(); i++)
    {
        IP->displayImage(IP->frames_[i]);
        
        cv::waitKey(100);
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
