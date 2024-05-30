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
    double maxTime= 0 ;
private:
    cv::Mat currentImage_, previousImage_;
    double previous_t,dt;
    pcl::PointCloud<PointType> current_pc_, previous_pc_;
    Eigen::Affine3f CurrentT_,ToCurrentT_,previousT_;
    rclcpp::Subscription<lio_sam::msg::CloudInfo>::SharedPtr subCloud_;
    rclcpp::Publisher<lio_sam::msg::CloudInfo>::SharedPtr pubLaserCloudInfo_;
    std::mutex mapMtx_;
    float k = 2.3849;
    int maxIterations = 30;
    
public:
    RemoveMovingObjects(const rclcpp::NodeOptions & options) :
            ParamServer("lio_sam_RemoveMovingObject", options),
            frames_({})
    {
         
        subCloud_ = create_subscription<lio_sam::msg::CloudInfo>( "lio_sam/deskew/cloud_info", qos, std::bind(&RemoveMovingObjects::pointCloudHandler, this, std::placeholders::_1));
        pubLaserCloudInfo_ = create_publisher<lio_sam::msg::CloudInfo>( "lio_sam/RemoveMovingObjects/cloud_info", qos);
        
    }


void pointCloudHandler(const lio_sam::msg::CloudInfo::SharedPtr msgIn){
    mapMtx_.lock();
    const auto StartMethod = std::chrono::system_clock::now();
    
    pcl::fromROSMsg(msgIn->cloud_deskewed,current_pc_);
    
    if (previous_pc_.empty())
    {
        previous_pc_ = current_pc_;
        previous_t = rclcpp::Time(msgIn->header.stamp).seconds();
        previousT_ = pcl::getTransformation(msgIn->initial_guess_x, msgIn->initial_guess_y, msgIn->initial_guess_z, msgIn->initial_guess_roll, msgIn->initial_guess_pitch, msgIn->initial_guess_yaw);
        pubLaserCloudInfo_->publish(*msgIn);
        mapMtx_.unlock();
        return;
    }
    dt = rclcpp::Time(msgIn->header.stamp).seconds() - previous_t;
    convertPointCloudToRangeImage(current_pc_,currentImage_);
    frames_.push_back(currentImage_);
    CurrentT_ = pcl::getTransformation(msgIn->initial_guess_x, msgIn->initial_guess_y, msgIn->initial_guess_z, msgIn->initial_guess_roll, msgIn->initial_guess_pitch, msgIn->initial_guess_yaw);
    ToCurrentT_ = previousT_.inverse() * CurrentT_;
    
    transformPointCloud(previous_pc_,ToCurrentT_);
    convertPointCloudToRangeImage(previous_pc_,previousImage_);
    applyMedianFilter(currentImage_,previousImage_);
    // TODO: fix the problem of blank frames.... Currently i think it is not a problem from my part math part..
    vector<cv::Mat> rangeFlow;
    rangeFlow = CalculateSpatio_temporalDer(currentImage_,previousImage_);
    const auto methodStartTime = std::chrono::system_clock::now();
    float V = sqrt(pow(msgIn->vel_x,2) + pow(msgIn->vel_y,2) + pow(msgIn->vel_z,2));
    xi tmp = CoarseEstimate(rangeFlow,dt);
    tmp = FineEstimate(tmp,rangeFlow,V,dt);
    cv::Mat out;
    out.create(32, MAXWIDTH, CV_32F);
    PointsSegmentation(tmp,rangeFlow,V,dt,out);
    const std::chrono::duration<double> durationOfFunction = std::chrono::system_clock::now() - methodStartTime;
    double durationOfFunctionMS = 1000 * durationOfFunction.count();
    std::cout<<"The function  Took " << durationOfFunctionMS <<" ms "<<endl;
    displayImage(out);
    maxTime = max(durationOfFunctionMS,maxTime);
    // TODO: look if this is the same with the received vel??
    previous_pc_ = current_pc_;
    previousT_ = CurrentT_;
    previous_t = rclcpp::Time(msgIn->header.stamp).seconds();
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
    cv::normalize(image, frame, 0, 5, cv::NORM_MINMAX);
    cv::imshow("test", frame);
    cv::waitKey(50);
    frame.release();
}

void applyMedianFilter(cv::Mat& currentImage, cv::Mat& previousImage, int kernelSize=3)
{
    cv::medianBlur(currentImage, currentImage, kernelSize);
    cv::medianBlur(previousImage, previousImage, kernelSize);
}

vector<cv::Mat> CalculateSpatio_temporalDer(const cv::Mat& rangeImage1, const cv::Mat& rangeImage2)
{
    // Calculate spatial-temporal gradients
    cv::Mat dx, dy, dR;
    cv::Sobel(rangeImage1, dx, CV_32F, 1, 0, 1);//alpha
    cv::Sobel(rangeImage1, dy, CV_32F, 0, 1, 1);//omega
    cv::subtract(rangeImage2, rangeImage1, dR);

    return {rangeImage1, dx, dy, dR};
}

float geometricConstraintResidual(const xi& rf, float R_w, float R_a, float R_t) {
    return R_w * rf.omega0 + R_a * rf.alpha0 + R_t - rf.R0;
}
float calculateVel(const xi& rf, float R_w, float R_a, float R_t, float w, float a, float R,float dt){
    float delta_x = R * cos(a) * sin(w)  - (R + rf.R0 * dt) * cos(a + rf.alpha0 * dt) * sin(w + rf.omega0 * dt);
    float delta_y = R * cos(a) * cos(w)  - (R + rf.R0 * dt) * cos(a + rf.alpha0 * dt) * cos(w + rf.omega0 * dt);
    float delta_z = R * sin(a)   - (R + rf.R0 * dt) * sin(a + rf.alpha0 * dt);
    return sqrt(pow(delta_x,2) + pow(delta_y,2) + pow(delta_z,2)) / dt;
}

float geometricConstraintResidual(const xi& rf, float R_w, float R_a, float R_t, float w, float a, float R, float V,float dt) {
    float v = calculateVel(rf,R_w,R_a,R_t,w,a,R,dt);

    return v - abs(V);
}



float weightFunction(float rho) {
    return 1.0 / (1.0 + pow(rho / k,2));
}

xi CoarseEstimate(const vector<cv::Mat>& rangeflow, double dt) {
    xi estimatedFlow = {0, 0, 0};
    int rows = rangeflow[0].rows;
    int cols = rangeflow[0].cols;
    float change,delta =  0.0001;
    float residuals;
    float weight;
    Eigen::VectorXf Y(3);
    Eigen::MatrixXf M(3,3);
    for (int iter = 0; iter < 10; ++iter) {
        int index = 0;
        float a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, b1 = 0, b2 = 0, b3 = 0;
        for (int i = 0; i < rows; ++i) {
            double dw = (i == 0) ? 2 : (LidarAngles[i] - LidarAngles[i-1]);
            for (int j = 0; j < cols; ++j) {
                float omega = rangeflow[1].at<float>(i, j) / dw;
                float alpha = rangeflow[2].at<float>(i, j) / ANGULARRESOLUTION_X;
                float R_t = rangeflow[3].at<float>(i, j) / dt;
                float rho = geometricConstraintResidual(estimatedFlow, omega, alpha, R_t);
                weight = weightFunction(rho);
                
                
                
                a1 += weight * omega * omega;
                a2 += weight * omega * alpha;
                a3 += weight * omega ;
                a4 += weight * alpha * alpha;
                a5 += weight * alpha ;
                a6 += weight ;
                b1 += weight * omega * R_t;
                b2 += weight * alpha * R_t;
                b3 += weight * R_t;
            }
        }
        
        M << a1, a2, a3,
            a2, a4, a5,
            a3, a5, a6;
        Y << b1, b2, b3;
        Eigen::VectorXf B = M.inverse() * Y;
        if (isnan(B.norm())) continue;
        xi newEstimate = {B(0), B(1), B(2)};
        change = abs(newEstimate.omega0 - estimatedFlow.omega0) +
                        abs(newEstimate.alpha0 - estimatedFlow.alpha0) +
                        abs(newEstimate.R0 - estimatedFlow.R0);
        if (change < 0.01) {
            estimatedFlow = newEstimate;
            break;
        }
        
        estimatedFlow = newEstimate;
    }
    cout << "change: "<<change << endl;
    return estimatedFlow;
}

xi FineEstimate(xi estimatedFlow, const vector<cv::Mat>& rangeflow,double V, double dt) {
    
    int rows = rangeflow[0].rows;
    int cols = rangeflow[0].cols;
    float change,delta =  0.001;
    float residuals;
    float weight,k = 1.2107;
    Eigen::VectorXf Y(3);
    Eigen::MatrixXf M(3,3);
    for (int iter = 0; iter < maxIterations; ++iter) {
        int index = 0;
        float a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, b1 = 0, b2 = 0, b3 = 0;
        for (int i = 0; i < rows; ++i) {
            double dw = (i == 0) ? 2 : (LidarAngles[i] - LidarAngles[i-1]);
            for (int j = 0; j < cols; ++j) {
                float R = rangeflow[0].at<float>(i, j) ;
                float omega = rangeflow[1].at<float>(i, j) / dw;
                float alpha = rangeflow[2].at<float>(i, j) / ANGULARRESOLUTION_X;
                float R_t = rangeflow[3].at<float>(i, j) / dt;
                float rho = geometricConstraintResidual(estimatedFlow, omega, alpha, R_t,LidarAngles[i],j*ANGULARRESOLUTION_X,R,V,dt);
                rho = abs(rho);
                if (rho<k)
                {
                    weight = 1;
                }else if (rho >= k)
                {
                    weight = k/rho;
                }
                a1 += weight * omega * omega;
                a2 += weight * omega * alpha;
                a3 += weight * omega ;
                a4 += weight * alpha * alpha;
                a5 += weight * alpha ;
                a6 += weight ;
                b1 += weight * omega * R_t;
                b2 += weight * alpha * R_t;
                b3 += weight * R_t;
            }
        }
        
        M << a1, a2, a3,
            a2, a4, a5,
            a3, a5, a6;
        Y << b1, b2, b3;
        Eigen::VectorXf B = M.inverse() * Y;
        if (isnan(B.norm())) continue;
        
        xi newEstimate = {B(0), B(1), B(2)};
        change = abs(newEstimate.omega0 - estimatedFlow.omega0) +
                        abs(newEstimate.alpha0 - estimatedFlow.alpha0) +
                        abs(newEstimate.R0 - estimatedFlow.R0);
        if (change < 0.001) {
            estimatedFlow = newEstimate;
            break;
        }
        
        estimatedFlow = newEstimate;
    }
    cout << "change: "<<change << endl;
    return estimatedFlow;
}

void PointsSegmentation(xi estimatedFlow, const vector<cv::Mat>& rangeflow,float V,float dt,cv::Mat& out){
    int rows = rangeflow[0].rows;
    int cols = rangeflow[0].cols;
    for (int i = 0; i < rows; ++i) {
        double dw = (i == 0) ? 2 : (LidarAngles[i] - LidarAngles[i-1]);
        for (int j = 0; j < cols; ++j) {
            float R = rangeflow[0].at<float>(i, j) ;
            float omega = rangeflow[1].at<float>(i, j) / dw;
            float alpha = rangeflow[2].at<float>(i, j) / ANGULARRESOLUTION_X;
            float R_t = rangeflow[3].at<float>(i, j) / dt;
            float& Rout = out.at<float>(i, j);
            float v  = geometricConstraintResidual(estimatedFlow, omega, alpha, R_t,LidarAngles[i],j*ANGULARRESOLUTION_X,R,V,dt);
            if (abs(v)<7 )
            {
               Rout = 1;
            }
            else{
                Rout = 0;
            }
            
            
        }
    }
        
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

    cout << "max time: "<<IP->maxTime<<endl;
    return 0;
}





    
// }
// void writeVideo(cv::Mat &image)
// {
//     cv::Mat frame;
//     cv::normalize(currentImage_, frame, 0, 255, cv::NORM_MINMAX);
    
//     cv::imwrite("image.png",frame);
// }
