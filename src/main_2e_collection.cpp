#include <iostream>
#include "livox_lidar_def.h"
#include "livox_lidar_api.h"
#include <eigen3/Eigen/Dense>
#include <unistd.h>
#include <experimental/filesystem>
#include <stdio.h>
#include <ctime>
#include <cstring>
#include "common_utils.h"
#include <iomanip>
#include <arpa/inet.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/ocl.hpp>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>
#include <omp.h>
#include "fisheye_undistorter.h"

// 引入 ROS2 及 PointCloud2 相关头文件
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include "camera_2eye.h"
#include "ply_fast_writer.h"
#include <CL/cl.h>
#include <deque>
#include <numeric>
#include <cmath>
#include <atomic>
#include "stepper_motor.h"

//#define IMU_UPSIDEDOWN
#define RECORD_POINT_CLOUD_FRAMES 20
#define SOFT_COLOR 0
#define SOFT_UNDISTORT 0

namespace fs = std::experimental::filesystem::v1;
using namespace cv;
using namespace std;

// 自定义带有 RGB 信息的点结构体，用于在线程间传递
// 1. 严格对齐结构体 (与 OpenCL 端 __attribute__((packed)) 对应)
struct __attribute__((packed)) ColoredPoint {
    float x, y, z;
    float reflectivity;
    uint8_t r, g, b, tag, line;
    uint8_t _padding[3]; // 显式填充，确保对齐到 32 字节或更高
    uint64_t timestamp;
};

// 2. 内联 OpenCL Kernel 代码
const char* FISHEYE_KERNEL_SRC = R"(
typedef struct __attribute__((packed)) {
    float x, y, z;
    float reflectivity;
    uchar r, g, b, tag, line;
    uchar _padding[3];
    ulong timestamp;
} ColoredPoint;

__kernel void process_pointcloud_rk3588(
    __global ColoredPoint* pts,
    __global const uchar* bgr_data,   // 修改为 Buffer 访问
    __constant float* mat,
    const float4 camera_k,
    const float4 camera_d,
    const int img_width,             // 传入图像宽度用于偏移计算
    const int img_height,
    const int num_points)
{
    int i = get_global_id(0);
    if (i >= num_points) return;

    float p_in_x = -pts[i].z;
    float p_in_y =  pts[i].y;
    float p_in_z =  pts[i].x;

    float z_c = mat[8]*p_in_x + mat[9]*p_in_y + mat[10]*p_in_z + mat[11];
    if (z_c <= 0.1f) return;

    float x_c = mat[0]*p_in_x + mat[1]*p_in_y + mat[2]*p_in_z + mat[3];
    float y_c = mat[4]*p_in_x + mat[5]*p_in_y + mat[6]*p_in_z + mat[7];

    float r = sqrt(x_c * x_c + y_c * y_c);
    float theta = atan2(r, z_c);
    float t2 = theta * theta;
    float t4 = t2 * t2;
    float t6 = t4 * t2;
    float t8 = t4 * t4;
    float theta_d = theta * (1.0f + camera_d.x*t2 + camera_d.y*t4 + camera_d.z*t6 + camera_d.w*t8);

    float scale = (r > 1e-6f) ? (theta_d / r) : 1.0f;
    int u = (int)(camera_k.x * (x_c * scale) + camera_k.z + 0.5f); // 四舍五入
    int v = (int)(camera_k.y * (y_c * scale) + camera_k.w + 0.5f);

    // 3. 手动计算 BGR 偏移并采样
    if (u >= 0 && u < img_width && v >= 0 && v < img_height) {
        // 每个像素 3 字节 (BGR)
        int idx = (v * img_width + u) * 3;

        // 直接赋值，注意 OpenCV 默认是 BGR 顺序
        pts[i].b = bgr_data[idx];     // Byte 0
        pts[i].g = bgr_data[idx + 1]; // Byte 1
        pts[i].r = bgr_data[idx + 2]; // Byte 2
    }
}
)";

volatile uint32_t mLidarHandle = std::numeric_limits<uint32_t>::max();
std::vector<ColoredPoint> frame_points;
cv::Mat frame_image;
volatile size_t framesScanned = 0;
volatile size_t g_frames_per_publish;
int image_actual_width;
int image_actual_height;

static Eigen::Matrix4f Rtilt = Eigen::Matrix4f::Identity();

volatile bool g_image_raw_compressed;
volatile bool g_tilt;
volatile bool g_undistort;
std::mutex g_undistort_params_mutex;
volatile double g_undistort_angle_h;
volatile double g_undistort_angle_v;
cv::Mat g_undistort_r;
volatile int g_undistort_w;
volatile int g_undistort_h;
volatile double g_undistort_fxy;
volatile double g_valid_incidence;
cv::Size g_undistort_img_size;
cv::Matx33d g_undistort_k = cv::Matx33d::eye();
std::atomic<bool> g_is_static{false};
volatile bool g_scanning = false;

//计算参数
cv::Mat g_k1, g_k2, g_r1, g_r2, g_extrinsic_left, g_extrinsic_right, g_map_x1, g_map_y1, g_map_x2, g_map_y2;
cv::Mat g_fisheye_stereo_map_x1, g_fisheye_stereo_map_y1, g_fisheye_stereo_map_x2, g_fisheye_stereo_map_y2;
cv::Mat g_panorama_map_x1, g_panorama_map_y1, g_panorama_map_x2, g_panorama_map_y2;
double g_baseline;
cv::Matx33d g_fisheye_k[2];
cv::Vec4d g_fisheye_d[2];     /* 摄像机的4个畸变系数：k1,k2,k3,k4*/
// ============================================================================
// 数据结构定义
// ============================================================================

// COLMAP cameras.txt 的相机参数结构体
struct ColmapCamera {
    int camera_id;
    std::string model;
    int width;
    int height;
    std::vector<double> params; // 例如 PINHOLE 模型对应: fx, fy, cx, cy
};

// COLMAP images.txt 的图像位姿结构体
struct ColmapImage {
    int image_id;
    double qw, qx, qy, qz;      // 四元数 (旋转)
    double tx, ty, tz;          // 平移向量
    int camera_id;
    std::string name;
    // COLMAP 格式要求第二行保存 2D 特征点和对应的 3D 点 ID。
    // 如果只想生成纯位姿（如 MVS 阶段），可留空。
    std::string points2d_line = "";
};

// ============================================================================
// 1. 生成 Middlebury 格式的 calib.txt
// ============================================================================
bool generate_calib_txt(const std::string& filepath,
                        double fx, double fy, double cx, double cy,
                        double doffs, double baseline,
                        int width, int height, int ndisp, int pano_ndisp) {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "[Error] 无法创建文件: " << filepath << std::endl;
        return false;
    }

    // 设置浮点数输出精度，去除多余的后导零
    out << std::defaultfloat << std::setprecision(6);

    // 写入 cam0 (内参矩阵 K [fx 0 cx; 0 fy cy; 0 0 1])
    out << "cam0=[" << fx << " 0 " << cx << "; "
        << "0 " << fy << " " << cy << "; "
        << "0 0 1]\n";

    // 写入 cam1 (通常在校正后的双目中，cam1 内参和 cam0 一致)
    out << "cam1=[" << fx << " 0 " << cx << "; "
        << "0 " << fy << " " << cy << "; "
        << "0 0 1]\n";

    // 写入标定基线和其他参数
    out << "doffs=" << doffs << "\n";
    out << "baseline=" << baseline << "\n";
    out << "width=" << width << "\n";
    out << "height=" << height << "\n";
    out << "ndisp=" << ndisp << "\n";
    out << "pano_ndisp=" << pano_ndisp << "\n";

    out.close();
    std::cout << "[Success] 已生成: " << filepath << std::endl;
    return true;
}

// ============================================================================
// 2. 生成 COLMAP 格式的 cameras.txt
// ============================================================================
bool generate_cameras_txt(const std::string& filepath,
                          const std::vector<ColmapCamera>& cameras) {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "[Error] 无法创建文件: " << filepath << std::endl;
        return false;
    }

    // 写入 COLMAP 标准头部注释
    out << "# Camera list with one line of data per camera:\n";
    out << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
    out << "# Number of cameras: " << cameras.size() << "\n";

    out << std::defaultfloat << std::setprecision(6);

    // 写入每一个相机模型
    for (const auto& cam : cameras) {
        out << cam.camera_id << " "
            << cam.model << " "
            << cam.width << " "
            << cam.height;
        for (double param : cam.params) {
            out << " " << param;
        }
        out << "\n";
    }

    out.close();
    std::cout << "[Success] 已生成: " << filepath << std::endl;
    return true;
}

// ============================================================================
// 3. 生成 COLMAP 格式的 images.txt
// ============================================================================
bool generate_images_txt(const std::string& filepath,
                         const std::vector<ColmapImage>& images) {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "[Error] 无法创建文件: " << filepath << std::endl;
        return false;
    }

    // 写入 COLMAP 标准头部注释
    out << "# Image list with two lines of data per image:\n";
    out << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
    out << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
    out << "# Number of images: " << images.size() << "\n";

    out << std::defaultfloat << std::setprecision(6);

    // COLMAP 格式要求：每一条 Image 数据必须占两行
    for (const auto& img : images) {
        // 第一行：图像位姿和文件名
        out << img.image_id << " "
            << img.qw << " " << img.qx << " " << img.qy << " " << img.qz << " "
            << img.tx << " " << img.ty << " " << img.tz << " "
            << img.camera_id << " "
            << img.name << "\n";

        // 第二行：2D特征点信息 (此处没有特征点，按 COLMAP 规范写入空行或预留数据)
        out << img.points2d_line << "\n";
    }

    out.close();
    std::cout << "[Success] 已生成: " << filepath << std::endl;
    return true;
}

struct ColmapPose {
    double qw, qx, qy, qz;
    double tx, ty, tz;
};

/**
 * @brief 从 4x4 World-to-Camera 矩阵中直接提取 COLMAP 格式的四元数和平移
 * @param M_w2c 4x4 的齐次变换矩阵 (世界坐标系到相机坐标系)
 */
ColmapPose extract_colmap_pose_from_w2c(const cv::Mat& trans_mat) {
    Eigen::Matrix4d M_w2c;
    cv::cv2eigen(trans_mat, M_w2c);
    // 1. 直接提取右上角的平移向量 (tx, ty, tz)
    Eigen::Vector3d t_w2c = M_w2c.block<3, 1>(0, 3);

    // 2. 直接提取左上角的旋转矩阵 R
    Eigen::Matrix3d R_w2c = M_w2c.block<3, 3>(0, 0);

    // 3. 将旋转矩阵转换为四元数
    Eigen::Quaterniond q(R_w2c);
    q.normalize(); // 归一化消除浮点误差

    // 4. 装填数据
    ColmapPose pose;
    pose.qw = q.w();
    pose.qx = q.x();
    pose.qy = q.y();
    pose.qz = q.z();

    pose.tx = t_w2c.x();
    pose.ty = t_w2c.y();
    pose.tz = t_w2c.z();

    return pose;
}

/**
 * @brief 从 3D 射线坐标 (float) 反求图像/投影网格的 (h, v) 坐标 (float)
 *
 * @param pt3d        3D 空间中的射线向量 (X, Y, Z) - cv::Point3f
 * @param outSize     目标网格的尺寸 (width, height)
 * @param alpha_min   alpha 的起始角度 (建议与正向保持同精度)
 * @param alpha_range alpha 的角度范围
 * @param beta_min    beta 的起始角度
 * @param beta_range  beta 的角度范围
 * @return cv::Point2f 返回计算出的 (h, v) 浮点坐标 - cv::Point2f
 */
cv::Point2f project3DToGrid(const cv::Point3f& pt3d,
                            const cv::Size& outSize,
                            float alpha_min, float alpha_range,
                            float beta_min, float beta_range)
{
    // 1. 向量归一化 (使用 float 版本的 std::sqrt)
    float norm = std::sqrt(pt3d.x * pt3d.x + pt3d.y * pt3d.y + pt3d.z * pt3d.z);

    // 防止除以 0 的极端情况
    if (norm < 1e-6f) {
        return cv::Point2f(0.0f, 0.0f);
    }

    float X = pt3d.x / norm;
    float Y = pt3d.y / norm;
    float Z = pt3d.z / norm;

    // 2. 反解 beta 和 alpha 角度 (调用 std::atan2 的 float 重载)
    float beta = std::atan2(Y, Z);

    // 使用乘积和开方求 alpha，比直接用 asin(X) 更稳定
    float alpha = std::atan2(X, std::sqrt(Y * Y + Z * Z));

    // 3. 线性映射回 pixel/grid 坐标
    float h = ((alpha - alpha_min) / alpha_range) * (outSize.width - 1);
    float v = ((beta - beta_min) / beta_range) * (outSize.height - 1);

    return cv::Point2f(h, v);
}

template<typename T, typename U>
auto clamp_value(const T& value, const U& low, const U& high) -> U {
    return (value < low) ? low : ((value > high) ? high : static_cast<U>(value));
}

/**
 * 模拟头部运动：先水平转头（Yaw），再竖直低头/抬头（Pitch）
 * @param yaw_deg   水平旋转角度（左右转头），正值表示向右转
 * @param pitch_deg 竖直旋转角度（上下点头），正值表示低头
 * @return 3x3 旋转矩阵
 */
// 更简洁的版本（直接操作旋转矩阵）
cv::Mat getHeadRotationCompact(double yaw_deg, double pitch_deg) {
    cv::Mat R_yaw, R_pitch, R_combined;

    cv::Rodrigues(cv::Vec3d(0, yaw_deg * CV_PI / 180.0, 0), R_yaw);
    cv::Rodrigues(cv::Vec3d(pitch_deg * CV_PI / 180.0, 0, 0), R_pitch);

    return R_pitch * R_yaw;// 先水平，再竖直
}

void update_undistort_r() {
    Mat pinholeR = getHeadRotationCompact(g_undistort_angle_h, g_undistort_angle_v);
    std::lock_guard<std::mutex> lock(g_undistort_params_mutex);
    g_undistort_r = pinholeR;
}
void update_undistort_k() {
    std::lock_guard<std::mutex> lock(g_undistort_params_mutex);
    g_undistort_img_size.width = g_undistort_w;
    g_undistort_img_size.height = g_undistort_h;
    g_undistort_k(0, 2) = g_undistort_img_size.width/2.0;//cx
    g_undistort_k(1, 2) = g_undistort_img_size.height/2.0;//cy
    g_undistort_k(0, 0) = g_undistort_fxy;//fx
    g_undistort_k(1, 1) = g_undistort_fxy;//fy
}

// 声明动态参数处理函数
rcl_interfaces::msg::SetParametersResult on_parameter_change(
        const std::vector<rclcpp::Parameter> & parameters) {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "success";
    auto logger = rclcpp::get_logger("param_callback");


    for (const auto & param : parameters) {
        if (param.get_name() == "undistort") {
            if (param.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) {
                result.successful = false;
                result.reason = "Type mismatch! Must be boolean.";
                return result;
            }
            g_undistort = param.as_bool();
            RCLCPP_INFO(logger, "undistort changed to: %s", g_undistort ? "true" : "false");
        }
        else if (param.get_name() == "undistort_angle_h") {
            double value;
            if (param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
                value = static_cast<double>(param.as_int());
            } else if (param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
                value = param.as_double();
            } else {
                result.successful = true;
                result.reason = "Type mismatch! Must be int or double.";
                return result;
            }
            g_undistort_angle_h = clamp_value(value, -45.0, 45.0);
            update_undistort_r();
            RCLCPP_INFO(logger, "undistort_angle_h changed to: %f", g_undistort_angle_h);
        }
        else if (param.get_name() == "undistort_angle_v") {
            double value;
            if (param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
                value = static_cast<double>(param.as_int());
            } else if (param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
                value = param.as_double();
            } else {
                result.successful = true;
                result.reason = "Type mismatch! Must be int or double.";
                return result;
            }
            g_undistort_angle_v = clamp_value(value, -45.0, 45.0);
            update_undistort_r();
            RCLCPP_INFO(logger, "undistort_angle_v changed to: %f", g_undistort_angle_v);
        }
        else if (param.get_name() == "undistort_w") {
            if (param.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER) {
                result.successful = false;
                result.reason = "Type mismatch! Must be int.";
                return result;
            }
            g_undistort_w = static_cast<int>(clamp_value(param.as_int(), 100.0, image_actual_width*0.75));
            update_undistort_k();
            RCLCPP_INFO(logger, "undistort_w changed to: %f", g_undistort_w);
        }
        else if (param.get_name() == "undistort_h") {
            if (param.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER) {
                result.successful = false;
                result.reason = "Type mismatch! Must be int.";
                return result;
            }
            g_undistort_h = static_cast<int>(clamp_value(param.as_int(), 100.0, image_actual_height*0.75));
            update_undistort_k();
            RCLCPP_INFO(logger, "undistort_h changed to: %f", g_undistort_h);
        }
        else if (param.get_name() == "undistort_fxy") {
            double value;
            if (param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
                value = static_cast<double>(param.as_int());
            } else if (param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
                value = param.as_double();
            } else {
                result.successful = true;
                result.reason = "Type mismatch! Must be int or double.";
                return result;
            }
            int max_len = g_undistort_w > g_undistort_h ? g_undistort_w : g_undistort_h;
            g_undistort_fxy = clamp_value(value, max_len*0.25, max_len*0.75);
            update_undistort_k();
            RCLCPP_INFO(logger, "undistort_fxy changed to: %f", g_undistort_fxy);
        }
        else if (param.get_name() == "image_raw_compressed") {
            if (param.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) {
                result.successful = false;
                result.reason = "Type mismatch! Must be boolean.";
                return result;
            }
            g_image_raw_compressed = param.as_bool();
            RCLCPP_INFO(logger, "image_raw_compressed changed to: %s", g_image_raw_compressed ? "true" : "false");
        } else if (param.get_name() == "tilt") {
            if (param.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) {
                result.successful = false;
                result.reason = "Type mismatch! Must be boolean.";
                return result;
            }
            g_tilt = param.as_bool();
            RCLCPP_INFO(logger, "tilt changed to: %s", g_tilt ? "true" : "false");
        }
    }

    return result;
}

// --- 全局队列与线程同步变量 ---
std::mutex g_cloud_mutex;
std::mutex g_motor_mutex;
std::condition_variable g_cloud_cv;
std::condition_variable g_motor_cv;
std::queue<std::vector<ColoredPoint>> g_cloud_queue;
std::queue<cv::Mat> g_image_queue;

// 防爆核心：最多只缓存 2 帧点云。来不及处理就丢弃旧的，保证 RViz 永远看最新帧。
const size_t MAX_QUEUE_SIZE = 1;

CAMERA2EYE::CameraSingle* pCameraSingle = nullptr;
Eigen::Matrix4d transform_to_camera_matrix;
FisheyeUndistorter* pFisheyeUndistorter = nullptr;

// 成员变量建议
cl_platform_id platform_id = NULL;
cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue queue_cl = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;

cl_mem buf_bgr_persistent = nullptr;
cl_mem buf_pts_persistent = nullptr;
cl_mem buf_mat_persistent = nullptr;
cl_float4 k_vec, d_vec;

void setupOpenCL(size_t bgr_buf_size, size_t pts_buf_size) {//预估最大 BGR 图像字节数, 最大点云字节数
    cl_int err;
    cl_uint num_platforms;
    cl_uint num_devices;

    // 1. 获取平台 (Platform)
    // RK3588 上通常只有一个 "ARM Platform"
    err = clGetPlatformIDs(1, &platform_id, &num_platforms);
    if (err != CL_SUCCESS || num_platforms <= 0) {
        std::cerr << "Failed to find OpenCL platform." << std::endl;
        return;
    }

    // 2. 获取设备 (Device)
    // 指定查找 GPU 设备 (Mali-G610)
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
    if (err != CL_SUCCESS || num_devices <= 0) {
        std::cerr << "Failed to find GPU device." << std::endl;
        return;
    }

    // 3. 创建上下文 (Context)
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return;
    }

    // 4. 创建命令队列 (Command Queue)
    // 注意：OpenCL 2.0+ 推荐使用 clCreateCommandQueueWithProperties
    // 但在某些 RK3588 驱动版本中，传统的 clCreateCommandQueue 兼容性更好
#if CL_TARGET_OPENCL_VERSION >= 200
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    queue_cl = clCreateCommandQueueWithProperties(context, device_id, props, &err);
#else
    queue_cl = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
#endif

    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue." << std::endl;
        return;
    }

    // 5. 编译程序 (Program) - 使用之前定义的 FISHEYE_KERNEL_SRC
    program = clCreateProgramWithSource(context, 1, &FISHEYE_KERNEL_SRC, NULL, &err);
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    if (err != CL_SUCCESS) {
        // 如果编译失败，获取错误日志
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        std::cerr << "OpenCL Build Error:\n" << log.data() << std::endl;
        return;
    }

    // 6. 创建内核 (Kernel)
    kernel = clCreateKernel(program, "process_pointcloud_rk3588", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel." << std::endl;
        return;
    }

    buf_bgr_persistent = clCreateBuffer(context, CL_MEM_READ_ONLY, bgr_buf_size, nullptr, &err);
    buf_pts_persistent = clCreateBuffer(context, CL_MEM_READ_WRITE, pts_buf_size, nullptr, &err);
    buf_mat_persistent = clCreateBuffer(context, CL_MEM_READ_ONLY, 12*sizeof(float), nullptr, &err);
    if (buf_bgr_persistent && buf_pts_persistent && buf_mat_persistent) {
        std::cout << "OpenCL initialized successfully on Mali-G610." << std::endl;
    } else {
        std::cout << "OpenCL clCreateBuffer failed!" << std::endl;
    }
}

void cleanupOpenCL() {
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (queue_cl) clReleaseCommandQueue(queue_cl);
    if (context) clReleaseContext(context);
}

void savePFM(const std::string& filename, const cv::Mat& mat) {
    cv::Mat flipped;
    cv::flip(mat, flipped, 0);//垂直翻转
    std::ofstream out(filename, std::ios::binary);
    out << "Pf" << std::endl; // Pf 表示单通道浮点数
    out << mat.cols << " " << mat.rows << std::endl;
    out << "-1.0" << std::endl; // -1.0 表示小端序
    out.write(reinterpret_cast<const char*>(flipped.data), mat.rows * mat.cols * sizeof(float));
    out.close();
}

template<typename ... Args>
static std::string str_format(const std::string &format, Args ... args)
{
    auto size_buf = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1;
    std::unique_ptr<char[]> buf(new(std::nothrow) char[size_buf]);

    if (!buf)
        return std::string("");

    std::snprintf(buf.get(), size_buf, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size_buf - 1);
}

std::string time_format(uint64_t ns) {
    auto t = (time_t)(ns/1000000000);
    std::tm *tm = std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}
// --- 回调函数：Livox 数据入口 (10Hz) ---
void PointCloudCallback(uint32_t handle, const uint8_t dev_type, LivoxLidarEthernetPacket *data, void *client_data) {
    mLidarHandle = handle;
    if (data == nullptr || !g_scanning) return;
//    printf("point cloud handle: %u, data_num: %d, data_type: %d, length: %d, frame_counter: %d, time_type: %d\n",
//           handle, data->dot_num, data->data_type, data->length, data->frame_cnt, data->time_type);

    // 假设 Mid-360 使用的是 kExtendCartesian 模式
    if (data->data_type == kLivoxLidarCartesianCoordinateHighData) {
        LivoxLidarCartesianHighRawPoint *p_point_data = (LivoxLidarCartesianHighRawPoint *)data->data;
        uint32_t points_num = data->dot_num;
        uint64_t frameTime = *((uint64_t*)data->timestamp);
        ColoredPoint cp = {};
        for (uint32_t i = 0; i < points_num; i++) {
            if (p_point_data[i].x == 0 && p_point_data[i].y == 0 && p_point_data[i].z == 0) {
                continue;
            }
#ifdef IMU_UPSIDEDOWN
            cp.x = -p_point_data[i].x / 1000.0f; // 毫米转米
            cp.y = p_point_data[i].y / 1000.0f;
            cp.z = -p_point_data[i].z / 1000.0f;
#else
            cp.x = p_point_data[i].x / 1000.0f; // 毫米转米
            cp.y = p_point_data[i].y / 1000.0f;
            cp.z = p_point_data[i].z / 1000.0f;
#endif

            cp.reflectivity = p_point_data[i].reflectivity;
            cp.r = 255; cp.g = 255; cp.b = 255; // 默认白色
            cp.tag = p_point_data[i].tag;
            cp.timestamp = frameTime;
            cp.line = i % points_num;
            frame_points.push_back(cp);
        }
    }

    framesScanned++;

    // 将解析好的点云深拷贝放入队列
    if (pCameraSingle != nullptr && frame_image.empty()) {
        pCameraSingle->cachedImage(frame_image);
    }

    if (framesScanned >= g_frames_per_publish) {
        {
            std::lock_guard<std::mutex> lock(g_cloud_mutex);

            if (g_cloud_queue.size() >= MAX_QUEUE_SIZE) {
                g_cloud_queue.pop();
                g_image_queue.pop();
            }

            g_cloud_queue.push(frame_points);
            g_image_queue.push(frame_image);
            g_cloud_cv.notify_one();
        }

        frame_points.clear();
        framesScanned = 0;
        {
            std::lock_guard<std::mutex> lock(g_motor_mutex);
            g_scanning = false;
            g_motor_cv.notify_one();
        }
    }
}

void SetFovCfgCallback(livox_status status, uint32_t handle, LivoxLidarAsyncControlResponse *response, void *client_data) {
    if (response == nullptr) {
        return;
    }
    printf("LivoxLidarFovCfgCallback, status:%u, handle:%u, ret_code:%u, error_key:%u",
           status, handle, response->ret_code, response->error_key);
}

void EnableFovCfgCallback(livox_status status, uint32_t handle, LivoxLidarAsyncControlResponse *response, void *client_data) {
    if (response == nullptr) {
        return;
    }
    printf("LivoxLidarEnableFovCfgCallback, status:%u, handle:%u, ret_code:%u, error_key:%u",
           status, handle, response->ret_code, response->error_key);
}

void LivoxLidarPushMsgCallback(const uint32_t handle, const uint8_t dev_type, const char* info, void* client_data) {
    struct in_addr tmp_addr;
    tmp_addr.s_addr = handle;
    std::cout << "handle: " << handle << ", ip: " << inet_ntoa(tmp_addr) << ", push msg info: " << std::endl;
    std::cout << info << std::endl;
    return;
}

void WorkModeCallback(livox_status status, uint32_t handle,LivoxLidarAsyncControlResponse *response, void *client_data) {
    if (response == nullptr) {
        return;
    }
    printf("WorkModeCallack, status:%u, handle:%u, ret_code:%u, error_key:%u",
           status, handle, response->ret_code, response->error_key);

}

void QueryInternalInfoCallback(livox_status status, uint32_t handle,
                               LivoxLidarDiagInternalInfoResponse* response, void* client_data) {
    if (status != kLivoxLidarStatusSuccess) {
        printf("Query lidar internal info failed:%d.\n", status);
        QueryLivoxLidarInternalInfo(handle, QueryInternalInfoCallback, nullptr);
        return;
    }
    if (response == nullptr) {
        return;
    }
    uint8_t host_point_ipaddr[4] {0};
    uint16_t host_point_port = 0;
    uint16_t lidar_point_port = 0;

    uint8_t host_imu_ipaddr[4] {0};
    uint16_t host_imu_data_port = 0;
    uint16_t lidar_imu_data_port = 0;

    uint16_t off = 0;

    std::string versionApp;
    std::string versionLoader;
    std::string mac;
    std::string versionHardware;
    std::string curWorkState;
    std::string coreTemp;
    std::string powerUpCnt;
    std::string localTimeNow;
    std::string lastSyncTime;
    long timeOffset;
    long timeSyncType;
    std::string errorCode;
    long fwType;
    uint8_t detectMode;
    for (uint8_t i = 0; i < response->param_num; ++i) {
        LivoxLidarKeyValueParam* kv = (LivoxLidarKeyValueParam*)&response->data[off];
        if (kv->key == kKeyLidarPointDataHostIpCfg) {
            memcpy(host_point_ipaddr, &(kv->value[0]), sizeof(uint8_t) * 4);
            memcpy(&(host_point_port), &(kv->value[4]), sizeof(uint16_t));
            memcpy(&(lidar_point_port), &(kv->value[6]), sizeof(uint16_t));
        } else if (kv->key == kKeyLidarImuHostIpCfg) {
            memcpy(host_imu_ipaddr, &(kv->value[0]), sizeof(uint8_t) * 4);
            memcpy(&(host_imu_data_port), &(kv->value[4]), sizeof(uint16_t));
            memcpy(&(lidar_imu_data_port), &(kv->value[6]), sizeof(uint16_t));
        } else if (kv->key == kKeyVersionApp) {
            versionApp = str_format("%d.%d.%d.%d", kv->value[0], kv->value[1], kv->value[2], kv->value[3]);
        } else if (kv->key == kKeyVersionLoader) {
            versionLoader = str_format("%d.%d.%d.%d", kv->value[0], kv->value[1], kv->value[2], kv->value[3]);
        } else if (kv->key == kKeyMac) {
            mac = str_format("%02x:%02x:%02x:%02x:%02x:%02x", kv->value[0], kv->value[1], kv->value[2], kv->value[3], kv->value[4], kv->value[5]);
        } else if (kv->key == kKeyVersionHardware) {
            versionHardware = str_format("%d.%d.%d.%d", kv->value[0], kv->value[1], kv->value[2], kv->value[3]);
        } else if (kv->key == kKeyCurWorkState) {
            curWorkState = std::to_string(*((uint8_t *)kv->value));
        } else if (kv->key == kKeyCoreTemp) {
            coreTemp = std::to_string(*((int32_t *)kv->value));
        } else if (kv->key == kKeyPowerUpCnt) {
            powerUpCnt = std::to_string(*((uint32_t *)kv->value));
        } else if (kv->key == kKeyLocalTimeNow) {
            localTimeNow = time_format(*((uint64_t *)kv->value));
        } else if (kv->key == kKeyLastSyncTime) {
            lastSyncTime = time_format(*((uint64_t *)kv->value));
        } else if (kv->key == kKeyTimeOffset) {
            timeOffset = *((int64_t *)kv->value);
        } else if (kv->key == kKeyTimeSyncType) {
            timeSyncType = *((uint8_t *)kv->value);
        } else if (kv->key == kKeyLidarDiagStatus) {
            errorCode = std::to_string(*((uint16_t*)kv->value));
        } else if (kv->key == kKeyFwType) {
            fwType = *((uint8_t *)kv->value);
        } else if (kv->key == kKeyDetectMode) {
            detectMode = *((uint8_t *)kv->value);
        }
        off += sizeof(uint16_t) * 2;
        off += kv->length;
    }

    printf("Host point cloud ip addr:%u.%u.%u.%u, host point cloud port:%u, lidar point cloud port:%u.\n",
           host_point_ipaddr[0], host_point_ipaddr[1], host_point_ipaddr[2], host_point_ipaddr[3], host_point_port, lidar_point_port);

    printf("Host imu ip addr:%u.%u.%u.%u, host imu port:%u, lidar imu port:%u.\n",
           host_imu_ipaddr[0], host_imu_ipaddr[1], host_imu_ipaddr[2], host_imu_ipaddr[3], host_imu_data_port, lidar_imu_data_port);

}
Eigen::Matrix4d euler_xyz_to_rotation_matrix(double phi_z, double theta_y, double psi_x) {
    // 创建绕Z轴旋转的矩阵
    Eigen::Matrix3d Rz;
    Rz << cos(phi_z), -sin(phi_z), 0,
            sin(phi_z),  cos(phi_z), 0,
            0,        0,        1;

    // 创建绕Y轴旋转的矩阵
    Eigen::Matrix3d Ry;
    Ry << cos(theta_y), 0, sin(theta_y),
            0,          1, 0,
            -sin(theta_y), 0, cos(theta_y);

    // 创建绕X轴旋转的矩阵
    Eigen::Matrix3d Rx;
    Rx << 1, 0,         0,
            0, cos(psi_x), -sin(psi_x),
            0, sin(psi_x),  cos(psi_x);

    // 计算总的旋转矩阵
    Eigen::Matrix3d R = Rz * Ry * Rx;
    Eigen::Matrix4d R4 = Eigen::Matrix4d::Identity();
    R4.block<3,3>(0,0) = R;
    return R4;
}

/**
 * @brief 判断设备是否处于静止状态 (基于角速度模长 + 加速度滑动窗口方差 + 迟滞去抖)
 * @param gx, gy, gz 绕 X, Y, Z 轴角速度 (单位: rad/s)
 * @param ax, ay, az X, Y, Z 轴加速度 (单位: g)
 * @return bool true 为静止，false 为运动
 */
bool CheckImuStaticState(double gx, double gy, double gz, double ax, double ay, double az) {
    // ------------------- 核心算法参数配置 -------------------
    const size_t WINDOW_SIZE = 40;          // 滑动窗口大小 (约 0.2 秒)
    const double GYRO_THRES = 0.05;         // 角速度模长阈值 (rad/s，抑制自转)
    const double ACC_VAR_THRES = 0.0015;    // 加速度模长方差阈值 (g^2，抑制平移与振动)
    const int STATIC_CONFIRM_FRAMES = 100;  // 确认静止所需连续静止帧数 (约 0.5 秒)
    // -----------------------------------------------------

    static std::deque<double> acc_mag_buffer;
    static int continuous_quiet_frames = 0;
    static bool is_static = false;

    // 1. 计算角速度模长
    double gyro_mag = std::sqrt(gx * gx + gy * gy + gz * gz);

    // 2. 计算加速度模长并更新滑动窗口
    double acc_mag = std::sqrt(ax * ax + ay * ay + az * az);
    acc_mag_buffer.push_back(acc_mag);
    if (acc_mag_buffer.size() > WINDOW_SIZE) {
        acc_mag_buffer.pop_front();
    }

    // 3. 计算滑动窗口内加速度模长的方差 (Variance)
    double acc_var = 1.0; // 默认赋予大值，在窗口填满前默认设备在动
    if (acc_mag_buffer.size() == WINDOW_SIZE) {
        double sum = std::accumulate(acc_mag_buffer.begin(), acc_mag_buffer.end(), 0.0);
        double mean = sum / WINDOW_SIZE;
        double sq_sum = 0.0;
        for (double val : acc_mag_buffer) {
            sq_sum += (val - mean) * (val - mean);
        }
        acc_var = sq_sum / WINDOW_SIZE;
    }

    // 4. 判断当前单帧是否满足“瞬间平静”条件
    bool is_current_frame_quiet = (gyro_mag < GYRO_THRES) && (acc_var < ACC_VAR_THRES);

    // 5. 快出慢进去抖机制
    if (is_current_frame_quiet) {
        continuous_quiet_frames++;
        // 只有连续平静达到设定帧数，才确认进入静止状态
        if (continuous_quiet_frames >= STATIC_CONFIRM_FRAMES) {
            is_static = true;
            continuous_quiet_frames = STATIC_CONFIRM_FRAMES; // 防止溢出
        }
    } else {
        // 敏感触发：只要检测到微小运动，立刻打破静止状态
        continuous_quiet_frames = 0;
        is_static = false;
    }

    return is_static;
}

//获取IMU数据回调
void ImuDataCallback(uint32_t handle, const uint8_t dev_type,  LivoxLidarEthernetPacket* data, void* client_data) {
    if (data == nullptr) {
        return;
    }
    if (data->data_type == kLivoxLidarImuData) {
        auto pData = (LivoxLidarImuRawPoint *) data->data;
#ifdef IMU_UPSIDEDOWN
        pData->gyro_x = -pData->gyro_x;
      pData->gyro_z = -pData->gyro_z;
      pData->acc_x = -pData->acc_x;
      pData->acc_z = -pData->acc_z;
#endif
//        printf("imudata gyro_x:%f gyro_y:%f gyro_z:%f acc_x:%f acc_y:%f acc_z:%f\n",pData->gyro_x,pData->gyro_y,pData->gyro_z,pData->acc_x,pData->acc_y,pData->acc_z);

        // 1. 静态变量：用于在多次回调之间持久化保存状态
        static double current_pitch = 0.0; // 融合后的俯仰角 (弧度)
        static double current_roll = 0.0;  // 融合后的横滚角 (弧度)
        static auto last_time = std::chrono::steady_clock::now();
        static bool is_first_frame = true;

        // 2. 提取当前帧的 IMU 原始数据
        double gx = pData->gyro_x; // 绕 X 轴角速度 (rad/s)
        double gy = pData->gyro_y; // 绕 Y 轴角速度 (rad/s)
        double gz = pData->gyro_z; // 绕 Z 轴角速度 (rad/s)
        double ax = pData->acc_x;  // X 轴加速度 (g)
        double ay = pData->acc_y;  // Y 轴加速度 (g)
        double az = pData->acc_z;  // Z 轴加速度 (g)

        // =================================================================
        // 【新增逻辑】调用独立封装的静止检测函数，直接控制全局开关
        g_is_static = CheckImuStaticState(gx, gy, gz, ax, ay, az);
        // =================================================================

        // 3. 计算纯加速度计得到的静态倾角 (重力对齐)
        // Mid360 的横滚和俯仰对应的加速度公式：
        double acc_pitch = std::atan2(ax, std::sqrt(ay * ay + az * az));
        double acc_roll  = std::atan2(-ay, az);

        // 4. 第一帧特殊处理：直接用加速度计初始化角度
        if (is_first_frame) {
            current_pitch = acc_pitch;
            current_roll = acc_roll;
            is_first_frame = false;
            return;
        }

        // 5. 计算两次回调之间的时间差 dt (秒)
        double dt = 0.005;

        // 6. 互补滤波融合 (Alpha 核心系数)
        // 0.98 表示 98% 信任角速度积分（动态时防抖），2% 信任加速度计（静态时纠偏）
//        const double alpha = 0.98;
        const double alpha = 0.8;

        // 根据 Mid360 坐标轴定义：
        // 绕 X 轴旋转 (gyro_x) 改变的是 Roll
        // 绕 Y 轴旋转 (gyro_y) 改变的是 Pitch
        current_pitch = alpha * (current_pitch + gy * dt) + (1.0 - alpha) * acc_pitch;
        current_roll  = alpha * (current_roll  + gx * dt) + (1.0 - alpha) * acc_roll;

        // 7. 转换为角度制（Degree）方便你直观观察
        double pitch_deg = current_pitch * 180.0 / M_PI;
        double roll_deg  = current_roll * 180.0 / M_PI;
        // 8. 打印输出
//        std::printf("[IMU 倾角状态] Pitch(俯仰): %7.2f° | Roll(横滚): %7.2f° | dt: %.4fs\n",
//                    pitch_deg, roll_deg, dt);

        // 关键点：因为要把倾斜的雷达扶正，所以旋转角度要取负号 (-)
        Eigen::AngleAxisd roll_inverse(-current_roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitch_inverse(-current_pitch, Eigen::Vector3d::UnitY());
        // 组合成最终的校正矩阵 (注意外积顺序：先变换的在右边)
        Eigen::Matrix3f R_correct = (pitch_inverse * roll_inverse).matrix().cast<float>();
        Rtilt.block<3, 3>(0, 0) = R_correct;
    }
}

void LidarInfoChangeCallback(const uint32_t handle, const LivoxLidarInfo* info, void* client_data) {
    if (info == nullptr) {
        printf("lidar info change callback failed, the info is nullptr.\n");
        return;
    }
    printf("LidarInfoChangeCallback Lidar handle: %u SN: %s\n", handle, info->sn);
    // set the work mode to kLivoxLidarNormal, namely start the lidar
    SetLivoxLidarWorkMode(handle, kLivoxLidarNormal, WorkModeCallback, nullptr);

    QueryLivoxLidarInternalInfo(handle, QueryInternalInfoCallback, nullptr);
    mLidarHandle = handle;
    {
        FovCfg fovCfg0, fovCfg1;

        fovCfg0.yaw_start = 0;
        fovCfg0.yaw_stop = 360;
        fovCfg0.pitch_start = -10;
        fovCfg0.pitch_stop = 60;

//        fovCfg1.yaw_start = 210;
//        fovCfg1.yaw_stop = 360;
//        fovCfg1.pitch_start = -10;
//        fovCfg1.pitch_stop = 60;

        livox_status status;

        status = SetLivoxLidarFovCfg0(mLidarHandle, &fovCfg0, SetFovCfgCallback, nullptr);
        if (status != kLivoxLidarStatusSuccess) {
            throw std::runtime_error("SetLivoxLidarFovCfg0 failed! Status = " + std::to_string(status));
        }
//        status = SetLivoxLidarFovCfg1(mLidarHandle, &fovCfg1, SetFovCfgCallback, nullptr);
//        if (status != kLivoxLidarStatusSuccess) {
//            throw std::runtime_error("SetLivoxLidarFovCfg1 failed! Status = " + std::to_string(status));
//        }
        status = EnableLivoxLidarFov(mLidarHandle, 0b00000001, EnableFovCfgCallback, nullptr);
        if (status != kLivoxLidarStatusSuccess) {
            throw std::runtime_error("EnableLivoxLidarFov failed! Status = " + std::to_string(status));
        }
    }
}

// --- 线程 1：相机持续采集 (15Hz) ---
void CameraCaptureThread(rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher, rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr image_compressed_publisher, rclcpp::Node::SharedPtr node) {
    std::string frame_id = node->get_parameter("cam_frame_id").as_string();
    cv::Matx33d tmp_undistort_k;
    cv::Size tmp_undistort_img_size;
    cv::UMat umat;
    long t_s = common_utils::currentTimeMilliseconds();
    while (rclcpp::ok()) {
        // 调用你补充好的相机抓拍缓存函数
        if (pCameraSingle == nullptr) {
            break;
        }
        if (g_image_raw_compressed) {
            auto msg = std::make_unique<sensor_msgs::msg::CompressedImage>();
            msg->header.stamp = node->now();
            msg->header.frame_id = frame_id;
            msg->format = "jpeg";
            pCameraSingle->shootAutoToCacheMatRealtime(&msg->data);
            image_compressed_publisher->publish(std::move(msg));
        } else {
            Mat mat = pCameraSingle->shootAutoToCacheMatRealtime();;
            auto msg = std::make_unique<sensor_msgs::msg::Image>();
            cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
            cv_ptr->header.stamp = node->now();
            cv_ptr->header.frame_id = frame_id;
            cv_ptr->encoding = sensor_msgs::image_encodings::BGR8;

            if (g_undistort && pFisheyeUndistorter) {
                {
                    std::lock_guard<std::mutex> lock(g_undistort_params_mutex);
                    tmp_undistort_k = g_undistort_k;
                    tmp_undistort_img_size = g_undistort_img_size;
                }
#if SOFT_UNDISTORT
                cv::Mat ud_mat;
//                long t = common_utils::currentTimeMilliseconds();
                pFisheyeUndistorter->undistort(mat, ud_mat, tmp_undistort_img_size, tmp_undistort_k, g_undistort_r);
//                printf("image undistorted, cost: %f s\n", (common_utils::currentTimeMilliseconds()-t)/1000.0f);
                mat = ud_mat;
#else
                long t = common_utils::currentTimeMilliseconds();
                pFisheyeUndistorter->undistort(mat, umat, tmp_undistort_k, g_undistort_r);
                mat = umat.getMat(cv::ACCESS_READ).clone();
                printf("image undistorted, cost: %f s\n", (common_utils::currentTimeMilliseconds()-t)/1000.0f);
#endif
            }

            cv_ptr->image = mat;
            cv_ptr->toImageMsg(*msg);
//            printf("shoot, cost: %f s\n", (common_utils::currentTimeMilliseconds()-t_s)/1000.0f);
//            auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", mat).toImageMsg();
//            msg->header.stamp = node->now();
//            msg->header.frame_id = frame_id;
            image_publisher->publish(std::move(msg));
        }

        long t_e = common_utils::currentTimeMilliseconds();
//        printf("image published, freq: %f Hz\n", 1000.0f/(t_e-t_s));
        t_s = t_e;
    }
}

// --- 线程 2：点云处理与赋色 (速率取决于此线程性能) ---
void ColorizationWorkerThread(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher, rclcpp::Node::SharedPtr node) {
    cv::Mat bgr_image;
    
    pfw::PlyFastWriter writer;
    writer.reserve(g_frames_per_publish*96);

    while (rclcpp::ok()) {
        std::vector<ColoredPoint> points_to_process;

        // 1. 阻塞等待，直到拿到最新的点云帧
        {
            std::unique_lock<std::mutex> lock(g_cloud_mutex);
            g_cloud_cv.wait(lock, [] { return (!g_cloud_queue.empty() && !g_image_queue.empty()) || !rclcpp::ok(); });
            if (!rclcpp::ok()) break;

            points_to_process = std::move(g_cloud_queue.front());
            bgr_image = std::move(g_image_queue.front());
            g_cloud_queue.pop();
            g_image_queue.pop();
        }

        long t_c = common_utils::currentTimeMilliseconds();

        std::string src_dir = "/alpcer3d/office_" + std::to_string(t_c) + "/";
        if (fs::exists(src_dir)) {
            if (!fs::is_directory(src_dir)) {
                throw std::runtime_error("not directory:"+src_dir);
            }
        } else {
            fs::create_directories(src_dir);
        }

        int width = image_actual_width, height = image_actual_height;
        cv::Mat image_left = bgr_image(Rect(0, 0, width/2, height));
        cv::Mat image_right = bgr_image(Rect(width/2, 0, width/2, height));

        //图像
        cv::Mat img1_rect, img2_rect;
        cv::remap(image_left, img1_rect, g_map_x1, g_map_y1, cv::INTER_LINEAR);
        cv::remap(image_right, img2_rect, g_map_x2, g_map_y2, cv::INTER_LINEAR);
        cv::imwrite(src_dir + "im0.png", img1_rect);
        cv::imwrite(src_dir + "im1.png", img2_rect);

        cv::imwrite(src_dir + "pano_im0.png", image_left);
        cv::imwrite(src_dir + "pano_im1.png", image_right);
        cv::Mat image_left_align, image_right_align;
        cv::remap(image_left, image_left_align, g_fisheye_stereo_map_x1, g_fisheye_stereo_map_y1, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        cv::remap(image_right, image_right_align, g_fisheye_stereo_map_x2, g_fisheye_stereo_map_y2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        cv::imwrite(src_dir + "pano_im0_align.png", image_left_align);
        cv::imwrite(src_dir + "pano_im1_align.png", image_right_align);

        cv::Mat image_panorama_left, image_panorama_right;
        cv::remap(image_left_align, image_panorama_left, g_panorama_map_x1, g_panorama_map_y1, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        cv::remap(image_right_align, image_panorama_right, g_panorama_map_x2, g_panorama_map_y2, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        cv::imwrite(src_dir + "pano_expand_im0.png", image_panorama_left);
        cv::imwrite(src_dir + "pano_expand_im1.png", image_panorama_right);

        //视差
        size_t count = points_to_process.size();
        cv::Mat parallax(img1_rect.size(), CV_32F);
        parallax.setTo(std::numeric_limits<float>::quiet_NaN());
        cv::Mat depth_left(img1_rect.size(), CV_32F), depth_right(img1_rect.size(), CV_32F), r2l_map(img1_rect.size(), CV_32SC2);
        depth_left.setTo(0.0f);
        depth_right.setTo(0.0f);
        r2l_map.setTo(cv::Vec2i(-1,-1));
        cv::Mat mask(img1_rect.size(), CV_8UC3);
        mask.setTo(cv::Scalar(0, 0, 0));
        std::vector<cv::Point3f> pts3d_left, pts3d_right;
        std::vector<cv::Point2f> pts2d_left, pts2d_right;

        pts3d_left.reserve(count);
        pts3d_right.reserve(count);

        for (uint64_t i=0; i<count; i++) {
            cv::Mat p = (cv::Mat_<double>(4, 1) << points_to_process[i].x, points_to_process[i].y, points_to_process[i].z, 1.0);
            cv::Mat pt_3d_left = g_extrinsic_left * p;
            cv::Mat pt_3d_right = g_extrinsic_right * p;

            pts3d_left.push_back(cv::Point3f(pt_3d_left.at<double>(0)*1000.0, pt_3d_left.at<double>(1)*1000.0, pt_3d_left.at<double>(2)*1000.0));
            pts3d_right.push_back(cv::Point3f(pt_3d_right.at<double>(0)*1000.0, pt_3d_right.at<double>(1)*1000.0, pt_3d_right.at<double>(2)*1000.0));
        }

        float pano_ndisp = 0.0f;
        {
            //计算panorama视差
            float alpha_min = static_cast<float>(-CV_PI/2.0);
            float beta_min = static_cast<float>(-CV_PI/2.0);
            cv::Size pano_size = image_panorama_left.size();
            cv::Mat parallax_pano(pano_size, CV_32F);
            parallax_pano.setTo(std::numeric_limits<float>::quiet_NaN());
            cv::Mat depth_pano_left(pano_size, CV_32F);
            depth_pano_left.setTo(0.0f);

            cv::Matx33f r1(g_r1);
            for (uint64_t i=0; i<count; i++) {
                auto& p3l_raw = pts3d_left[i];
                cv::Point3f p3l = r1*p3l_raw;
                cv::Point3f p3r = p3l;
                p3r.x -= g_baseline;
                if (p3l.z > 0 && p3r.z > 0) {

//                    auto& p3r = pts3d_right[i];
                    cv::Point2f p1 = project3DToGrid(p3l, pano_size, alpha_min, CV_PI, beta_min, CV_PI);
                    cv::Point2f p2 = project3DToGrid(p3r, pano_size, alpha_min, CV_PI, beta_min, CV_PI);
                    if (p1.x >=0 && p1.x <parallax_pano.cols && p1.y >= 0 && p1.y <parallax_pano.rows && p2.x >=0 && p2.x <parallax_pano.cols && p2.y >= 0 && p2.y <parallax_pano.rows) {
                        auto& dp_left = depth_pano_left.at<float>(static_cast<int>(p1.y), static_cast<int>(p1.x));
                        float rl2 = p3l.x*p3l.x + p3l.y*p3l.y + p3l.z*p3l.z;
                        double incidence = std::acos(p3l.z/std::sqrt(rl2))*180.0/CV_PI;
                        if (incidence > g_valid_incidence) {
                            continue;
                        }
                        if (dp_left <= 0.0f || rl2 < dp_left) {
                            float disp = p1.x-p2.x;
                            if (disp > pano_ndisp) {
                                pano_ndisp = disp;
                            }
                            int u = cvRound(p1.x);
                            int v = cvRound(p1.y);
                            parallax_pano.at<float>(v, u) = disp;
                            dp_left = rl2;
                        }
                    }
                }
            }
            savePFM(src_dir + "GT.pfm", parallax_pano);
        }

        cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        cv::Vec3d translation(0,0,0);
        cv::projectPoints(pts3d_left, g_r1, translation, g_k1, distCoeffs, pts2d_left);
        cv::projectPoints(pts3d_right, g_r2, translation, g_k2, distCoeffs, pts2d_right);

        cv::Vec3b pix_lv(128, 128, 128), pix_av(255, 255, 255);
        float ndisp = 0.0f;

        for (uint64_t i=0; i<count; i++) {
            auto& p3l = pts3d_left[i];
            auto& p3r = pts3d_right[i];
            auto& p1 = pts2d_left[i];
            auto& p2 = pts2d_right[i];

            if (p3l.z > 0 && p3r.z > 0 && p1.x >=0 && p1.x <parallax.cols && p1.y >= 0 && p1.y <parallax.rows && p2.x >=0 && p2.x <parallax.cols && p2.y >= 0 && p2.y <parallax.rows) {
                auto& dp_left = depth_left.at<float>(static_cast<int>(p1.y), static_cast<int>(p1.x));
                float rl2 = p3l.x*p3l.x + p3l.y*p3l.y + p3l.z*p3l.z;
                if (dp_left <= 0.0f || rl2 < dp_left) {
                    float disp = p1.x-p2.x;
                    if (disp > ndisp) {
                        ndisp = disp;
                    }
                    parallax.at<float>(static_cast<int>(p1.y), static_cast<int>(p1.x)) = disp;
                    dp_left = rl2;

                    //mask
                    auto& dp_right = depth_right.at<float>(static_cast<int>(p2.y), static_cast<int>(p2.x));
                    float rr2 = std::sqrt(p3r.x*p3r.x + p3r.y*p3r.y + p3r.z*p3r.z);
                    if (dp_right <= 0.0f) {
                        mask.at<cv::Vec3b>(static_cast<int>(p1.y), static_cast<int>(p1.x)) = pix_av;
                        dp_right = rr2;
                    } else if (rr2 < (dp_right - 30.0f)) {//增加阈值排除点云厚度影响
                        //这个位置的上一个点被遮挡了
                        auto& lxy = r2l_map.at<cv::Vec2i>(p2.y, p2.x);
                        mask.at<cv::Vec3b>(lxy[1], lxy[0]) = pix_lv;//上一个点的对应左图像点设为遮挡点
                        mask.at<cv::Vec3b>(static_cast<int>(p1.y), static_cast<int>(p1.x)) = pix_av;//当前做图像点设为共有点
                        dp_right = rr2;
                    }

                    //更新当前索引表
                    r2l_map.at<cv::Vec2i>(p2.y, p2.x) = cv::Vec2i(static_cast<int>(p1.x), static_cast<int>(p1.y));
                }
            }
        }

        savePFM(src_dir + "disp0GT.pfm", parallax);
        cv::imwrite(src_dir + "mask0nocc.png", mask);

        //评估视差图比例
        size_t valid_pix_count = 0;
        for (int i=0; i<parallax.cols; i++) {
            for (int j=0; j<parallax.rows; j++) {
                if (!std::isnan(parallax.at<float>(j,i))) {
                    valid_pix_count++;
                }
            }
        }
        double valid_rate = valid_pix_count*100.0/(parallax.cols*parallax.rows);
        spdlog::info("视差图片像素有效率：{}%", valid_rate);

        //参数
        // 1. 生成 calib.txt
        generate_calib_txt(src_dir + "calib.txt",
                           g_k1.at<double>(0,0), g_k1.at<double>(1,1), g_k1.at<double>(0,2), g_k1.at<double>(1,2), // fx, fy, cx, cy
                           0, g_baseline,                        // doffs, baseline
                           width, height, ndisp, pano_ndisp);                    // width, height

        // 2. 生成 cameras.txt
        std::vector<ColmapCamera> cameras = {
                {0, "OPENCV_FISHEYE", width, height, {g_fisheye_k[0](0,0), g_fisheye_k[0](1,1), g_fisheye_k[0](0,2), g_fisheye_k[0](1,2), g_fisheye_d[0][0], g_fisheye_d[0][1], g_fisheye_d[0][2], g_fisheye_d[0][3]}},
                {1, "OPENCV_FISHEYE", width, height, {g_fisheye_k[1](0,0), g_fisheye_k[1](1,1), g_fisheye_k[1](0,2), g_fisheye_k[1](1,2), g_fisheye_d[1][0], g_fisheye_d[1][1], g_fisheye_d[1][2], g_fisheye_d[1][3]}}
        };
        generate_cameras_txt(src_dir + "cameras.txt", cameras);
        auto cl = extract_colmap_pose_from_w2c(g_extrinsic_left);
        auto cr = extract_colmap_pose_from_w2c(g_extrinsic_right);

        // 3. 生成 images.txt
        std::vector<ColmapImage> images = {
                {0, cl.qw, cl.qx, cl.qy, cl.qz, cl.tx, cl.ty, cl.tz, 0, "im0.png"},
                {1, cr.qw, cr.qx, cr.qy, cr.qz, cr.tx, cr.ty, cr.tz, 0, "im1.png"}
        };
        generate_images_txt(src_dir + "images.txt", images);

        //点云
        for (const auto& pt : points_to_process) {
            writer.addPoint({(float)pt.x, (float)pt.y, (float)pt.z, (uint8_t)pt.reflectivity, pt.r, pt.g, pt.b});
        }

        writer.write(src_dir+"pc_gt.ply");
        writer.clear();

        {
            FileStorage fs(src_dir + "extrinsic.xml", FileStorage::WRITE);
            if (!fs.isOpened()) {
                throw std::runtime_error("Open file storage failed!");
            }
            fs << "left" << g_extrinsic_left;
            fs << "right" << g_extrinsic_right;
            fs.release();
        }

        printf("data published, processing cost: %f s\n", (common_utils::currentTimeMilliseconds()-t_c)/1000.0f);
    }
}

bool waitingUntilAngleArrived(SMC::StepperMotor& motor, int deviceId, int64_t angleToSet, int64_t tolerance = 2) {
    int64_t angleRead;
    int read_fail_count = 0;
    do {
        common_utils::sleepMilliseconds(1000);
        bool ret = motor.readMultiCircleAngle(deviceId, angleRead);
        if (ret) {
            read_fail_count = 0;
            std::cout << "read angle:" << angleRead <<std::endl;
        } else {
            read_fail_count++;
            if (read_fail_count > 5) {
                return false;
            }
        }

    } while (std::abs(angleRead - angleToSet) > tolerance);
    common_utils::sleepMilliseconds(1000);
    return true;
}
bool setMotorAngle(SMC::StepperMotor& motor, int deviceId, int64_t angleToSet, bool waitUntilArrived, uint32_t speed, int64_t tolerance = 2) {
    bool ret = motor.multiCircleAngleClosedLoopControl2(deviceId, angleToSet, speed, false);
    if (ret && waitUntilArrived) {
        return waitingUntilAngleArrived(motor, deviceId, angleToSet, tolerance);
    }
    return ret;
}

void MotorControllerThread() {
    SMC::StepperMotor motor("/dev/ttyS7", SMC::StepperMotor::SERIES_MG_V3);
    int deviceId = 1;
    float angle_interval = 10.0f;
    int64_t angle_stamp_interval = static_cast<int64_t>(angle_interval*100);
    bool is_new_round = true;
    while (rclcpp::ok()) {
        if (g_is_static) {
            if (is_new_round) {
                spdlog::info("新一轮扫描开始");
                uint32_t speed = 1500;
                int64_t angle = 0*100;

                int64_t max_angle_stamp = 360*100;
                int64_t cur_angle_stamp = 0;
                while (cur_angle_stamp < max_angle_stamp) {
                    spdlog::info("移动到{}度", cur_angle_stamp/100.0f);
                    //移动
                    bool ret = setMotorAngle(motor, deviceId, cur_angle_stamp, true, speed, 3);
                    if (ret) {
                        g_scanning = true;
                        //挂起直到扫描完成
                        {
                            std::unique_lock<std::mutex> lock(g_motor_mutex);
                            g_motor_cv.wait(lock, [] { return (!g_scanning || !rclcpp::ok()); });
                            if (!rclcpp::ok()) break;
                        }
                        //扫描完成，可以继续移动
                        cur_angle_stamp += angle_stamp_interval;
                    } else {
                        throw std::runtime_error("设置角度或确认角度位置失败!");
                    }
                }
                if (!setMotorAngle(motor, deviceId, 0, true, speed, 3)) {
                    throw std::runtime_error("设置归零角度或确认角度位置失败!");
                }
                //一圈扫描完成，标记当前已处理完成，下一次检测到非静止时刷新
                is_new_round = false;
                spdlog::info("一轮扫描完成，等待下一次静止");
            }
        } else {
            is_new_round = true;
        }
        common_utils::sleepMilliseconds(500);
    }
}

bool parseResolution(const std::string& resolution, int& width, int& height) {
    std::stringstream ss(resolution);
    char separator;

    ss >> width >> separator >> height;

    // 检查解析是否成功，并且分隔符是 'x'
    if (!ss.fail() && separator == 'x') {
        return true;
    }

    return false;
}

/**
 * @brief 检查像素点是否在180度FOV的鱼眼圆形区域内
 */
inline bool isInFisheyeCircle(double u, double v, const cv::Matx33d& K) {
    double cx = K(0, 2);
    double cy = K(1, 2);
    double fx = K(0, 0);

    // 估算鱼眼180度视野下的半径 (r = f * pi / 2)
    double r_max = fx * CV_PI / 2.0;
    double dx = u - cx;
    double dy = v - cy;
    return (dx * dx + dy * dy) <= (r_max * r_max);
}

/**
 * @brief 生成基于球面旋转的鱼眼反向映射表（支持 cv::Mat 格式的 R）
 */
void generateFisheyeRotateMap(const cv::Size& imageSize,
                              const cv::Matx33d& K,
                              const cv::Vec4d& D,
                              const cv::Mat& R_in, // 这里修改为 cv::Mat 格式
                              cv::Mat& map_x,
                              cv::Mat& map_y) {

    // 1. 安全检查：确保输入的 R 是 3x3 的矩阵
    CV_Assert(R_in.rows == 3 && R_in.cols == 3);

    // 2. 类型转换：不管标定出来的是 float(CV_32F) 还是 double(CV_64F)，统一转为 double 以防丢失精度
    cv::Mat R_double;
    R_in.convertTo(R_double, CV_64F);

    // 3. 将 cv::Mat 赋值给 cv::Matx33d（OpenCV 支持兼容尺寸的直接赋值）
    cv::Matx33d R = R_double;

    // =======================================================
    // 后续的算法核心逻辑完全不需要变动
    // =======================================================
    map_x = cv::Mat(imageSize, CV_32FC1, cv::Scalar(-1.0f));
    map_y = cv::Mat(imageSize, CV_32FC1, cv::Scalar(-1.0f));

    double fx = K(0, 0), fy = K(1, 1);
    double cx = K(0, 2), cy = K(1, 2);
    double k1 = D[0], k2 = D[1], k3 = D[2], k4 = D[3];

    // 旋转矩阵的逆 (正交阵的逆等于其转置)
    cv::Matx33d R_inv = R.t();

    for (int v = 0; v < imageSize.height; ++v) {
        for (int u = 0; u < imageSize.width; ++u) {
            if (!isInFisheyeCircle(u, v, K)) {
                continue;
            }

            double x_dst = (u - cx) / fx;
            double y_dst = (v - cy) / fy;
            double r_dst = std::sqrt(x_dst * x_dst + y_dst * y_dst);

            double theta = r_dst;
            double sin_theta = 0.0, cos_theta = 1.0;

            if (r_dst > 1e-8) {
                for (int iter = 0; iter < 5; ++iter) {
                    double theta2 = theta * theta;
                    double theta4 = theta2 * theta2;
                    double theta6 = theta4 * theta2;
                    double theta8 = theta4 * theta4;

                    double f = theta * (1.0 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8) - r_dst;
                    double df = 1.0 + 3.0*k1*theta2 + 5.0*k2*theta4 + 7.0*k3*theta6 + 9.0*k4*theta8;
                    theta -= f / df;
                }
                sin_theta = std::sin(theta);
                cos_theta = std::cos(theta);
            }

            double scale = (r_dst > 1e-8) ? (sin_theta / r_dst) : 0.0;
            cv::Vec3d P_dst(x_dst * scale, y_dst * scale, cos_theta);

            cv::Vec3d P_src = R_inv * P_dst;
            if (P_src[2] <= 1e-6) continue;

            double x_src = P_src[0] / P_src[2];
            double y_src = P_src[1] / P_src[2];
            double r_src = std::sqrt(x_src * x_src + y_src * y_src);

            double theta_src = std::atan(r_src);
            double theta_src2 = theta_src * theta_src;
            double theta_src4 = theta_src2 * theta_src2;
            double theta_src6 = theta_src4 * theta_src2;
            double theta_src8 = theta_src4 * theta_src4;

            double theta_d = theta_src * (1.0 + k1*theta_src2 + k2*theta_src4 + k3*theta_src6 + k4*theta_src8);
            double scale_d = (r_src > 1e-8) ? (theta_d / r_src) : 1.0;

            double u_src = fx * (x_src * scale_d) + cx;
            double v_src = fy * (y_src * scale_d) + cy;

            if (u_src >= 0 && u_src < imageSize.width && v_src >= 0 && v_src < imageSize.height) {
                map_x.at<float>(v, u) = static_cast<float>(u_src);
                map_y.at<float>(v, u) = static_cast<float>(v_src);
            }
        }
    }
}

/**
 * @brief 构建鱼眼图像横向经纬度展开（Equirectangular）的 Remap 映射表
 * * @param K 鱼眼相机内参矩阵 (cv::Matx33d)
 * @param D 鱼眼相机畸变系数 (cv::Vec4d -> k1, k2, k3, k4)
 * @param outSize 目标展开图的尺寸 (cv::Size)
 * @param fovHorizontal 水平展开总视角 (弧度, 例如 M_PI 表示 180度)
 * @param fovVertical 垂直展开总视角 (弧度, 例如 M_PI/2 表示 90度)
 * @param mapx 输出的 X 映射表 (CV_32FC1)
 * @param mapy 输出的 Y 映射表 (CV_32FC1)
 */
void buildFisheyeUnwrapMap(
        const cv::Matx33d& K,
        const cv::Vec4d& D,
        const cv::Size& outSize,
        double fovHorizontal,
        double fovVertical,
        cv::Mat& mapx,
        cv::Mat& mapy)
{
    mapx.create(outSize, CV_32FC1);
    mapy.create(outSize, CV_32FC1);

    // 计算展开角度范围
    double alpha_min = -fovHorizontal / 2.0;
    double alpha_range = fovHorizontal;
    double beta_min = -fovVertical / 2.0;
    double beta_range = fovVertical;

    std::vector<cv::Point3d> ray3D_list;
    ray3D_list.reserve(outSize.width * outSize.height);

    for (int v = 0; v < outSize.height; ++v) {
        double beta = beta_min + (static_cast<double>(v) / (outSize.height - 1)) * beta_range;
        double cos_beta = std::cos(beta);
        double sin_beta = std::sin(beta);

        for (int u = 0; u < outSize.width; ++u) {
            double alpha = alpha_min + (static_cast<double>(u) / (outSize.width - 1)) * alpha_range;
            double cos_alpha = std::cos(alpha);
            double sin_alpha = std::sin(alpha);

            double X = sin_alpha;
            double Y = cos_alpha * sin_beta;
            double Z = cos_alpha * cos_beta;

            ray3D_list.push_back(cv::Point3d(X, Y, Z));
        }
    }

    std::vector<cv::Point2d> projected_points;
    cv::fisheye::projectPoints(ray3D_list, projected_points, cv::Vec3d(0,0,0), cv::Vec3d(0,0,0), K, D);

    // 填充至 remap 映射表
    int idx = 0;
    for (int v = 0; v < outSize.height; ++v) {
        float* ptr_x = mapx.ptr<float>(v);
        float* ptr_y = mapy.ptr<float>(v);
        for (int u = 0; u < outSize.width; ++u) {
            ptr_x[u] = static_cast<float>(projected_points[idx].x);
            ptr_y[u] = static_cast<float>(projected_points[idx].y);
            idx++;
        }
    }
}

void calculateRectifiedFOV(const cv::Mat& P, const cv::Size& newImageSize) {
    // 提取校正后的焦距
    double fx = P.at<double>(0, 0);
    double fy = P.at<double>(1, 1);

    double W = static_cast<double>(newImageSize.width);
    double H = static_cast<double>(newImageSize.height);

    // 标准针孔 FOV 计算 (转换为角度)
    double fov_x = 2.0 * std::atan(W / (2.0 * fx)) * 180.0 / CV_PI;
    double fov_y = 2.0 * std::atan(H / (2.0 * fy)) * 180.0 / CV_PI;
    double fov_d = 2.0 * std::atan(std::hypot(W, H) / (2.0 * fx)) * 180.0 / CV_PI;

    std::cout << "--- Rectified Image FOV ---" << std::endl;
    std::cout << "Horizontal FOV (FOV_x): " << fov_x << " degrees" << std::endl;
    std::cout << "Vertical FOV   (FOV_y): " << fov_y << " degrees" << std::endl;
    std::cout << "Diagonal FOV   (FOV_D): " << fov_d << " degrees" << std::endl;
}

void initCalculation() {
    cv::Mat R, T;
    cv::Matx33d k[2];    /*****    摄像机内参数矩阵    ****/
    cv::Vec4d d[2];     /* 摄像机的4个畸变系数：k1,k2,k3,k4*/
    {
        FileStorage fs("/camera/camera_stereo.xml", FileStorage::READ);
        if (!fs.isOpened()) {
            throw std::runtime_error("Open file storage failed!");
        }
        fs["R"] >> R;
        fs["T"] >> T;
        fs["K0"] >> k[0];
        fs["D0"] >> d[0];
        fs["K1"] >> k[1];
        fs["D1"] >> d[1];
        fs.release();
    }

    int width = image_actual_width, height = image_actual_height;
    cv::Size image_size = cv::Size(width/2, height);

// ==================== 新增：基于目标 FOV 与内切正方形的精确控制 ====================

    // 1. 定义你期望保留的目标 FOV（单位：度，例如 100.0 度）
    double target_fov_deg = 120.0;
    double theta = (target_fov_deg / 2.0) * CV_PI / 180.0; // 转换为弧度半视角

    // 2. 借助鱼眼畸变模型，计算该 FOV 在原始鱼眼图像上的物理投影半径（以左目为例）
    double f_fish = k[0](0, 0); // 鱼眼原始焦距
    double k1 = d[0][0], k2 = d[0][1], k3 = d[0][2], k4 = d[0][3];
    double theta2 = theta * theta;
    double theta_d = theta * (1.0 + k1 * theta2 + k2 * theta2 * theta2 + k3 * theta2 * theta2 * theta2 + k4 * theta2 * theta2 * theta2 * theta2);
    double r_fov = f_fish * theta_d;

    // 3. 计算圆的内切正方形边长 S，并向下取偶数以优化内存对齐
    int S = static_cast<int>(std::sqrt(2.0) * r_fov);
    S = S - S%32;
    cv::Size rectified_image_size(S, S);

    int panorama_s;
    {
        double tmp_theta = (180.0 / 2.0) * CV_PI / 180.0;
        double tmp_theta2 = tmp_theta * tmp_theta;
        double tmp_theta_d = tmp_theta * (1.0 + k1 * tmp_theta2 + k2 * tmp_theta2 * tmp_theta2 + k3 * tmp_theta2 * tmp_theta2 * tmp_theta2 + k4 * tmp_theta2 * tmp_theta2 * tmp_theta2 * tmp_theta2);
        double tmp_r_fov = f_fish * tmp_theta_d;
        panorama_s = tmp_r_fov*2;
        panorama_s = panorama_s - panorama_s%32;
    }
    cv::Size panorama_size(panorama_s, panorama_s);

    std::cout << ">>> 目标 FOV: " << target_fov_deg << " 度" << std::endl;
    std::cout << ">>> 对应的鱼眼圆半径: " << r_fov << " 像素" << std::endl;
    std::cout << ">>> 动态计算出的内切正方形大小 (重映射目标尺寸): " << S << "x" << S << std::endl;

    // 4. 调用 OpenCV 计算基础校正矩阵（这里将 rectified_image_size 传给 newImageSize 参数）
    cv::Mat R1, R2, P1, P2, Q;
    std::cout << "image_size:" << image_size << std::endl;
    std::cout << "rectified_image_size:" << rectified_image_size << std::endl;
    std::cout << "k0:" << k[0] << std::endl;
    std::cout << "k1:" << k[1] << std::endl;
    std::cout << "d0:" << d[0] << std::endl;
    std::cout << "d1:" << d[1] << std::endl;
    std::cout << "R:" << R << std::endl;
    std::cout << "T:" << T << std::endl;
    cv::fisheye::stereoRectify(k[0], d[0], k[1], d[1], image_size, R, T, R1, R2, P1, P2, Q,
                               cv::CALIB_ZERO_DISPARITY, rectified_image_size, 0.0, 1.0);
    std::cout << "R1 :" << R1 << std::endl;
    std::cout << "R2 :" << R2 << std::endl;
    std::cout << "P1 :" << P1 << std::endl;
    std::cout << "P2 :" << P2 << std::endl;
    std::cout << "Q  :" << Q  << std::endl;
    calculateRectifiedFOV(P1, rectified_image_size);

    // 5. 依据针孔模型，精确计算并强制覆盖焦距 f_rect，从而绝对控制输出 FOV
    double f_rect = S / (2.0 * std::tan(theta));
    double f_old = P1.at<double>(0, 0); // 备份旧焦距用于等比例缩放基线

    // 重新将主点精准定位到新方形图像的正中心
    double cx_new = S / 2.0;
    double cy_new = S / 2.0;

    // 强制覆写 P1 的内参
    P1.at<double>(0, 0) = P1.at<double>(1, 1) = f_rect;
    P1.at<double>(0, 2) = cx_new;
    P1.at<double>(1, 2) = cy_new;

    // 强制覆写 P2 的内参，保持 fx, fy, cx, cy 与 P1 绝对一致以保证极线水平对齐
    P2.at<double>(0, 0) = P2.at<double>(1, 1) = f_rect;
    P2.at<double>(0, 2) = cx_new;
    P2.at<double>(1, 2) = cy_new;
    // 基线平移项 (Tx * f) 必须根据新旧焦距比值进行等比例缩放
    P2.at<double>(0, 3) = P2.at<double>(0, 3) * (f_rect / f_old);

    // 6. 必须同步重构 Q 矩阵，否则后文 reprojectImageTo3D 还原出的三维点云会产生严重的拉伸和畸变
    Q.at<double>(0, 0) = 1.0;  Q.at<double>(0, 1) = 0.0;  Q.at<double>(0, 2) = 0.0;  Q.at<double>(0, 3) = -cx_new;
    Q.at<double>(1, 0) = 0.0;  Q.at<double>(1, 1) = 1.0;  Q.at<double>(1, 2) = 0.0;  Q.at<double>(1, 3) = -cy_new;
    Q.at<double>(2, 0) = 0.0;  Q.at<double>(2, 1) = 0.0;  Q.at<double>(2, 2) = 0.0;  Q.at<double>(2, 3) = f_rect;
    // Q(3,2) 是 -1/B（B为双目物理基线），不受焦距改变影响，维持 stereoRectify 的默认输出即可
    // Q(3,3) 在 CALIB_ZERO_DISPARITY 且左右主点严格对齐时恒为 0.0
    Q.at<double>(3, 3) = 0.0;

    std::cout << "P1 (Modified):" << P1 << std::endl;
    std::cout << "P2 (Modified):" << P2 << std::endl;
    std::cout << "Q  (Modified):" << Q  << std::endl;
    calculateRectifiedFOV(P1, rectified_image_size);

    // ===================================================================================

    // 2. 生成映射表 (此时传入动态计算的 rectified_image_size)
    cv::Mat map1x, map1y, map2x, map2y;
    cv::fisheye::initUndistortRectifyMap(k[0], d[0], R1, P1, rectified_image_size, CV_32FC1, map1x, map1y);
    cv::fisheye::initUndistortRectifyMap(k[1], d[1], R2, P2, rectified_image_size, CV_32FC1, map2x, map2y);

    cv::Mat extrinsic_mat_left, extrinsic_mat_right;
    {
        FileStorage fs("/camera/config.xml", FileStorage::READ);
        if (!fs.isOpened()) {
            throw std::runtime_error("Open camera configs file failed!");
        }
        fs["cam_left"] >> extrinsic_mat_left;
        fs.release();
    }

    {
        cv::Mat f = cv::Mat::eye(4, 4, CV_64F);

        R.copyTo(f(cv::Rect(0, 0, 3, 3)));           // 左上角 3x3
        cv::Mat Tm = T / 1000.0;                     // 毫米转米
        Tm.copyTo(f(cv::Rect(3, 0, 1, 3)));          // 右上角 3x1 (第3列, 第0行开始)
        extrinsic_mat_right = f*extrinsic_mat_left;
        std::cout << "extrinsic_mat_right:" << extrinsic_mat_right << std::endl;
    }

    g_k1 = P1.colRange(0, 3).clone();
    g_k2 = P2.colRange(0, 3).clone();
    g_r1 = R1.clone();
    g_r2 = R2.clone();
    g_map_x1 = map1x.clone();
    g_map_y1 = map1y.clone();
    g_map_x2 = map2x.clone();
    g_map_y2 = map2y.clone();
    g_extrinsic_left = extrinsic_mat_left.clone();
    g_extrinsic_right = extrinsic_mat_right.clone();
    g_baseline = cv::norm(T);
    g_fisheye_k[0] = k[0];
    g_fisheye_k[1] = k[1];
    g_fisheye_d[0] = d[0];
    g_fisheye_d[1] = d[1];

    //球面极线校正
    generateFisheyeRotateMap(image_size, k[0], d[0], R1, g_fisheye_stereo_map_x1, g_fisheye_stereo_map_y1);
    generateFisheyeRotateMap(image_size, k[1], d[1], R2, g_fisheye_stereo_map_x2, g_fisheye_stereo_map_y2);

    //横向展开
    buildFisheyeUnwrapMap(k[0], d[0], panorama_size, CV_PI, CV_PI, g_panorama_map_x1, g_panorama_map_y1);
    buildFisheyeUnwrapMap(k[1], d[1], panorama_size, CV_PI, CV_PI, g_panorama_map_x2, g_panorama_map_y2);
}

int main(int argc, const char *argv[]) {
    cv::Mat cam_left, scannerMat;
    FileStorage fs("/camera/config.xml", FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Open config file failed!");
    }
//    fs["scannerMat"] >> scannerMat;
    fs["cam_left"] >> cam_left;
    fs.release();

//    Eigen::Matrix4d scannerMatrix, cam_left_matrix;
//    cv::cv2eigen(scannerMat, scannerMatrix);
//    cv::cv2eigen(cam_left, cam_left_matrix);
//    transform_to_camera_matrix = cam_left_matrix*scannerMatrix;
    cv::cv2eigen(cam_left, transform_to_camera_matrix);

//    transform_to_camera_matrix.setIdentity();
    std::cout << "transform_to_camera_matrix:" << std::endl << transform_to_camera_matrix << std::endl;

    // 初始化 ROS 2
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("livox_color_node");
    // --- 声明并获取参数 ---
    node->declare_parameter<bool>("is_color", true);
    node->declare_parameter<std::string>("pc_frame_id", "livox_frame");
    node->declare_parameter<std::string>("cam_frame_id", "camera");
    node->declare_parameter<int64_t>("frames_per_publish", 200);
    node->declare_parameter<std::string>("resolution", "3840x1080");
    node->declare_parameter<bool>("image_raw_compressed", false);
    node->declare_parameter<std::string>("qos", "SystemDefaults");
    node->declare_parameter<bool>("tilt", true);
    node->declare_parameter<bool>("undistort", false);
    node->declare_parameter("undistort_angle_h", 0.0);
    node->declare_parameter("undistort_angle_v", 0.0);
    node->declare_parameter<int>("undistort_w", 800);
    node->declare_parameter<int>("undistort_h", 600);
    node->declare_parameter("undistort_fxy", 414.18);
    node->declare_parameter("valid_incidence", 90.0);

    std::string resolution = node->get_parameter("resolution").as_string();
    if (!parseResolution(resolution, image_actual_width, image_actual_height)) {
        throw std::runtime_error("parsing image resolution fail!");
    }
    std::cout << "image_actual_width:" << image_actual_width << ", image_actual_height:" << image_actual_height << std::endl;

    g_tilt = node->get_parameter("tilt").as_bool();
    g_undistort = node->get_parameter("undistort").as_bool();

    rclcpp::Parameter param = node->get_parameter("undistort_angle_h");
    if (param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
        g_undistort_angle_h = static_cast<double>(param.as_int());
    } else if (param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
        g_undistort_angle_h = param.as_double();
    } else {
        std::cerr << "undistort_angle_h Type mismatch! Must be int or double." << std::endl;
        return -1;
    }
    g_undistort_angle_h = clamp_value(g_undistort_angle_h, -45.0, 45.0);

    param = node->get_parameter("undistort_angle_v");
    if (param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
        g_undistort_angle_v = static_cast<double>(param.as_int());
    } else if (param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
        g_undistort_angle_v = param.as_double();
    } else {
        std::cerr << "undistort_angle_v Type mismatch! Must be int or double." << std::endl;
        return -1;
    }
    g_undistort_angle_v = clamp_value(g_undistort_angle_v, -45.0, 45.0);

    update_undistort_r();

    param = node->get_parameter("undistort_w");
    if (param.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER) {
        std::cerr << "undistort_w Type mismatch! Must be int" << std::endl;
        return -1;
    }
    g_undistort_w = param.as_int();
    param = node->get_parameter("undistort_h");
    if (param.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER) {
        std::cerr << "undistort_h Type mismatch! Must be int" << std::endl;
        return -1;
    }
    g_undistort_h = param.as_int();
    g_undistort_w = static_cast<int>(clamp_value(g_undistort_w, 100.0, image_actual_width*0.75));
    g_undistort_h = static_cast<int>(clamp_value(g_undistort_h, 100.0, image_actual_height*0.75));

    param = node->get_parameter("undistort_fxy");
    if (param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
        g_undistort_fxy = static_cast<double>(param.as_int());
    } else if (param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
        g_undistort_fxy = param.as_double();
    } else {
        std::cerr << "undistort_fxy Type mismatch! Must be int or double." << std::endl;
        return -1;
    }
    int max_len = g_undistort_w > g_undistort_h ? g_undistort_w : g_undistort_h;
    g_undistort_fxy = clamp_value(g_undistort_fxy, max_len*0.25, max_len*0.75);

    update_undistort_k();

    param = node->get_parameter("valid_incidence");
    if (param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
        g_valid_incidence = static_cast<double>(param.as_int());
    } else if (param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
        g_valid_incidence = param.as_double();
    } else {
        std::cerr << "undistort_fxy Type mismatch! Must be int or double." << std::endl;
        return -1;
    }
    g_valid_incidence = clamp_value(g_valid_incidence, 50.0, 90.0);

    //注册动态参数回调函数
    auto parameter_callback_handle = node->add_on_set_parameters_callback(on_parameter_change);

    std::string camera_intrinsic_filepath_le, camera_intrinsic_filepath_re;
    camera_intrinsic_filepath_le = "/camera/camera_params_le.xml";
    camera_intrinsic_filepath_re = "/camera/camera_params_re.xml";
//    if ((image_actual_width == 1280 && image_actual_height == 720) ||
//        (image_actual_width == 1920 && image_actual_height == 1080) ||
//        (image_actual_width == 3840 && image_actual_height == 2160)) {
//        camera_intrinsic_filepath = "/camera/camera_params_1.7.xml";
//    } else if ((image_actual_width == 4000 && image_actual_height == 3000) ||
//               (image_actual_width == 8000 && image_actual_height == 6000) ||
//               (image_actual_width == 4608 && image_actual_height == 3456)) {
//        camera_intrinsic_filepath = "/camera/camera_params_1.3.xml";
//    } else {
//        throw std::runtime_error("Unsupported resolution!");
//    }
    std::cout << "camera_intrinsic_filepath_le:" << camera_intrinsic_filepath_le << std::endl;
    std::cout << "camera_intrinsic_filepath_re:" << camera_intrinsic_filepath_re << std::endl;

    int64_t frames_count = node->get_parameter("frames_per_publish").as_int();
    g_frames_per_publish = static_cast<size_t>(frames_count);
    frame_points.reserve(g_frames_per_publish * 96);

    setupOpenCL(image_actual_width * image_actual_height * 3, g_frames_per_publish * 96 * sizeof(ColoredPoint));
    std::cout << "frames_per_publish:" << g_frames_per_publish << std::endl;

    // 创建发布者
    rclcpp::QoS qos = rclcpp::SystemDefaultsQoS();
    std::string qos_param = node->get_parameter("qos").as_string();
    if (qos_param == "SystemDefaults") {
        qos = rclcpp::SystemDefaultsQoS();
    } else if (qos_param == "SensorData") {
        qos = rclcpp::SensorDataQoS();
    } else if (qos_param == "Services") {
        qos = rclcpp::ServicesQoS();
    } else {
        std::cout << "unknown QoS param, set to SystemDefaults." << std::endl;
        qos = rclcpp::SystemDefaultsQoS();
    }
    auto publisher_pc = node->create_publisher<sensor_msgs::msg::PointCloud2>("/livox/colored_point_cloud", qos);
    auto publisher_image = node->create_publisher<sensor_msgs::msg::Image>("camera/image_raw", qos);
    auto publisher_image_compressed = node->create_publisher<sensor_msgs::msg::CompressedImage>("camera/image_raw/compressed", qos);

    pCameraSingle = new CAMERA2EYE::CameraSingle("/dev/video0", image_actual_width, image_actual_height, 2560, 720, camera_intrinsic_filepath_le, camera_intrinsic_filepath_re);
//    {
//        CAMERA::CameraParams cameraParamsMain = pCameraSingle->getMainCameraParams();
//        if (image_actual_width != cameraParamsMain.width) {
//            if (std::abs(1.0*image_actual_width/image_actual_height - 1.0*cameraParamsMain.width/cameraParamsMain.height) > 0.001) {
//                throw std::runtime_error("Camera resolution is not proportional to the intrinsic!");
//            } else {
//                double scale = 1.0*image_actual_width/cameraParamsMain.width;
//                cameraParamsMain.k(0, 0) *= scale;//fx
//                cameraParamsMain.k(1, 1) *= scale;//fy
//                cameraParamsMain.k(0, 2) *= scale;//cx
//                cameraParamsMain.k(1, 2) *= scale;//cy
//            }
//        }
//        k_vec = { (float)cameraParamsMain.k(0,0), (float)cameraParamsMain.k(1,1),
//                  (float)cameraParamsMain.k(0,2), (float)cameraParamsMain.k(1,2) };
//        d_vec = { (float)cameraParamsMain.d(0), (float)cameraParamsMain.d(1),
//                  (float)cameraParamsMain.d(2), (float)cameraParamsMain.d(3) };
//        pFisheyeUndistorter = new FisheyeUndistorter(cameraParamsMain.k, cameraParamsMain.d);
//    }

    initCalculation();

    // --- 启动双线程 ---
    // 线程1：相机以 ~15Hz 疯狂向缓存写入 JPEG
    std::thread t_camera(CameraCaptureThread, publisher_image, publisher_image_compressed, node);
    // 线程2：点云处理线程消费队列，并发布 ROS2 消息
    std::thread t_worker(ColorizationWorkerThread, publisher_pc, node);

    std::thread t_motor(MotorControllerThread);

    std::string config_filepath = "/usr/local/mid360s_config.json";
    // REQUIRED, to init Livox SDK2
    if (!LivoxLidarSdkInit(config_filepath.c_str())) {
        printf("Livox Init Failed\n");
        rclcpp::shutdown();
        {
            std::lock_guard<std::mutex> lock(g_cloud_mutex);
            g_cloud_cv.notify_all();
        }
        {
            std::lock_guard<std::mutex> lock(g_motor_mutex);
            g_motor_cv.notify_all();
        }
        if (t_camera.joinable()) t_camera.join();
        if (t_worker.joinable()) t_worker.join();
        if (t_motor.joinable()) t_motor.join();
        LivoxLidarSdkUninit();
        cleanupOpenCL();
        if (pCameraSingle) {
            delete pCameraSingle;
            pCameraSingle = nullptr;
        }
        if (pFisheyeUndistorter) {
            delete pFisheyeUndistorter;
            pFisheyeUndistorter = nullptr;
        }
        return -1;
    }

    // REQUIRED, to get point cloud data via 'PointCloudCallback'
    SetLivoxLidarPointCloudCallBack(PointCloudCallback, nullptr);

    // OPTIONAL, to get imu data via 'ImuDataCallback'
    // some lidar types DO NOT contain an imu component
    SetLivoxLidarImuDataCallback(ImuDataCallback, nullptr);

    SetLivoxLidarInfoCallback(LivoxLidarPushMsgCallback, nullptr);

    // REQUIRED, to get a handle to targeted lidar and set its work mode to NORMAL
    SetLivoxLidarInfoChangeCallback(LidarInfoChangeCallback, nullptr);

    printf("初始化完毕.\n");

    // ROS 2 异步 Spin
    rclcpp::spin(node);
    rclcpp::shutdown();

    // 2. 关键：唤醒可能正在等待数据的线程
    {
        std::lock_guard<std::mutex> lock(g_cloud_mutex);
        g_cloud_cv.notify_all();
    }
    {
        std::lock_guard<std::mutex> lock(g_motor_mutex);
        g_motor_cv.notify_all();
    }
    common_utils::sleepMilliseconds(1000);
    if (t_worker.joinable()) t_worker.join();
    if (t_camera.joinable()) t_camera.join();
    if (t_motor.joinable()) t_motor.join();
    {
        livox_status status = SetLivoxLidarWorkMode(mLidarHandle, kLivoxLidarWakeUp, WorkModeCallback, nullptr);
        if (status == kLivoxLidarStatusSuccess) {
            printf("调用成功\n");
        }
        if (status != kLivoxLidarStatusSuccess) {
            throw std::runtime_error("SetLivoxLidarWorkMode failed! Status = " + std::to_string(status));
        }
        status = SetLivoxLidarWorkMode(mLidarHandle, kLivoxLidarSleep, WorkModeCallback, nullptr);
        if (status == kLivoxLidarStatusSuccess) {
            printf("调用成功\n");
        }
        if (status != kLivoxLidarStatusSuccess) {
            throw std::runtime_error("SetLivoxLidarWorkMode failed! Status = " + std::to_string(status));
        }
    }
    LivoxLidarSdkUninit();
    cleanupOpenCL();
    if (pCameraSingle) {
        delete pCameraSingle;
        pCameraSingle = nullptr;
    }
    if (pFisheyeUndistorter) {
        delete pFisheyeUndistorter;
        pFisheyeUndistorter = nullptr;
    }
    printf("livox_color end!\n");
    return 0;
}
