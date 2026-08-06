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
#include "camera.h"
#include "ply_fast_writer.h"
#include <CL/cl.h>

//#define RECORD_POINT_CLOUD
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
cv::Size g_undistort_img_size;
cv::Matx33d g_undistort_k = cv::Matx33d::eye();

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
std::condition_variable g_cloud_cv;
std::queue<std::vector<ColoredPoint>> g_cloud_queue;
std::queue<cv::Mat> g_image_queue;
// 防爆核心：最多只缓存 2 帧点云。来不及处理就丢弃旧的，保证 RViz 永远看最新帧。
const size_t MAX_QUEUE_SIZE = 1;

CAMERA::CameraSingle* pCameraSingle = nullptr;
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
    if (data == nullptr) return;
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

            cp.x = p_point_data[i].x / 1000.0f; // 毫米转米
            cp.y = p_point_data[i].y / 1000.0f;
            cp.z = p_point_data[i].z / 1000.0f;
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
    } else {
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
            if (pCameraSingle != nullptr) {
                pCameraSingle->cachedImage(frame_image);
            }
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

//获取IMU数据回调
void ImuDataCallback(uint32_t handle, const uint8_t dev_type,  LivoxLidarEthernetPacket* data, void* client_data) {
    if (data == nullptr) {
        return;
    }
    if (data->data_type == kLivoxLidarImuData) {
        auto pData = (LivoxLidarImuRawPoint *) data->data;

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
                pFisheyeUndistorter->undistort(mat, umat, tmp_undistort_img_size, tmp_undistort_k, g_undistort_r);
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
        printf("image published, freq: %f Hz\n", 1000.0f/(t_e-t_s));
        t_s = t_e;
    }
}

void color_with_cl(std::vector<ColoredPoint>& points, cv::Mat& bgr_image, const Eigen::Matrix4d& transform) {
    cl_int err;
    int num_pts = points.size();
    int img_w = bgr_image.cols;
    int img_h = bgr_image.rows;

    float m[12];
    for(int i=0; i<3; ++i)
        for(int j=0; j<4; ++j) m[i*4+j] = (float)transform(i,j);

    // 注意：确保 bgr_image.isContinuous() 为 true
    clEnqueueWriteBuffer(queue_cl, buf_pts_persistent, CL_FALSE, 0,
                         points.size() * sizeof(ColoredPoint), points.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue_cl, buf_bgr_persistent, CL_FALSE, 0,
                         bgr_image.total() * bgr_image.elemSize(), bgr_image.data, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue_cl, buf_mat_persistent, CL_FALSE, 0,
                         12*sizeof(float), m, 0, nullptr, nullptr);

    // 5. 设置参数
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_pts_persistent);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_bgr_persistent); // 图像 Buffer
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_mat_persistent);
    clSetKernelArg(kernel, 3, sizeof(cl_float4), &k_vec);
    clSetKernelArg(kernel, 4, sizeof(cl_float4), &d_vec);
    clSetKernelArg(kernel, 5, sizeof(int), &img_w);      // 图像宽
    clSetKernelArg(kernel, 6, sizeof(int), &img_h);      // 图像高
    clSetKernelArg(kernel, 7, sizeof(int), &num_pts);

    // 6. 执行
    size_t global_size = num_pts;

    clEnqueueNDRangeKernel(queue_cl, kernel, 1, NULL, &global_size, NULL, 0, nullptr, nullptr);

    // 6. 将结果读回 (由 GPU 传向 Host)
    // 这里使用阻塞读取 CL_TRUE，确保下一行访问 points 时数据已就绪
    clEnqueueReadBuffer(queue_cl, buf_pts_persistent, CL_TRUE, 0,
                        points.size() * sizeof(ColoredPoint), points.data(), 0, nullptr, nullptr);
}

// --- 线程 2：点云处理与赋色 (速率取决于此线程性能) ---
void ColorizationWorkerThread(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher, rclcpp::Node::SharedPtr node) {
    cv::Mat bgr_image;
    long timestamp;
#ifdef RECORD_POINT_CLOUD
    pfw::PlyFastWriter writer;
    writer.reserve(g_frames_per_publish*96*RECORD_POINT_CLOUD_FRAMES);
    size_t count = 0;
#endif
    bool is_color = node->get_parameter("is_color").as_bool();
    std::string frame_id = node->get_parameter("pc_frame_id").as_string();
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

        if (is_color) {
#if SOFT_COLOR
            // 3. 开始赋色
            std::vector<cv::Point3f> object_points;
            std::vector<Point2f> image_points;
            object_points.resize(points_to_process.size());

            // 提前提取矩阵元素，避免在循环中重复访问（如果是 Eigen 矩阵，访问可能涉及函数调用）
            const double m00 = transform_to_camera_matrix(0, 0), m01 = transform_to_camera_matrix(0, 1),
                    m02 = transform_to_camera_matrix(0, 2), m03 = transform_to_camera_matrix(0, 3);
            const double m10 = transform_to_camera_matrix(1, 0), m11 = transform_to_camera_matrix(1, 1),
                    m12 = transform_to_camera_matrix(1, 2), m13 = transform_to_camera_matrix(1, 3);
            const double m20 = transform_to_camera_matrix(2, 0), m21 = transform_to_camera_matrix(2, 1),
                    m22 = transform_to_camera_matrix(2, 2), m23 = transform_to_camera_matrix(2, 3);

#pragma omp parallel for
            for (size_t i = 0; i < points_to_process.size(); ++i) {
                const auto& pt = points_to_process[i];

                // 这里的变量名对应输入：x' = -pt.z, y' = pt.y, z' = pt.x
                double src_x = -static_cast<double>(pt.z);
                double src_y =  static_cast<double>(pt.y);
                double src_z =  static_cast<double>(pt.x);

                // 展开计算
                object_points[i].x = static_cast<float>(m00 * src_x + m01 * src_y + m02 * src_z + m03);
                object_points[i].y = static_cast<float>(m10 * src_x + m11 * src_y + m12 * src_z + m13);
                object_points[i].z = static_cast<float>(m20 * src_x + m21 * src_y + m22 * src_z + m23);
            }
            printf("transform cost: %f s\n", (common_utils::currentTimeMilliseconds()-t_c)/1000.0f);

            if (pCameraSingle->projectPoints(object_points, image_points)) {
//#pragma omp parallel for default(none) shared(points_to_process, object_points, image_points, bgr_image)
                for (size_t i=0; i<points_to_process.size(); i++) {
                    auto& pi = image_points[i];
                    if (object_points[i].z > 0 && pi.y >= 0 && pi.y < bgr_image.rows && pi.x >=0 && pi.x < bgr_image.cols) {
                        auto pix = bgr_image.at<Vec3b>(pi.y, pi.x).val;
                        auto& pt = points_to_process[i];
                        pt.b = pix[0];
                        pt.g = pix[1];
                        pt.r = pix[2];
                    }
                }
            } else {
                printf("projectPoints fail\n");
            }
            printf("color cost: %f s\n", (common_utils::currentTimeMilliseconds()-t_c)/1000.0f);
#else
            color_with_cl(points_to_process, bgr_image, transform_to_camera_matrix);
            printf("color cost: %f s\n", (common_utils::currentTimeMilliseconds()-t_c)/1000.0f);
#endif
        }


        // 4. 将处理后的点云转换为 ROS 2 消息
        sensor_msgs::msg::PointCloud2 msg;
        msg.header.stamp = rclcpp::Clock().now();
        msg.header.frame_id = frame_id;
        msg.height = 1;
        msg.width = points_to_process.size();

        // 配置 RViz 支持的字段：xyz, intensity, rgb
        sensor_msgs::PointCloud2Modifier modifier(msg);

        bool tile_enable = g_tilt;

        if (is_color) {
            modifier.setPointCloud2Fields(8,
                                          "x", 1, sensor_msgs::msg::PointField::FLOAT32,
                                          "y", 1, sensor_msgs::msg::PointField::FLOAT32,
                                          "z", 1, sensor_msgs::msg::PointField::FLOAT32,
                                          "intensity", 1, sensor_msgs::msg::PointField::FLOAT32,
                                          "rgb", 1, sensor_msgs::msg::PointField::FLOAT32, // rviz通常使用FLOAT32装载RGB
                                          "tag", 1, sensor_msgs::msg::PointField::UINT8,
                                          "line", 1, sensor_msgs::msg::PointField::UINT8,
                                          "timestamp", 1, sensor_msgs::msg::PointField::FLOAT64
            );
            modifier.resize(points_to_process.size());

            sensor_msgs::PointCloud2Iterator<float> iter_x(msg, "x");
            sensor_msgs::PointCloud2Iterator<float> iter_y(msg, "y");
            sensor_msgs::PointCloud2Iterator<float> iter_z(msg, "z");
            sensor_msgs::PointCloud2Iterator<float> iter_intensity(msg, "intensity");
            sensor_msgs::PointCloud2Iterator<uint8_t> iter_rgb(msg, "rgb");
            sensor_msgs::PointCloud2Iterator<uint8_t> iter_tag(msg, "tag");
            sensor_msgs::PointCloud2Iterator<uint8_t> iter_line(msg, "line");
            sensor_msgs::PointCloud2Iterator<double> iter_timestamp(msg, "timestamp");

            if (tile_enable) {
                for (const auto& pt : points_to_process) {
                    *iter_x = Rtilt(0,0)*pt.x + Rtilt(0,1)*pt.y + Rtilt(0,2)*pt.z;//pt.x;
                    *iter_y = Rtilt(1,0)*pt.x + Rtilt(1,1)*pt.y + Rtilt(1,2)*pt.z;//pt.y;
                    *iter_z = Rtilt(2,0)*pt.x + Rtilt(2,1)*pt.y + Rtilt(2,2)*pt.z;//pt.z;
                    *iter_intensity = pt.reflectivity;
                    *iter_tag = pt.tag;
                    *iter_line = pt.line;
                    *iter_timestamp = static_cast<double>(pt.timestamp);

                    // ROS 中 rgb 打包为 uint32
                    uint32_t rgb = (static_cast<uint32_t>(pt.r) << 16 |
                                    static_cast<uint32_t>(pt.g) << 8 |
                                    static_cast<uint32_t>(pt.b));
                    memcpy(&iter_rgb[0], &rgb, sizeof(uint32_t));

                    ++iter_x; ++iter_y; ++iter_z; ++iter_intensity; ++iter_rgb; ++iter_tag; ++iter_line; ++iter_timestamp;
                }
            } else {
                for (const auto& pt : points_to_process) {
                    *iter_x = pt.x;
                    *iter_y = pt.y;
                    *iter_z = pt.z;
                    *iter_intensity = pt.reflectivity;
                    *iter_tag = pt.tag;
                    *iter_line = pt.line;
                    *iter_timestamp = static_cast<double>(pt.timestamp);

                    // ROS 中 rgb 打包为 uint32
                    uint32_t rgb = (static_cast<uint32_t>(pt.r) << 16 |
                                    static_cast<uint32_t>(pt.g) << 8 |
                                    static_cast<uint32_t>(pt.b));
                    memcpy(&iter_rgb[0], &rgb, sizeof(uint32_t));

                    ++iter_x; ++iter_y; ++iter_z; ++iter_intensity; ++iter_rgb; ++iter_tag; ++iter_line; ++iter_timestamp;
                }
            }
        } else {
            modifier.setPointCloud2Fields(7,
                                          "x", 1, sensor_msgs::msg::PointField::FLOAT32,
                                          "y", 1, sensor_msgs::msg::PointField::FLOAT32,
                                          "z", 1, sensor_msgs::msg::PointField::FLOAT32,
                                          "intensity", 1, sensor_msgs::msg::PointField::FLOAT32,
                                          "tag", 1, sensor_msgs::msg::PointField::UINT8,
                                          "line", 1, sensor_msgs::msg::PointField::UINT8,
                                          "timestamp", 1, sensor_msgs::msg::PointField::FLOAT64

            );
            modifier.resize(points_to_process.size());

            sensor_msgs::PointCloud2Iterator<float> iter_x(msg, "x");
            sensor_msgs::PointCloud2Iterator<float> iter_y(msg, "y");
            sensor_msgs::PointCloud2Iterator<float> iter_z(msg, "z");
            sensor_msgs::PointCloud2Iterator<float> iter_intensity(msg, "intensity");
            sensor_msgs::PointCloud2Iterator<uint8_t> iter_tag(msg, "tag");
            sensor_msgs::PointCloud2Iterator<uint8_t> iter_line(msg, "line");
            sensor_msgs::PointCloud2Iterator<double> iter_timestamp(msg, "timestamp");

            if (tile_enable) {
                for (const auto& pt : points_to_process) {
                    *iter_x = Rtilt(0,0)*pt.x + Rtilt(0,1)*pt.y + Rtilt(0,2)*pt.z;//pt.x;
                    *iter_y = Rtilt(1,0)*pt.x + Rtilt(1,1)*pt.y + Rtilt(1,2)*pt.z;//pt.y;
                    *iter_z = Rtilt(2,0)*pt.x + Rtilt(2,1)*pt.y + Rtilt(2,2)*pt.z;//pt.z;
                    *iter_intensity = pt.reflectivity;
                    *iter_tag = pt.tag;
                    *iter_line = pt.line;
                    *iter_timestamp = static_cast<double>(pt.timestamp);

                    ++iter_x; ++iter_y; ++iter_z; ++iter_intensity; ++iter_tag; ++iter_line; ++iter_timestamp;
                }
            } else {
                for (const auto& pt : points_to_process) {
                    *iter_x = pt.x;
                    *iter_y = pt.y;
                    *iter_z = pt.z;
                    *iter_intensity = pt.reflectivity;
                    *iter_tag = pt.tag;
                    *iter_line = pt.line;
                    *iter_timestamp = static_cast<double>(pt.timestamp);

                    ++iter_x; ++iter_y; ++iter_z; ++iter_intensity; ++iter_tag; ++iter_line; ++iter_timestamp;
                }
            }


        }
#ifdef RECORD_POINT_CLOUD
        if (count < RECORD_POINT_CLOUD_FRAMES) {
            for (const auto& pt : points_to_process) {
                writer.addPoint({(float)pt.x, (float)pt.y, (float)pt.z, (uint8_t)pt.reflectivity, pt.r, pt.g, pt.b});
            }
            count++;
        } else if (count == RECORD_POINT_CLOUD_FRAMES) {
            writer.write("/factory_tools/tmp/test.ply");
            count++;
        }
#endif
        // 5. 发布给 RViz
        publisher->publish(msg);
        printf("pc published, processing cost: %f s\n", (common_utils::currentTimeMilliseconds()-t_c)/1000.0f);
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

int main(int argc, const char *argv[]) {
    cv::Mat scannerMat, cam_left;
    FileStorage fs("/camera/config.xml", FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Open config file failed!");
    }
    fs["scannerMat"] >> scannerMat;
    fs["cam_left"] >> cam_left;
    fs.release();

    Eigen::Matrix4d scannerMatrix, cam_left_matrix;
    cv::cv2eigen(scannerMat, scannerMatrix);
    cv::cv2eigen(cam_left, cam_left_matrix);
    transform_to_camera_matrix = cam_left_matrix*scannerMatrix;

    std::cout << "transform_to_camera_matrix:" << std::endl << transform_to_camera_matrix << std::endl;

    // 初始化 ROS 2
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("livox_color_node");
    // --- 声明并获取参数 ---
    node->declare_parameter<bool>("is_color", true);
    node->declare_parameter<std::string>("pc_frame_id", "livox_frame");
    node->declare_parameter<std::string>("cam_frame_id", "camera");
    node->declare_parameter<int64_t>("frames_per_publish", 200);
    node->declare_parameter<std::string>("resolution", "1280x720");
    node->declare_parameter<bool>("image_raw_compressed", false);
    node->declare_parameter<std::string>("qos", "SystemDefaults");
    node->declare_parameter<bool>("tilt", true);
    node->declare_parameter<bool>("undistort", false);
    node->declare_parameter("undistort_angle_h", 0.0);
    node->declare_parameter("undistort_angle_v", 0.0);
    node->declare_parameter<int>("undistort_w", 800);
    node->declare_parameter<int>("undistort_h", 600);
    node->declare_parameter("undistort_fxy", 414.18);

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

    //注册动态参数回调函数
    auto parameter_callback_handle = node->add_on_set_parameters_callback(on_parameter_change);

    std::string camera_intrinsic_filepath;
    if ((image_actual_width == 1280 && image_actual_height == 720) ||
        (image_actual_width == 1920 && image_actual_height == 1080) ||
        (image_actual_width == 3840 && image_actual_height == 2160)) {
        camera_intrinsic_filepath = "/camera/camera_params_1.7.xml";
    } else if ((image_actual_width == 4000 && image_actual_height == 3000) ||
               (image_actual_width == 8000 && image_actual_height == 6000) ||
               (image_actual_width == 4608 && image_actual_height == 3456)) {
        camera_intrinsic_filepath = "/camera/camera_params_1.3.xml";
    } else {
        throw std::runtime_error("Unsupported resolution!");
    }
    std::cout << "camera_intrinsic_filepath:" << camera_intrinsic_filepath << std::endl;

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

    pCameraSingle = new CAMERA::CameraSingle("/dev/video0", camera_intrinsic_filepath, image_actual_width, image_actual_height);
    {
        CAMERA::CameraParams cameraParamsMain = pCameraSingle->getMainCameraParams();
        if (image_actual_width != cameraParamsMain.width) {
            if (std::abs(1.0*image_actual_width/image_actual_height - 1.0*cameraParamsMain.width/cameraParamsMain.height) > 0.001) {
                throw std::runtime_error("Camera resolution is not proportional to the intrinsic!");
            } else {
                double scale = 1.0*image_actual_width/cameraParamsMain.width;
                cameraParamsMain.k(0, 0) *= scale;//fx
                cameraParamsMain.k(1, 1) *= scale;//fy
                cameraParamsMain.k(0, 2) *= scale;//cx
                cameraParamsMain.k(1, 2) *= scale;//cy
            }
        }
        k_vec = { (float)cameraParamsMain.k(0,0), (float)cameraParamsMain.k(1,1),
                  (float)cameraParamsMain.k(0,2), (float)cameraParamsMain.k(1,2) };
        d_vec = { (float)cameraParamsMain.d(0), (float)cameraParamsMain.d(1),
                  (float)cameraParamsMain.d(2), (float)cameraParamsMain.d(3) };
        pFisheyeUndistorter = new FisheyeUndistorter(cameraParamsMain.k, cameraParamsMain.d);
    }

    // --- 启动双线程 ---
    // 线程1：相机以 ~15Hz 疯狂向缓存写入 JPEG
    std::thread t_camera(CameraCaptureThread, publisher_image, publisher_image_compressed, node);
    // 线程2：点云处理线程消费队列，并发布 ROS2 消息
    std::thread t_worker(ColorizationWorkerThread, publisher_pc, node);

    std::string config_filepath = "/usr/local/mid360_config.json";
    // REQUIRED, to init Livox SDK2
    if (!LivoxLidarSdkInit(config_filepath.c_str())) {
        printf("Livox Init Failed\n");
        rclcpp::shutdown();
        {
            std::lock_guard<std::mutex> lock(g_cloud_mutex);
            g_cloud_cv.notify_all();
        }
        if (t_camera.joinable()) t_camera.join();
        if (t_worker.joinable()) t_worker.join();
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

    // ROS 2 异步 Spin
    rclcpp::spin(node);
    rclcpp::shutdown();

    // 2. 关键：唤醒可能正在等待数据的线程
    {
        std::lock_guard<std::mutex> lock(g_cloud_mutex);
        g_cloud_cv.notify_all();
    }
    common_utils::sleepMilliseconds(1000);
    if (t_worker.joinable()) t_worker.join();
    if (t_camera.joinable()) t_camera.join();
    {
        livox_status status = SetLivoxLidarWorkMode(mLidarHandle, kLivoxLidarWakeUp, WorkModeCallback, nullptr);
        if (status != kLivoxLidarStatusSuccess) {
            throw std::runtime_error("SetLivoxLidarWorkMode failed! Status = " + std::to_string(status));
        }
        status = SetLivoxLidarWorkMode(mLidarHandle, kLivoxLidarSleep, WorkModeCallback, nullptr);
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
