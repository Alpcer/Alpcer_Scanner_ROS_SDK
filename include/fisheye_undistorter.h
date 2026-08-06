//
// Created by Zachary on 2026/5/20.
// Modified with Custom OpenCL Remap Kernel to bypass OpenCV 4.2.0 Bug
//

#ifndef LIVOX_COLOR_FISHEYE_UNDISTORTER_H
#define LIVOX_COLOR_FISHEYE_UNDISTORTER_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include "common_utils.h"

class FisheyeUndistorter {
private:
    cv::Matx33d fisheye_k;
    cv::Vec4d fisheye_d;

    cv::UMat mapX_ocl;
    cv::UMat mapY_ocl;

    cv::Mat mapX_cpu;
    cv::Mat mapY_cpu;

    // 缓存上一帧的参数，用于对比
    cv::Size cached_img_size;
    cv::Matx33d cached_k;
    cv::Mat cached_r;
    bool is_initialized = false;

    // 方案三新增：OpenCL 内核对象
    cv::ocl::Kernel ocl_kernel;

    // 自定义 OpenCL 双线性插值 Remap 内核字串（完美避开 -D depth=0 冲突）
    // 注：此内核针对标准的 8位3通道（BGR/RGB）图像进行了深度优化
    const char* custom_remap_kernel = R"(
    __kernel void MyFisheyeRemap(
        __global const uchar* src, int src_step, int src_offset, int src_cols, int src_rows,
        __global uchar* dst, int dst_step, int dst_offset,
        __global const uchar* map_x, int mapx_step, int mapx_offset,
        __global const uchar* map_y, int mapy_step, int mapy_offset)
    {
        int x = get_global_id(0);
        int y = get_global_id(1);

        if (x >= src_cols || y >= src_rows) return;

        // 1. 读取坐标时，必须加上 mapx_offset 和 mapy_offset
        float fx = *(__global const float*)(map_x + mapx_offset + y * mapx_step + x * 4);
        float fy = *(__global const float*)(map_y + mapy_offset + y * mapy_step + x * 4);

        int2 map_dataA = convert_int2_sat_rtn((float2)(fx, fy));
        int2 map_dataB = (int2)(map_dataA.x + 1, map_dataA.y);
        int2 map_dataC = (int2)(map_dataA.x, map_dataA.y + 1);
        int2 map_dataD = (int2)(map_dataA.x + 1, map_dataA.y + 1);

        float2 _u = (float2)(fx, fy) - convert_float2(map_dataA);
        float2 u = convert_float2(convert_int2_rte(_u * 32.0f)) / 32.0f;

        float3 a = (float3)(0.0f);
        float3 b = (float3)(0.0f);
        float3 c = (float3)(0.0f);
        float3 d = (float3)(0.0f);

        // 2. 读取源图像像素时，必须加上 src_offset
        if (map_dataA.x >= 0 && map_dataA.x < src_cols && map_dataA.y >= 0 && map_dataA.y < src_rows) {
            int idx = src_offset + map_dataA.y * src_step + map_dataA.x * 3;
            a = (float3)(src[idx], src[idx+1], src[idx+2]);
        }
        if (map_dataB.x >= 0 && map_dataB.x < src_cols && map_dataB.y >= 0 && map_dataB.y < src_rows) {
            int idx = src_offset + map_dataB.y * src_step + map_dataB.x * 3;
            b = (float3)(src[idx], src[idx+1], src[idx+2]);
        }
        if (map_dataC.x >= 0 && map_dataC.x < src_cols && map_dataC.y >= 0 && map_dataC.y < src_rows) {
            int idx = src_offset + map_dataC.y * src_step + map_dataC.x * 3;
            c = (float3)(src[idx], src[idx+1], src[idx+2]);
        }
        if (map_dataD.x >= 0 && map_dataD.x < src_cols && map_dataD.y >= 0 && map_dataD.y < src_rows) {
            int idx = src_offset + map_dataD.y * src_step + map_dataD.x * 3;
            d = (float3)(src[idx], src[idx+1], src[idx+2]);
        }

        float3 dst_data = a * (1.0f - u.x) * (1.0f - u.y) +
                          b * (u.x)        * (1.0f - u.y) +
                          c * (1.0f - u.x) * (u.y) +
                          d * (u.x)        * (u.y);

        uchar3 final_val = convert_uchar3_sat_rte(dst_data);

        // 3. 写入目标图像时，必须加上 dst_offset
        int dst_offset_idx = dst_offset + y * dst_step + x * 3;
        dst[dst_offset_idx]     = final_val.x;
        dst[dst_offset_idx + 1] = final_val.y;
        dst[dst_offset_idx + 2] = final_val.z;
    }
    )";

public:
    FisheyeUndistorter(cv::Matx33d& fisheye_k, cv::Vec4d& fisheye_d) : fisheye_k(fisheye_k), fisheye_d(fisheye_d) {
        cv::ocl::setUseOpenCL(true);
    }

    void undistort(const cv::Mat& distorted_img_cpu,
                   cv::UMat& undistorted_img_ocl,
                   cv::Size& img_size,
                   const cv::Matx33d& k,
                   const cv::Mat& r)
    {
        cv::UMat distorted_img_ocl = distorted_img_cpu.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_SHARED_MEMORY);

        // 核心逻辑：检查 k 或 r 是否发生改变（或者第一次运行）
        bool need_update = !is_initialized ||
                           (img_size != cached_img_size) ||
                           (k != cached_k) ||
                           cached_r.empty() ||
                           (cv::norm(r, cached_r, cv::NORM_INF) > 1e-6); // 检查 r 矩阵元素是否有微小变动

        if (need_update) {
            // 参数改变，在 CPU 中重新计算映射表
            cv::fisheye::initUndistortRectifyMap(
                    fisheye_k, fisheye_d, r, k,
                    img_size, CV_32FC1, mapX_cpu, mapY_cpu
            );

            // 重新上传到 GPU (OpenCL 内存)
            mapX_cpu.copyTo(mapX_ocl);
            mapY_cpu.copyTo(mapY_ocl);

            // 更新缓存记录
            cached_img_size = img_size;
            cached_k = k;
            r.copyTo(cached_r);
            is_initialized = true;
            std::cout << "Fisheye maps updated due to parameter change." << std::endl;
        }

        // --- 方案三核心改进点开始 ---

        // 1. 延迟初始化并编译自定义 OpenCL 内核（仅在第一帧或重新编译时执行一次）
        // 1. 延迟初始化并编译自定义 OpenCL 内核
        if (ocl_kernel.empty()) {
            // 使用 4 参数版本：模块名、内核名、源码、哈希
            // 这里我们给一个简单的名称，并留空哈希值（OpenCV 会自动计算）
            cv::ocl::ProgramSource programSource("my_fisheye_module", "MyFisheyeRemap", custom_remap_kernel, "");

            cv::String errmsg;
            cv::ocl::Program program = cv::ocl::Context::getDefault().getProg(programSource, "", errmsg);
            ocl_kernel.create("MyFisheyeRemap", program);

            if (ocl_kernel.empty()) {
                std::cerr << "错误：自定义 OpenCL 内核编译失败! 错误日志:\n" << errmsg << std::endl;
                return;
            }
        }

        // 2. 手动确保输出的 UMat 分配了正确的空间（因为绕过了自动分配空间的 cv::remap）
        if (undistorted_img_ocl.empty() ||
            undistorted_img_ocl.size() != img_size ||
            undistorted_img_ocl.type() != distorted_img_ocl.type())
        {
            undistorted_img_ocl.create(img_size, distorted_img_ocl.type());
        }

        // 3. 严格绑定参数，必须加上各个 UMat 的 offset！
        ocl_kernel.args(
                // Source 图像参数
                cv::ocl::KernelArg::PtrReadOnly(distorted_img_ocl),
                (int)distorted_img_ocl.step, (int)distorted_img_ocl.offset,
                (int)distorted_img_ocl.cols, (int)distorted_img_ocl.rows,

                // Destination 图像参数
                cv::ocl::KernelArg::PtrWriteOnly(undistorted_img_ocl),
                (int)undistorted_img_ocl.step, (int)undistorted_img_ocl.offset,

                // Map X 参数
                cv::ocl::KernelArg::PtrReadOnly(mapX_ocl),
                (int)mapX_ocl.step, (int)mapX_ocl.offset,

                // Map Y 参数
                cv::ocl::KernelArg::PtrReadOnly(mapY_ocl),
                (int)mapY_ocl.step, (int)mapY_ocl.offset
        );

        // 4. 配置二维并行网格大小（每个像素分配一个 GPU 线程处理）
        size_t globalThreads[3] = {(size_t)img_size.width, (size_t)img_size.height, 1};

        // 5. 异步推入 GPU 硬件执行队列
        ocl_kernel.run(2, globalThreads, NULL, false);
        // --- 方案三核心改进点结束 ---
    }
    void undistort(const cv::Mat& distorted_img,
                   cv::Mat& undistorted_img,
                   cv::Size& img_size,
                   const cv::Matx33d& k,
                   const cv::Mat& r)
    {
        // 核心逻辑：检查 k 或 r 是否发生改变（或者第一次运行）
        bool need_update = !is_initialized ||
                (img_size != cached_img_size) ||
                           (k != cached_k) ||
                           cached_r.empty() ||
                           (cv::norm(r, cached_r, cv::NORM_INF) > 1e-6); // 检查 r 矩阵元素是否有微小变动

        if (need_update) {
            // 参数改变，在 CPU 中重新计算映射表
            cv::fisheye::initUndistortRectifyMap(
                    fisheye_k, fisheye_d, r, k,
                    img_size, CV_32FC1, mapX_cpu, mapY_cpu
            );

            // 更新缓存记录
            cached_img_size = img_size;
            cached_k = k;
            r.copyTo(cached_r);
            is_initialized = true;
            std::cout << "Fisheye maps updated due to parameter change." << std::endl;
        }

        // 核心加速点：直接复用 GPU 内部更新好的 map
        cv::remap(distorted_img, undistorted_img, mapX_cpu, mapY_cpu, cv::INTER_LINEAR);
    }

    void undistort_umat(const cv::UMat& distorted_img,
                   cv::UMat& undistorted_img,
                   cv::Size& img_size,
                   const cv::Matx33d& k,
                   const cv::Mat& r)
    {
        // 核心逻辑：检查 k 或 r 是否发生改变（或者第一次运行）
        bool need_update = !is_initialized ||
                           (img_size != cached_img_size) ||
                           (k != cached_k) ||
                           cached_r.empty() ||
                           (cv::norm(r, cached_r, cv::NORM_INF) > 1e-6); // 检查 r 矩阵元素是否有微小变动

        if (need_update) {
            // 参数改变，在 CPU 中重新计算映射表
            cv::fisheye::initUndistortRectifyMap(
                    fisheye_k, fisheye_d, r, k,
                    img_size, CV_32FC1, mapX_cpu, mapY_cpu
            );

            // 更新缓存记录
            cached_img_size = img_size;
            cached_k = k;
            r.copyTo(cached_r);
            is_initialized = true;
            std::cout << "Fisheye maps updated due to parameter change." << std::endl;
        }

        // 核心加速点：直接复用 GPU 内部更新好的 map
        cv::remap(distorted_img, undistorted_img, mapX_cpu, mapY_cpu, cv::INTER_LINEAR);
    }
};

#endif //LIVOX_COLOR_FISHEYE_UNDISTORTER_H