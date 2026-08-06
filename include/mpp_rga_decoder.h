//
// Created by Zachary on 2026/4/22.
// 完美复刻 mpi_dec_test.c 中的 dec_advanced 逻辑
// 全局单例 Frame，零分配开销，零报错
// 双缓冲 (Ping-Pong Buffer) 升级版，解决消费端耗时导致的花屏问题
//

#ifndef LIVOX_COLOR_MPP_RGA_DECODER_H
#define LIVOX_COLOR_MPP_RGA_DECODER_H

#include <rockchip/rk_mpi.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/mpp_buffer.h>
#include <rockchip/mpp_meta.h>
#include <rga/im2d.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>

#define MPP_ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))

class MppRgaDecoder {
public:
    MppRgaDecoder(int target_width, int target_height)
            : width_(target_width), height_(target_height) {

        MPP_RET ret = mpp_create(&mpp_ctx_, &mpp_api_);
        if (ret != MPP_OK) {
            std::cerr << "Failed to create MPP context: " << ret << std::endl;
            return;
        }

        MppDecCfg cfg = nullptr;
        mpp_dec_cfg_init(&cfg);
        ret = mpp_api_->control(mpp_ctx_, MPP_DEC_GET_CFG, cfg);
        if (ret) {
            std::cerr << "failed to get decoder cfg ret:" << ret << std::endl;
        }
        ret = mpp_dec_cfg_set_u32(cfg, "base:split_parse", 1);
        if (ret) {
            std::cerr << "failed to set split_parse ret:" << ret << std::endl;
        }
        ret = mpp_api_->control(mpp_ctx_, MPP_DEC_SET_CFG, cfg);
        if (ret) {
            std::cerr << "failed to set cfg ret:" << ret << std::endl;
        }
        mpp_dec_cfg_deinit(cfg);

        ret = mpp_init(mpp_ctx_, MPP_CTX_DEC, MPP_VIDEO_CodingMJPEG);
        if (ret != MPP_OK) {
            std::cerr << "Failed to init MPP for MJPEG: " << ret << std::endl;
            return;
        }

        mpp_buffer_group_get_internal(&mem_group_,static_cast<MppBufferType>(MPP_BUFFER_TYPE_DRM |
                                                                             MPP_BUFFER_FLAGS_DMA32 |
                                                                             MPP_BUFFER_FLAGS_CACHABLE));

        // 初始化内存与全局唯一的 output_frame_
        allocate_buffers();
        initialized_ = true;
        std::cout << "MPP MJPEG Decoder initialized. Official advanced mode ready (Double Buffering)." << std::endl;
    }

    ~MppRgaDecoder() {
        std::cout << "MppRgaDecoder destroying..." << std::endl;

        // 1. 如果 ctx 存在，先处理它
        if (mpp_ctx_ && mpp_api_) {
            // 让底层停止解码并清空队列
            MPP_RET ret = mpp_api_->reset(mpp_ctx_);
            if (ret) {
                std::cerr << "mpi->reset failed ret:" << ret << std::endl;
            }

            // 取出并清理底层队列中残留的 Frame
            MppFrame leftover_frame = nullptr;
            while (mpp_api_->decode_get_frame(mpp_ctx_, &leftover_frame) == MPP_OK && leftover_frame) {
                if (leftover_frame != output_frame_) {
                    mpp_frame_deinit(&leftover_frame);
                }
                leftover_frame = nullptr;
            }

            // 2. Context 彻底销毁后，再安全地释放外围分配的全局资产
            if (output_frame_) {
                mpp_frame_deinit(&output_frame_);
                output_frame_ = nullptr;
            }

            // 【最关键修改】：必须在销毁 buffer_group 之前先销毁 mpp_ctx_ ！
            mpp_destroy(mpp_ctx_);
            mpp_ctx_ = nullptr;
            mpp_api_ = nullptr;
        }

        if (yuv_buf_) {
            mpp_buffer_put(yuv_buf_);
            yuv_buf_ = nullptr;
        }

        // 循环释放双缓冲 BGR
        for (int i = 0; i < 2; ++i) {
            if (bgr_buf_[i]) {
                mpp_buffer_put(bgr_buf_[i]);
                bgr_buf_[i] = nullptr;
            }
        }

        if (mem_group_) {
            mpp_buffer_group_put(mem_group_);
            mem_group_ = nullptr;
        }

        std::cout << "MppRgaDecoder destroyed safely." << std::endl;
    }

    bool decode(const uint8_t* jpeg_data, size_t size, cv::Mat& out_mat) {
        if (!initialized_ || !jpeg_data || size == 0) return false;

        MppPacket packet = nullptr;
        MppBuffer pkt_buf = nullptr;
        MPP_RET ret;

        // --- 1. 拷贝 JPEG 数据到 DMA 缓冲区 ---
        ret = mpp_buffer_get(mem_group_, &pkt_buf, size);
        if (ret != MPP_OK || !pkt_buf) {
            std::cerr << "Failed to get buffer for packet: " << ret << std::endl;
            return false;
        }
        void* buf_ptr = mpp_buffer_get_ptr(pkt_buf);
        if (!buf_ptr) {
            mpp_buffer_put(pkt_buf);
            return false;
        }
        memcpy(buf_ptr, jpeg_data, size);
        // --- 2. 用 DMA 缓冲区初始化 Packet ---
        ret = mpp_packet_init_with_buffer(&packet, pkt_buf);
        mpp_buffer_put(pkt_buf);  // 释放本地引用，packet 已持有
        if (ret != MPP_OK) {
            std::cerr << "Failed to init packet with buffer: " << ret << std::endl;
            return false;
        }

        // --- 3. 注入全局唯一的 output_frame_ ---
        MppMeta meta = mpp_packet_get_meta(packet);
        if (meta) {
            // 对齐官方：反复注入同一个 frame，绝不设置宽高，留给底层填空
            mpp_meta_set_frame(meta, KEY_OUTPUT_FRAME, output_frame_);
        }

        // --- 4. 发送到解码器 ---
        ret = mpp_api_->decode_put_packet(mpp_ctx_, packet);

        // 发送完立即销毁 packet，但绝对不要销毁 frame！
        mpp_packet_deinit(&packet);

        if (ret != MPP_OK) {
//            std::cerr << "decode_put_packet failed: " << ret << std::endl;
            return false;
        }

        // --- 5. 获取解码结果 ---
        MppFrame result_frame = nullptr;
        ret = mpp_api_->decode_get_frame(mpp_ctx_, &result_frame);

        if (ret != MPP_OK || !result_frame) {
            std::cerr << "decode_get_frame failed: " << ret << std::endl;
            return false;
        }

        // 检查是否有硬件内部错误
        if (mpp_frame_get_errinfo(result_frame) != 0) {
            std::cerr << "MPP returned an error frame!" << std::endl;
            // 注意：哪怕报错，也不能 deinit 这个 result_frame，因为它指向我们的单例！
            // 如果返回的是非预期的独立帧，必须 deinit 释放！
            if (result_frame != output_frame_) {
                mpp_frame_deinit(&result_frame);
            }
            return false;
        }

        // --- 6. RGA 转换 ---
        // 此时 result_frame 里的 width/height 已经被 MPP 底层解析器动态填好了
        uint32_t actual_width  = mpp_frame_get_width(result_frame);
        uint32_t actual_height = mpp_frame_get_height(result_frame);
        MppFrameFormat fmt     = mpp_frame_get_fmt(result_frame);

        // 每次解码前，翻转索引 (0 -> 1 -> 0 -> 1...)
        current_bgr_idx_ = (current_bgr_idx_ + 1) % 2;

        bool success = process_with_rga(result_frame, actual_width, actual_height, fmt, current_bgr_idx_);

        // ！！！核心修改：严禁在此处调用 mpp_frame_deinit(&result_frame) ！！！
        // 因为 result_frame == output_frame_，是我们的全局资产。

        // 对于意外产生的独立帧，依然要清理
        if (result_frame != output_frame_) {
            mpp_frame_deinit(&result_frame);
        }

        if (success) {
            out_mat = cv::Mat(height_, width_, CV_8UC3, bgr_ptr_[current_bgr_idx_]);
        }
        return success;
    }

    // 同样为 commented-out 的函数加入 bgr 索引支持，确保你可以无缝反注释使用
//    bool decode(const uint8_t* jpeg_data, size_t size, cv::Rect crop_rect, cv::Mat& out_mat, bool scale_to_target = true) {
//        if (!initialized_ || !jpeg_data || size == 0) return false;
//
//        MppPacket packet = nullptr;
//        MppBuffer pkt_buf = nullptr;
//        MPP_RET ret;
//
//        // --- 1. 拷贝 JPEG 数据到 DMA 缓冲区 ---
//        ret = mpp_buffer_get(mem_group_, &pkt_buf, size);
//        if (ret != MPP_OK || !pkt_buf) {
//            std::cerr << "Failed to get buffer for packet: " << ret << std::endl;
//            return false;
//        }
//        void* buf_ptr = mpp_buffer_get_ptr(pkt_buf);
//        if (!buf_ptr) {
//            mpp_buffer_put(pkt_buf);
//            return false;
//        }
//        memcpy(buf_ptr, jpeg_data, size);
//
//        // --- 2. 用 DMA 缓冲区初始化 Packet ---
//        ret = mpp_packet_init_with_buffer(&packet, pkt_buf);
//        mpp_buffer_put(pkt_buf);  // 释放本地引用
//        if (ret != MPP_OK) {
//            std::cerr << "Failed to init packet with buffer: " << ret << std::endl;
//            return false;
//        }
//
//        // --- 3. 注入全局唯一的 output_frame_ ---
//        MppMeta meta = mpp_packet_get_meta(packet);
//        if (meta) {
//            mpp_meta_set_frame(meta, KEY_OUTPUT_FRAME, output_frame_);
//        }
//
//        // --- 4. 发送到解码器 ---
//        ret = mpp_api_->decode_put_packet(mpp_ctx_, packet);
//        mpp_packet_deinit(&packet);
//
//        if (ret != MPP_OK) {
//            std::cerr << "decode_put_packet failed: " << ret << std::endl;
//            return false;
//        }
//
//        // --- 5. 获取解码结果 ---
//        MppFrame result_frame = nullptr;
//        ret = mpp_api_->decode_get_frame(mpp_ctx_, &result_frame);
//
//        if (ret != MPP_OK || !result_frame) {
//            std::cerr << "decode_get_frame failed: " << ret << std::endl;
//            return false;
//        }
//
//        if (mpp_frame_get_errinfo(result_frame) != 0) {
//            std::cerr << "MPP returned an error frame!" << std::endl;
//            if (result_frame != output_frame_) {
//                mpp_frame_deinit(&result_frame);
//            }
//            return false;
//        }
//
//        // 动态获取当前帧的真实宽高
//        uint32_t actual_width  = mpp_frame_get_width(result_frame);
//        uint32_t actual_height = mpp_frame_get_height(result_frame);
//        MppFrameFormat fmt     = mpp_frame_get_fmt(result_frame);
//
//        // --- 6. 边界安全检查（防止 RGA 硬件由于非法输入引发挂起或报错） ---
//        if (crop_rect.x < 0 || crop_rect.y < 0 ||
//                                             crop_rect.x + crop_rect.width > (int)actual_width ||
//                                                                                  crop_rect.y + crop_rect.height > (int)actual_height ||
//                                                                                  crop_rect.width <= 0 || crop_rect.height <= 0) {
//            std::cerr << "Invalid crop rect! Frame size: " << actual_width << "x" << actual_height
//                      << ", Crop rect: [" << crop_rect.x << ", " << crop_rect.y
//                      << ", " << crop_rect.width << "x" << crop_rect.height << "]" << std::endl;
//            if (result_frame != output_frame_) {
//                mpp_frame_deinit(&result_frame);
//            }
//            return false;
//        }
//
//        // 切换缓冲索引
//        current_bgr_idx_ = (current_bgr_idx_ + 1) % 2;
//
//        // --- 7. 调用带裁剪的 RGA 处理 ---
//        bool success = process_with_rga_crop(result_frame, actual_width, actual_height, fmt, crop_rect, scale_to_target, current_bgr_idx_);
//
//        if (result_frame != output_frame_) {
//            mpp_frame_deinit(&result_frame);
//        }
//
//        if (success) {
//            if (scale_to_target) {
//                out_mat = cv::Mat(height_, width_, CV_8UC3, bgr_ptr_[current_bgr_idx_]);
//            } else {
//                // 不缩放时，由于 RGA 写入目标内存是紧凑排列的，直接用裁剪宽高映射 Mat
//                out_mat = cv::Mat(crop_rect.height, crop_rect.width, CV_8UC3, bgr_ptr_[current_bgr_idx_]);
//            }
//        }
//        return success;
//    }

private:
    MppCtx mpp_ctx_ = nullptr;
    MppApi *mpp_api_ = nullptr;
    int width_, height_;
    bool initialized_ = false;

    MppBufferGroup mem_group_ = nullptr;
    MppBuffer yuv_buf_ = nullptr;

    // 修改为双缓冲数组
    MppBuffer bgr_buf_[2] = {nullptr, nullptr};
    int bgr_fd_[2] = {-1, -1};
    void* bgr_ptr_[2] = {nullptr, nullptr};

    // 当前使用的 buffer 索引
    int current_bgr_idx_ = 0;

    // 新增：全局唯一的输出 Frame 实例
    MppFrame output_frame_ = nullptr;

    void allocate_buffers() {
        MPP_RET ret = MPP_OK;
        size_t bgr_size = width_ * height_ * 3;

        // 循环分配两个 BGR 缓冲区
        for (int i = 0; i < 2; ++i) {
            mpp_buffer_get(mem_group_, &bgr_buf_[i], bgr_size);
            bgr_fd_[i]  = mpp_buffer_get_fd(bgr_buf_[i]);
            bgr_ptr_[i] = mpp_buffer_get_ptr(bgr_buf_[i]);
        }

        // 对齐官方：YUV 缓冲区按照 4 倍大小极限分配，防止 422SP 越界
        uint32_t hor_stride = MPP_ALIGN(width_, 16);
        uint32_t ver_stride = MPP_ALIGN(height_, 16);
        size_t yuv_size = hor_stride * ver_stride * 4;
        ret = mpp_buffer_get(mem_group_, &yuv_buf_, yuv_size);
        if (ret) {
            std::cerr << "mpp_buffer_get failed:" << ret << std::endl;
        }
        ret = mpp_frame_init(&output_frame_);
        if (ret) {
            std::cerr << "mpp_frame_init failed:" << ret << std::endl;
        }
        mpp_frame_set_buffer(output_frame_, yuv_buf_);
    }

    bool process_with_rga(MppFrame decoded_frame,
                          uint32_t fw, uint32_t fh,
                          MppFrameFormat fmt,
                          int bgr_idx) {
        int src_fd = mpp_buffer_get_fd(yuv_buf_);
        uint32_t hor_stride = mpp_frame_get_hor_stride(decoded_frame);
        uint32_t ver_stride = mpp_frame_get_ver_stride(decoded_frame);

        int rga_src_fmt;
        switch (fmt & MPP_FRAME_FMT_MASK) {
            case MPP_FMT_YUV420SP: rga_src_fmt = RK_FORMAT_YCbCr_420_SP; break;
            case MPP_FMT_YUV422SP: rga_src_fmt = RK_FORMAT_YCbCr_422_SP; break;
            case MPP_FMT_YUV420P:  rga_src_fmt = RK_FORMAT_YCbCr_420_P;  break;
            case MPP_FMT_YUV422P:  rga_src_fmt = RK_FORMAT_YCbCr_422_P;  break;
            default:
                std::cerr << "Unsupported MPP format: " << (fmt & MPP_FRAME_FMT_MASK) << std::endl;
                return false;
        }

        rga_buffer_t src_img = wrapbuffer_fd(src_fd, (int)fw, (int)fh,
                                             rga_src_fmt,
                                             (int)hor_stride, (int)ver_stride);

        // 使用对应的 bgr_fd_[bgr_idx]
        rga_buffer_t dst_img = wrapbuffer_fd(bgr_fd_[bgr_idx], width_, height_, RK_FORMAT_BGR_888);

        IM_STATUS rga_ret = imcvtcolor(src_img, dst_img, src_img.format, dst_img.format);
        if (rga_ret != IM_STATUS_SUCCESS) {
            std::cerr << "RGA conversion failed: " << rga_ret << std::endl;
            return false;
        }
        return true;
    }

//    /**
//     * @brief 专用于裁剪的 RGA 核心调用
//     */
//    bool process_with_rga_crop(MppFrame decoded_frame,
//                               uint32_t fw, uint32_t fh,
//                               MppFrameFormat fmt,
//                               cv::Rect crop_rect,
//                               bool scale_to_target,
//                               int bgr_idx) {
//        int src_fd = mpp_buffer_get_fd(yuv_buf_);
//        uint32_t hor_stride = mpp_frame_get_hor_stride(decoded_frame);
//        uint32_t ver_stride = mpp_frame_get_ver_stride(decoded_frame);
//
//        int rga_src_fmt;
//        switch (fmt & MPP_FRAME_FMT_MASK) {
//            case MPP_FMT_YUV420SP: rga_src_fmt = RK_FORMAT_YCbCr_420_SP; break;
//            case MPP_FMT_YUV422SP: rga_src_fmt = RK_FORMAT_YCbCr_422_SP; break;
//            case MPP_FMT_YUV420P:  rga_src_fmt = RK_FORMAT_YCbCr_420_P;  break;
//            case MPP_FMT_YUV422P:  rga_src_fmt = RK_FORMAT_YCbCr_422_P;  break;
//            default:
//                std::cerr << "Unsupported MPP format: " << (fmt & MPP_FRAME_FMT_MASK) << std::endl;
//                return false;
//        }
//
//        // 1. 包装完整的源输入 buffer
//        rga_buffer_t src_img = wrapbuffer_fd(src_fd, (int)fw, (int)fh,
//                                             rga_src_fmt,
//                                             (int)hor_stride, (int)ver_stride);
//
//        // 2. 根据模式包装目标输出 buffer
//        rga_buffer_t dst_img;
//        im_rect_t dst_roi;
//
//        if (scale_to_target) {
//            // 模式 A：满幅输出，RGA 会自动做 Bilinear 缩放
//            dst_img = wrapbuffer_fd(bgr_fd_[bgr_idx], width_, height_, RK_FORMAT_BGR_888);
//            dst_roi = { 0, 0, width_, height_ };
//        } else {
//            // 模式 B：1:1 输出。将 dst 虚拟宽高设为裁剪宽高，使 RGA 硬件在内存中紧凑 packed 写入
//            // 注意：需确保 crop_rect.width * crop_rect.height * 3 <= bgr_buf_ 分配的总大小
//            if (crop_rect.width * crop_rect.height * 3 > width_ * height_ * 3) {
//                std::cerr << "Crop size exceeds pre-allocated BGR buffer size!" << std::endl;
//                return false;
//            }
//            dst_img = wrapbuffer_fd(bgr_fd_[bgr_idx], crop_rect.width, crop_rect.height, RK_FORMAT_BGR_888);
//            dst_roi = { 0, 0, crop_rect.width, crop_rect.height };
//        }
//
//        // 3. 定义源 ROI (裁剪窗口)
//        im_rect_t src_roi = { crop_rect.x, crop_rect.y, crop_rect.width, crop_rect.height };
//
//        // 4. 调用 improcess。它是 librga 的全能核心接口，显式传递 src_roi 和 dst_roi
//        // 硬件内部会完成：[从 src_fd 切出 src_roi] -> [颜色空间转换及缩放] -> [写入 dst_img 的 dst_roi 区域]
//        IM_STATUS rga_ret = improcess(src_img, dst_img, {}, src_roi, dst_roi, {}, 0);
//        if (rga_ret != IM_STATUS_SUCCESS) {
//            std::cerr << "RGA crop/scale/csc operation failed: " << rga_ret << std::endl;
//            return false;
//        }
//        return true;
//    }
};

#endif //LIVOX_COLOR_MPP_RGA_DECODER_H