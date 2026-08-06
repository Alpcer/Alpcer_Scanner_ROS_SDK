//
// Created by Zachary on 2026/7/31.
//

#ifndef LIVOX_COLOR_CAMERA_REALTIME_H
#define LIVOX_COLOR_CAMERA_REALTIME_H
#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/poll.h>
#include <linux/videodev2.h>

// Rockchip MPP & RGA Headers
#include <rockchip/rk_mpi.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/mpp_buffer.h>
#include <rockchip/mpp_meta.h>
#include <rga/im2d.h>

#include <opencv2/opencv.hpp>

#ifndef MPP_ALIGN
#define MPP_ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))
#endif
namespace CAMERA_REALTIME {
    // ============================================================================
// 1. MPP + RGA 硬件解码器（封装与内存零分配）
// ============================================================================
    class MppRgaDecoder {
    public:
        MppRgaDecoder(int target_width, int target_height)
                : width_(target_width), height_(target_height) {

            MPP_RET ret = mpp_create(&mpp_ctx_, &mpp_api_);
            if (ret != MPP_OK) {
                std::cerr << "[MPP] Failed to create MPP context: " << ret << std::endl;
                return;
            }

            MppDecCfg cfg = nullptr;
            mpp_dec_cfg_init(&cfg);
            ret = mpp_api_->control(mpp_ctx_, MPP_DEC_GET_CFG, cfg);
            if (ret) {
                std::cerr << "[MPP] Failed to get decoder cfg ret:" << ret << std::endl;
            }
            ret = mpp_dec_cfg_set_u32(cfg, "base:split_parse", 1);
            if (ret) {
                std::cerr << "[MPP] Failed to set split_parse ret:" << ret << std::endl;
            }
            ret = mpp_api_->control(mpp_ctx_, MPP_DEC_SET_CFG, cfg);
            if (ret) {
                std::cerr << "[MPP] Failed to set cfg ret:" << ret << std::endl;
            }
            mpp_dec_cfg_deinit(cfg);

            ret = mpp_init(mpp_ctx_, MPP_CTX_DEC, MPP_VIDEO_CodingMJPEG);
            if (ret != MPP_OK) {
                std::cerr << "[MPP] Failed to init MPP for MJPEG: " << ret << std::endl;
                return;
            }

            mpp_buffer_group_get_internal(&mem_group_, static_cast<MppBufferType>(
                    MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_DMA32 | MPP_BUFFER_FLAGS_CACHABLE));

            allocate_buffers();
            initialized_ = true;
        }

        ~MppRgaDecoder() {
            if (mpp_ctx_ && mpp_api_) {
                mpp_api_->reset(mpp_ctx_);

                MppFrame leftover_frame = nullptr;
                while (mpp_api_->decode_get_frame(mpp_ctx_, &leftover_frame) == MPP_OK && leftover_frame) {
                    if (leftover_frame != output_frame_) {
                        mpp_frame_deinit(&leftover_frame);
                    }
                    leftover_frame = nullptr;
                }

                if (output_frame_) {
                    mpp_frame_deinit(&output_frame_);
                    output_frame_ = nullptr;
                }

                mpp_destroy(mpp_ctx_);
                mpp_ctx_ = nullptr;
                mpp_api_ = nullptr;
            }

            if (yuv_buf_) {
                mpp_buffer_put(yuv_buf_);
                yuv_buf_ = nullptr;
            }
            if (bgr_buf_) {
                mpp_buffer_put(bgr_buf_);
                bgr_buf_ = nullptr;
            }
            if (mem_group_) {
                mpp_buffer_group_put(mem_group_);
                mem_group_ = nullptr;
            }
        }

        bool isInitialized() const { return initialized_; }

        bool decode(const uint8_t* jpeg_data, size_t size, cv::Mat& out_mat) {
            if (!initialized_ || !jpeg_data || size == 0) return false;

            MppPacket packet = nullptr;
            MppBuffer pkt_buf = nullptr;
            MPP_RET ret;

            // 1. 拷贝 JPEG 数据到 DMA 缓冲区
            ret = mpp_buffer_get(mem_group_, &pkt_buf, size);
            if (ret != MPP_OK || !pkt_buf) {
                return false;
            }
            void* buf_ptr = mpp_buffer_get_ptr(pkt_buf);
            if (!buf_ptr) {
                mpp_buffer_put(pkt_buf);
                return false;
            }
            std::memcpy(buf_ptr, jpeg_data, size);

            // 2. 用 DMA 缓冲区初始化 Packet
            ret = mpp_packet_init_with_buffer(&packet, pkt_buf);
            mpp_buffer_put(pkt_buf);
            if (ret != MPP_OK) return false;

            // 3. 注入全局唯一的 output_frame_
            MppMeta meta = mpp_packet_get_meta(packet);
            if (meta) {
                mpp_meta_set_frame(meta, KEY_OUTPUT_FRAME, output_frame_);
            }

            // 4. 发送到解码器
            ret = mpp_api_->decode_put_packet(mpp_ctx_, packet);
            mpp_packet_deinit(&packet);
            if (ret != MPP_OK) return false;

            // 5. 获取解码结果
            MppFrame result_frame = nullptr;
            ret = mpp_api_->decode_get_frame(mpp_ctx_, &result_frame);

            if (ret != MPP_OK || !result_frame) return false;

            if (mpp_frame_get_errinfo(result_frame) != 0) return false;

            // 6. RGA 颜色空间转换 (YUV422/420 -> BGR888)
            uint32_t actual_width  = mpp_frame_get_width(result_frame);
            uint32_t actual_height = mpp_frame_get_height(result_frame);
            MppFrameFormat fmt     = mpp_frame_get_fmt(result_frame);

            bool success = process_with_rga(result_frame, actual_width, actual_height, fmt);

            if (success) {
                out_mat = cv::Mat(height_, width_, CV_8UC3, bgr_ptr_);
            }
            return success;
        }

    private:
        void allocate_buffers() {
            size_t bgr_size = width_ * height_ * 3;
            mpp_buffer_get(mem_group_, &bgr_buf_, bgr_size);
            bgr_fd_  = mpp_buffer_get_fd(bgr_buf_);
            bgr_ptr_ = mpp_buffer_get_ptr(bgr_buf_);

            uint32_t hor_stride = MPP_ALIGN(width_, 16);
            uint32_t ver_stride = MPP_ALIGN(height_, 16);
            size_t yuv_size = hor_stride * ver_stride * 4;

            mpp_buffer_get(mem_group_, &yuv_buf_, yuv_size);
            mpp_frame_init(&output_frame_);
            mpp_frame_set_buffer(output_frame_, yuv_buf_);
        }

        bool process_with_rga(MppFrame decoded_frame, uint32_t fw, uint32_t fh, MppFrameFormat fmt) {
            int src_fd = mpp_buffer_get_fd(yuv_buf_);
            uint32_t hor_stride = mpp_frame_get_hor_stride(decoded_frame);
            uint32_t ver_stride = mpp_frame_get_ver_stride(decoded_frame);

            int rga_src_fmt;
            switch (fmt & MPP_FRAME_FMT_MASK) {
                case MPP_FMT_YUV420SP: rga_src_fmt = RK_FORMAT_YCbCr_420_SP; break;
                case MPP_FMT_YUV422SP: rga_src_fmt = RK_FORMAT_YCbCr_422_SP; break;
                case MPP_FMT_YUV420P:  rga_src_fmt = RK_FORMAT_YCbCr_420_P;  break;
                case MPP_FMT_YUV422P:  rga_src_fmt = RK_FORMAT_YCbCr_422_P;  break;
                default: return false;
            }

            rga_buffer_t src_img = wrapbuffer_fd(src_fd, (int)fw, (int)fh, rga_src_fmt, (int)hor_stride, (int)ver_stride);
            rga_buffer_t dst_img = wrapbuffer_fd(bgr_fd_, width_, height_, RK_FORMAT_BGR_888);

            IM_STATUS rga_ret = imcvtcolor(src_img, dst_img, src_img.format, dst_img.format);
            return (rga_ret == IM_STATUS_SUCCESS);
        }

        MppCtx mpp_ctx_ = nullptr;
        MppApi *mpp_api_ = nullptr;
        int width_, height_;
        bool initialized_ = false;

        MppBufferGroup mem_group_ = nullptr;
        MppBuffer yuv_buf_ = nullptr;
        MppBuffer bgr_buf_ = nullptr;
        MppFrame output_frame_ = nullptr;

        int bgr_fd_ = -1;
        void* bgr_ptr_ = nullptr;
    };

// ============================================================================
// 2. V4L2 高速实时摄像头采集类 (带硬解流水线)
// ============================================================================
    class CameraRealtime {
    public:
        struct V4L2Buffer {
            void* start = nullptr;
            size_t length = 0;
        };

        /**
         * @param dev_path  设备路径, 如 "/dev/video0"
         * @param width     分辨率宽度 (例如 3840 或 3200)
         * @param height    分辨率高度 (例如 1920 或 1200)
         * @param fps       捕获帧率 (默认 60)
         */
        CameraRealtime(const std::string& dev_path, int width, int height, int fps = 60)
                : dev_path_(dev_path), width_(width), height_(height), fps_(fps) {}

        ~CameraRealtime() {
            stop();
        }

        bool start() {
            if (is_running_) return true;

            // 初始化 MPP 硬解码器
            decoder_ = std::make_unique<MppRgaDecoder>(width_, height_);
            if (!decoder_->isInitialized()) {
                std::cerr << "[CameraRealtime] Error: Failed to initialize MPP Decoder." << std::endl;
                return false;
            }

            if (!initV4L2()) return false;

            v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
                std::cerr << "[CameraRealtime] Error: VIDIOC_STREAMON failed." << std::endl;
                uninitV4L2();
                return false;
            }

            is_running_ = true;
            worker_ = std::thread(&CameraRealtime::capturePipelineLoop, this);
            return true;
        }

        void stop() {
            if (!is_running_) return;
            is_running_ = false;

            if (worker_.joinable()) {
                worker_.join();
            }

            if (fd_ >= 0) {
                v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                ioctl(fd_, VIDIOC_STREAMOFF, &type);
            }
            uninitV4L2();
            decoder_.reset();
        }

        /**
         * @brief 非阻塞获取最新的实时画面
         * @param frame 输出解码后的 Mat 图像
         * @param timestamp_ms 毫秒级 capture 时间戳
         */
        bool getLatestFrame(cv::Mat& frame, uint64_t* timestamp_ms = nullptr) {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            if (!has_frame_ || latest_frame_.empty()) {
                return false;
            }
            frame = latest_frame_.clone(); // 深拷贝，防止主线程使用时被后台解码覆盖
            if (timestamp_ms) {
                *timestamp_ms = latest_timestamp_ms_;
            }
            return true;
        }

    private:
        bool initV4L2() {
            fd_ = open(dev_path_.c_str(), O_RDWR | O_NONBLOCK, 0);
            if (fd_ < 0) {
                std::cerr << "[CameraRealtime] Error: Cannot open device " << dev_path_ << std::endl;
                return false;
            }

            // 设置像素格式为 MJPEG
            v4l2_format fmt{};
            fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            fmt.fmt.pix.width = width_;
            fmt.fmt.pix.height = height_;
            fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
            fmt.fmt.pix.field = V4L2_FIELD_ANY;

            if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
                std::cerr << "[CameraRealtime] Error: Failed to set V4L2 format." << std::endl;
                close(fd_);
                fd_ = -1;
                return false;
            }

            // 设置目标帧率
            v4l2_streamparm streamparm{};
            streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            streamparm.parm.capture.timeperframe.numerator = 1;
            streamparm.parm.capture.timeperframe.denominator = fps_;
            ioctl(fd_, VIDIOC_S_PARM, &streamparm);

            // 申请 MMAP 内存缓冲区
            v4l2_requestbuffers req{};
            req.count = BUFFER_COUNT;
            req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            req.memory = V4L2_MEMORY_MMAP;

            if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0 || req.count < 2) {
                std::cerr << "[CameraRealtime] Error: Insufficient V4L2 buffer memory." << std::endl;
                close(fd_);
                fd_ = -1;
                return false;
            }

            buffers_.resize(req.count);
            for (size_t i = 0; i < req.count; ++i) {
                v4l2_buffer buf{};
                buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                buf.memory = V4L2_MEMORY_MMAP;
                buf.index = i;

                if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) return false;

                buffers_[i].length = buf.length;
                buffers_[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
                if (buffers_[i].start == MAP_FAILED) return false;
            }

            for (size_t i = 0; i < req.count; ++i) {
                v4l2_buffer buf{};
                buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                buf.memory = V4L2_MEMORY_MMAP;
                buf.index = i;
                if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) return false;
            }

            return true;
        }

        void uninitV4L2() {
            for (auto& buf : buffers_) {
                if (buf.start && buf.start != MAP_FAILED) {
                    munmap(buf.start, buf.length);
                }
            }
            buffers_.clear();
            if (fd_ >= 0) {
                close(fd_);
                fd_ = -1;
            }
        }

        void capturePipelineLoop() {
            struct pollfd fds[1];
            fds[0].fd = fd_;
            fds[0].events = POLLIN;

            cv::Mat decoded_mat;

            while (is_running_) {
                int ret = poll(fds, 1, 100); // 100ms 超时
                if (ret <= 0) continue;

                v4l2_buffer buf{};
                buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                buf.memory = V4L2_MEMORY_MMAP;

                // 取出 V4L2 内核填充好的 Buffer
                if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
                    continue;
                }

                uint64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();

                // 流水线关键操作：使用 MPP 硬解码驱动解出当前 MJPEG 帧
                const uint8_t* mjpeg_ptr = static_cast<uint8_t*>(buffers_[buf.index].start);
                size_t mjpeg_size = buf.bytesused;

                bool decode_success = decoder_->decode(mjpeg_ptr, mjpeg_size, decoded_mat);

                // 解码完成后立刻将 V4L2 buffer QBUF 重新入队，防止底层溢出丢帧
                ioctl(fd_, VIDIOC_QBUF, &buf);

                // 更新缓存区供 getLatestFrame 获取
                if (decode_success) {
                    std::lock_guard<std::mutex> lock(frame_mutex_);
                    latest_frame_ = decoded_mat.clone();
                    latest_timestamp_ms_ = now_ms;
                    has_frame_ = true;
                }
            }
        }

        std::string dev_path_;
        int width_;
        int height_;
        int fps_;
        int fd_ = -1;

        static constexpr size_t BUFFER_COUNT = 4;
        std::vector<V4L2Buffer> buffers_;

        std::unique_ptr<MppRgaDecoder> decoder_;
        std::thread worker_;
        std::atomic<bool> is_running_{false};

        std::mutex frame_mutex_;
        cv::Mat latest_frame_;
        uint64_t latest_timestamp_ms_ = 0;
        bool has_frame_ = false;
    };
}
#endif //LIVOX_COLOR_CAMERA_REALTIME_H
