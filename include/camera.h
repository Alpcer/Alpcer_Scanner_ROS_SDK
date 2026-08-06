//
// Created by Zachary on 2024/12/27.
//

#ifndef CAMERA_TEST_TOOL_CAMERA_H
#define CAMERA_TEST_TOOL_CAMERA_H
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <string>
#include <utility>
#include <cstring>
#include <sys/mman.h>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <random>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <opencv2/xphoto.hpp>
#include <opencv2/imgproc.hpp>
#include "common_utils.h"
#include <eigen3/Eigen/Dense>
#include "spdlog/spdlog.h"
#include <iostream>
#include <fstream>
#include "mpp_rga_decoder.h"
#include <mutex>

#define SKIP_FRAMES 5
#define SKIP_FRAMES_AUTO 10

namespace CAMERA {
    using namespace cv;
    using namespace std;

    const int EV_LEVEL_ARRAY[] = {10000, 5000, 2500, 1250, 625, 312, 156, 78, 39, 20, 10};
    const int EV_LEVEL_COUNT = 11;
    const int LOW_TO_HIGH_EV_LEVEL_MAP[] = {0, 2, 6, 7, 8, 8, 9, 9, 10, 10, 10};
    const double HIGH_TO_LOW_EV_LEVEL_MAP[] = {0, 0.1304, 1.0033, 1.1678, 1.2960, 1.5799, 1.9413, 2.7692, 4.2380, 6.0282, 7.8153};

    struct CameraParams {
        Range row_range;
        Range col_range;
        int width = 0;
        int height = 0;
        int centerX = 0;
        int centerY = 0;
        int radius = 0;
        cv::Matx33d k;    /*    摄像机内参数矩阵    */
        cv::Vec4d d;     /* 摄像机的4个畸变系数：k1,k2,k3,k4*/
    };

    class CameraInternal {

    public:
        const static int INITIAL_WIDTH = 640;
        const static int INITIAL_HEIGHT = 480;
        const uint32_t width;
        const uint32_t height;
        const uint32_t width_little;
        const uint32_t height_little;
        std::vector<uchar> buf, buf_auto, buf_over, buf_under;
        const std::string comName;

        Range valid_row_range, valid_col_range;
        Mat valid_range_mask;

        bool useRgaResize = false;
        Range valid_row_range_resized, valid_col_range_resized;

    private:
        uint32_t cur_width = INITIAL_WIDTH;
        uint32_t cur_height = INITIAL_HEIGHT;
        bool cur_is_auto = true;

        const std::string cameraParamsPath;
        const int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        volatile int fd = -1;
        unsigned char *mptr[4];//保存映射后用户空间的首地址
        unsigned int size[4];
        struct v4l2_requestbuffers requestBuffer;
        struct v4l2_buffer frameBuffer;
        struct v4l2_control ctrl;
        struct v4l2_format fmt;

        volatile bool kernelSpaceRequested = false;

        CameraParams cameraParamsMain;
        CameraParams cameraParamsLittle;

        MppRgaDecoder* pMppRgaDecoder = nullptr;
        std::mutex camera_mutex;
        std::vector<uchar> cached_image_raw;
        size_t bytesused_image_raw;
        long timestamp_image_raw;
        cv::Mat cached_image;

    public:
        CameraInternal(std::string comName, uint32_t width, uint32_t height, uint32_t width_little, uint32_t height_little,  std::string cameraParamsPath) : comName(std::move(comName)), width(width), height(height), width_little(width_little), height_little(height_little), cameraParamsPath(std::move(cameraParamsPath)) {
            init();
            initShootingParams();
        }
        ~CameraInternal() {
            releaseKernelSpace();
            closeFd();
            if (pMppRgaDecoder != nullptr) {
                delete pMppRgaDecoder;
                pMppRgaDecoder = nullptr;
            }
        }

        inline void setAutoExposure(bool isAuto, bool wait_for_frame_update = false) {
            if (isAuto) {
                if (!cur_is_auto) {
                    ctrl.id = V4L2_CID_EXPOSURE_AUTO;
                    ctrl.value = V4L2_EXPOSURE_APERTURE_PRIORITY;
                    if (ioctl(fd, VIDIOC_S_CTRL, &ctrl) < 0) {
                        throw std::runtime_error("设置自动曝光失败");
                    }
                    cur_is_auto = true;
                }
                if (wait_for_frame_update) {
                    skipFrames(SKIP_FRAMES_AUTO);
                }
            } else {
                if (cur_is_auto) {
                    ctrl.id = V4L2_CID_EXPOSURE_AUTO;
                    ctrl.value = V4L2_EXPOSURE_MANUAL;
                    if (ioctl(fd, VIDIOC_S_CTRL, &ctrl) < 0) {
                        throw std::runtime_error("设置手动曝光失败");
                    }
                    cur_is_auto = false;
                }
            }
        }

        void setExposure(int exposure) {
            ctrl.id = V4L2_CID_EXPOSURE_ABSOLUTE;
            ctrl.value = exposure;
            if (ioctl(fd, VIDIOC_S_CTRL, &ctrl) < 0) {
                throw std::runtime_error("设置曝光时间失败");
            }
        }

        void streamOn() {
            if (ioctl(fd,VIDIOC_STREAMON,&type) < 0) {
                throw std::runtime_error("开启摄像头失败");
            }
        }

        void streamOff() {
            //停止采集
            if (ioctl(fd,VIDIOC_STREAMOFF,&type) < 0) {
                throw std::runtime_error("关闭摄像头失败");
            }
        }

        void skipFrames(int num) {
            for (int i=0; i<num; i++) {
                queueFrameBuffer();
                dequeueFrameBuffer();
            }
        }

        inline void copyFrameData(std::vector<uchar>* p_buf = nullptr) {
            if (p_buf == nullptr) {
                memcpy(buf.data(), mptr[0], buf.size());
            } else {
                p_buf->resize(buf.size());
                memcpy(p_buf->data(), mptr[0], p_buf->size());
            }
        }

        inline void acquireFrame(std::vector<uchar>* p_buf = nullptr) {
            queueFrameBuffer();
            dequeueFrameBuffer();
            copyFrameData(p_buf);
        }

        static inline void correctImageRotation(Mat& src, Mat& dst) {
            rotate(src, dst, RotateFlags::ROTATE_180);
        }

        Mat decodeBufferFrame(bool onlyValidRange, bool rotationCorrect, std::vector<uchar>* p_buf = nullptr) {
            Mat frame;
            if (cur_width == width) {
                frame = onlyValidRange ? imdecode(p_buf == nullptr ? buf : (*p_buf), ImreadModes::IMREAD_COLOR)(valid_row_range, valid_col_range) : imdecode(p_buf == nullptr ? buf : (*p_buf), ImreadModes::IMREAD_COLOR);
            } else {
                Mat yuyv(cur_height, cur_width, CV_8UC2, p_buf == nullptr ? buf.data() : p_buf->data()), bgr;
                cvtColor(yuyv, bgr, COLOR_YUV2BGR_YUYV);
                frame = onlyValidRange ? bgr(valid_row_range, valid_col_range) : bgr;
            }

            if (rotationCorrect) {
                Mat frame2;
                correctImageRotation(frame, frame2);
                return frame2;
            } else {
                return frame;
            }
        }

        Mat decodeBufferFrameGray(bool onlyValidRange, bool rotationCorrect, std::vector<uchar>* p_buf = nullptr) {
            Mat frame;
            if (cur_width == width) {
                frame = onlyValidRange ? imdecode(p_buf == nullptr ? buf : (*p_buf), ImreadModes::IMREAD_GRAYSCALE)(valid_row_range, valid_col_range) : imdecode(p_buf == nullptr ? buf : (*p_buf), ImreadModes::IMREAD_GRAYSCALE);
            } else {
                Mat yuyv(cur_height, cur_width, CV_8UC2, p_buf == nullptr ? buf.data() : p_buf->data()), bgr, gray;
                cvtColor(yuyv, bgr, COLOR_YUV2BGR_YUYV);
                cvtColor(bgr, gray, COLOR_BGR2GRAY);
                frame = onlyValidRange ? gray(valid_row_range, valid_col_range) : gray;
            }
            if (rotationCorrect) {
                Mat frame2;
                correctImageRotation(frame, frame2);
                return frame2;
            } else {
                return frame;
            }
        }

        inline size_t dequeueFrameBuffer() {
            fd_set fds;
            struct timeval tv;

            FD_ZERO(&fds);
            FD_SET(fd, &fds);
            tv.tv_sec = 5;
            tv.tv_usec = 0;
            int ret = select(fd + 1, &fds, nullptr, nullptr, &tv);

            if (ret == 0) {
                spdlog::info("等待数据超时，尝试重新开关流\n");
                streamOff();
                streamOn();
                queueFrameBuffer();

                FD_ZERO(&fds);
                FD_SET(fd, &fds);
                tv.tv_sec = 5;
                tv.tv_usec = 0;
                ret = select(fd + 1, &fds, nullptr, nullptr, &tv);
                if (ret == 0) {
                    throw std::runtime_error("等待数据超时");
                }
            }
            if (ret == -1) {
                throw std::runtime_error("等待数据出错");
            }
            if (ioctl(fd,VIDIOC_DQBUF,&frameBuffer) < 0) {
                throw std::runtime_error("读取数据失败");
            }
            return frameBuffer.bytesused;
        }

        inline void queueFrameBuffer() {
            if(ioctl(fd, VIDIOC_QBUF, &frameBuffer) < 0) {
                throw std::runtime_error("放回队列失败");
            }
        }

        double decodeBufferFrameGrayValue(std::vector<uchar>* p_buf = nullptr) {
            return mean(decodeBufferFrameGray(false, false, p_buf), valid_range_mask).val[0];
        }

        void setSolution(bool main) {
            if ((main && cur_width != width) || (!main && cur_width != width_little)) {
                streamOff();
                releaseKernelSpace();
                setSolutionInternal(main);
                requestKernelSpace();
                streamOn();
            }
        }

        double measureExposureLvByBase(double gv_base) {
            long t = common_utils::currentTimeMilliseconds();

            setSolution(false);

            setAutoExposure(false);
            //初始化曝光等级灰度表
            double lv_grays[EV_LEVEL_COUNT];
            for (int i=0; i<EV_LEVEL_COUNT; i++) {
                lv_grays[i] = -1.0;
            }

            const int lv_origin = 5;
            int lv = lv_origin, lv_last, lv_e_max = 0, lv_e_min = 10;
            int calculateTimes = 0;
            Mat tmp, gray;

            bool cal_next;
            int lv_left = 0, lv_right = 10;
            double gv_cur;
            double lv_f;
            do {
                //获取当前lv灰度，更新目标灰度区间
                setExposure(EV_LEVEL_ARRAY[lv]);
                skipFrames(SKIP_FRAMES);
                acquireFrame();
                gv_cur = decodeBufferFrameGrayValue();
                spdlog::info("{} measure base {}, gv_cur:{}, gv_base:{}, cur exposure:{}\n", comName.c_str(), calculateTimes, gv_cur, gv_base, EV_LEVEL_ARRAY[lv]);

                lv_grays[lv] = gv_cur;
                if (lv_right - lv_left > 1) {
                    if (gv_cur > gv_base) {//拍得过亮
                        lv_left = lv;
                    } else {
                        lv_right = lv;
                    }
                }

                //是否已找到目标档位(1. 包围完成;)
                if (lv_right - lv_left == 1 && lv_grays[lv_left] > 0 && lv_grays[lv_right] > 0) {
                    spdlog::info("{} found! lv_left:{} gv:{} lv_right:{} gv:{} gv_cur:{} gv_base:{}\n", comName.c_str(), lv_left, lv_grays[lv_left], lv_right, lv_grays[lv_right], gv_cur, gv_base);
                    lv_f = (lv_grays[lv_left] - gv_base)/(lv_grays[lv_left] - lv_grays[lv_right]) + lv_left;
                    spdlog::info("{} accurate lv {}\n", comName.c_str(), lv_f);
                    break;
                }

                //未找到目标档位，更新下一次拍摄档位
                if (lv_left == lv) {//目标区间右缩，新档位以左边为基准向右计算
                    if (lv_right - lv_left > 2) {
                        lv = lv_left + (calculateTimes == 0 ? 4 : 2);
                    } else {
                        lv = lv_left + 1;
                    }
                } else {
                    if (lv_right - lv_left > 2) {
                        lv = lv_right - (calculateTimes == 0 ? 4 : 2);
                    } else {
                        lv = lv_right - 1;
                    }
                }

                calculateTimes++;
            } while (calculateTimes < 10);

            setSolution(true);

            spdlog::info("{} Measure cost time: {} ms\n", comName.c_str(), common_utils::currentTimeMilliseconds() - t);
            return lv_f;
        }

        bool projectPoints(std::vector<Point3f>& objPoints, std::vector<Point2f>& imgPoints) {
            cv::Vec3d rotation, translation(0,0,0);
            Rodrigues(Mat::eye(3, 3, CV_64F), rotation);
            fisheye::projectPoints(objPoints, imgPoints, rotation, translation, cameraParamsMain.k, cameraParamsMain.d);
            float max_x = cameraParamsMain.col_range.end - cameraParamsMain.col_range.start - 1.0f;
            float max_y = cameraParamsMain.row_range.end - cameraParamsMain.row_range.start - 1.0f;
            for (auto& p2f : imgPoints) {
                if (p2f.x < 0 || p2f.y < 0 || p2f.x > max_x || p2f.y > max_y) {
                    return false;
                }
            }
            return true;
        }

        CameraParams getMainCameraParams() {
            return cameraParamsMain;
        }

        std::array<double,4> undistort(cv::Mat& distorted_img, cv::Mat& undistorted_img, cv::Mat& r) {
            cv::Mat mapX, mapY;
            cv::Matx33d k = cv::Matx33d::eye();
            int w = 512, h = 960;

            k(0, 2) = w/2.0;//cx
            k(1, 2) = h/2.0;//cy
            k(0, 0) = w*0.6;//fx
            k(1, 1) = w*0.6;//fy

            cv::fisheye::initUndistortRectifyMap(cameraParamsMain.k, cameraParamsMain.d, r, k, cv::Size(w, h), CV_16SC2, mapX, mapY);
            cv::remap(distorted_img, undistorted_img, mapX, mapY, cv::INTER_LINEAR);
            std::array<double,4> intrinsic{};//fx,fy,cx,cy
            intrinsic[0] = k(0, 0);//fx
            intrinsic[1] = k(1, 1);//fy
            intrinsic[2] = k(0, 2);//cx
            intrinsic[3] = k(1, 2);//cy
            return intrinsic;
        }

        void shootExposureBracketing(std::vector<std::vector<uchar>>& images_raw) {
            spdlog::info("{} shootExposureBracketing start.\n", comName.c_str());
            images_raw.resize(1);
            setAutoExposure(true, false);
            acquireFrame(&(images_raw[0]));
            double gv_base;
            int retry_times = 1;
            while ((gv_base = decodeBufferFrameGrayValue(&(images_raw[0]))) > 170.0 || gv_base < 70.0 && retry_times > 0) {
                spdlog::info("Base-auto is overexposure, reshoot times: {}", retry_times);
                skipFrames(SKIP_FRAMES);
                acquireFrame(&(images_raw[0]));
                retry_times--;
            }
            if (retry_times <=0) {
                spdlog::info("Reshoot times exceeded, going on.");
            }

            double lv_f = measureExposureLvByBase(gv_base);
            //获取在大分辨率上的档次区间
            int lv_target_left = EV_LEVEL_COUNT-1, lv_target_right = EV_LEVEL_COUNT-1;
            for (int i=0; i<EV_LEVEL_COUNT; i++) {
                if (lv_f >= HIGH_TO_LOW_EV_LEVEL_MAP[i] && lv_f < HIGH_TO_LOW_EV_LEVEL_MAP[i+1]) {
                    lv_target_left = i;
                    lv_target_right = i+1;
                    break;
                }
            }

            //获取低曝过曝
            if (lv_target_right > lv_target_left) {
                images_raw.resize(3);
                int ev_level_over = lv_target_left, ev_level_under = lv_target_right;
                if (ev_level_over > 0 && ev_level_under < (EV_LEVEL_COUNT-1)) {
                    ev_level_over--;
                    ev_level_under++;
                }

                spdlog::info("{} ev under {}, over {}, base little lv {}\n", comName.c_str(), EV_LEVEL_ARRAY[ev_level_under], EV_LEVEL_ARRAY[ev_level_over], lv_f);

                long time = common_utils::currentTimeMilliseconds();
                setExposure(EV_LEVEL_ARRAY[ev_level_under]);
                skipFrames(SKIP_FRAMES);
                acquireFrame(&(images_raw[1]));
                spdlog::info("{} under cost time: {}\n", comName.c_str(), common_utils::currentTimeMilliseconds() - time);

                time = common_utils::currentTimeMilliseconds();
                setExposure(EV_LEVEL_ARRAY[ev_level_over]);
                skipFrames(SKIP_FRAMES);
                acquireFrame(&(images_raw[2]));
                spdlog::info("{} over cost time: {}\n", comName.c_str(), common_utils::currentTimeMilliseconds() - time);
            }

            long time = common_utils::currentTimeMilliseconds();
            setExposure(500);
            setAutoExposure(true, false);
            spdlog::info("{} set auto back cost time: {}\n", comName.c_str(), common_utils::currentTimeMilliseconds() - time);

            spdlog::info("{} shootExposureBracketing end.\n", comName.c_str());
        }

        void shootAutoExposure(cv::Mat& result, bool wait_for_frame_update = true) {
            spdlog::info("shootAutoExposure start.\n");
            setAutoExposure(true, wait_for_frame_update);
            acquireFrame();
            result = decodeBufferFrame(true, false);
            spdlog::info("shootAutoExposure end.\n");
        }

        void shootManualExposure(cv::Mat& result, int ev_lv) {//ev_lv:[0,10]
            if (ev_lv < 0) {
                ev_lv = 0;
            } else if (ev_lv > 10) {
                ev_lv = 10;
            }
            spdlog::info("shootManualExposure start, lv:{}\n", ev_lv);
            setAutoExposure(false);
            setExposure(EV_LEVEL_ARRAY[ev_lv]);
            skipFrames(SKIP_FRAMES);
            acquireFrame();
            result = decodeBufferFrame(true, false);
            spdlog::info("shootManualExposure end.\n");
        }

        Mat decodeBufferFrameHASync(bool onlyValidRange, bool rotationCorrect, size_t bytesused) {//同步解码
            Mat frame;
            if (cur_width == width) {
                if (pMppRgaDecoder != nullptr) {
                    Mat frame2;

//                    long t = common_utils::currentTimeMilliseconds();
                    if (!pMppRgaDecoder->decode(mptr[0], bytesused, frame2)) {
                        spdlog::info("mpp decode fail, replace by opencv\n");
                        return frame;
                        copyFrameData();
                        frame2 = imdecode(buf, ImreadModes::IMREAD_COLOR);
                    }
//                    spdlog::info("image decoded, cost:{}s.\n", (common_utils::currentTimeMilliseconds()-t)/1000.0f);
                    frame = onlyValidRange ? frame2(valid_row_range, valid_col_range) : frame2;
                } else {
                    copyFrameData();
                    frame = onlyValidRange ? imdecode(buf, ImreadModes::IMREAD_COLOR)(valid_row_range, valid_col_range) : imdecode(buf, ImreadModes::IMREAD_COLOR);
                }
            } else {
                copyFrameData();
                Mat yuyv(cur_height, cur_width, CV_8UC2, buf.data()), bgr;
                cvtColor(yuyv, bgr, COLOR_YUV2BGR_YUYV);
                frame = onlyValidRange ? bgr(valid_row_range, valid_col_range) : bgr;
            }

            if (rotationCorrect) {
                Mat frame2;
                correctImageRotation(frame, frame2);
                return frame2;
            } else {
                return frame;
            }
        }

//        Mat decodeBufferFrameHASync2(bool onlyValidRange, bool rotationCorrect, size_t bytesused, int target_width, int target_height) {//同步解码
//            Mat frame;
//            if (cur_width == width) {
//                if (cur_width != target_width && !useRgaResize) {
//                    useRgaResize = true;
//                    valid_row_range_resized = Range(target_height*cameraParamsMain.row_range.start/cameraParamsMain.height, target_height*cameraParamsMain.row_range.end/cameraParamsMain.height);
//                    valid_col_range_resized = Range(target_width*cameraParamsMain.col_range.start/cameraParamsMain.width, target_width*cameraParamsMain.col_range.end/cameraParamsMain.width);
//                }
//                Mat frame2;
//                long t = common_utils::currentTimeMilliseconds();
//                if (pMppRgaDecoder != nullptr && !pMppRgaDecoder->decode(mptr[0], bytesused, frame2, target_width, target_height)) {
//                    if (pMppRgaDecoder != nullptr) {
//                        spdlog::info("mpp decode fail, replace by opencv\n");
//                    }
//                    copyFrameData();
//                    if (useRgaResize) {
//                        Mat frame3 = imdecode(buf, ImreadModes::IMREAD_COLOR);
//                        cv::resize(frame3, frame2, cv::Size(target_width, target_height));
//                    } else {
//                        frame2 = imdecode(buf, ImreadModes::IMREAD_COLOR);
//                    }
//                }
//                spdlog::info("image decoded, cost:{}s.\n", (common_utils::currentTimeMilliseconds()-t)/1000.0f);
//                if (onlyValidRange) {
//                    if (useRgaResize) {
//                        frame = frame2(valid_row_range_resized, valid_col_range_resized);
//                    } else {
//                        frame = frame2(valid_row_range, valid_col_range);
//                    }
//                } else {
//                    frame = frame2;
//                }
//            } else {
//                copyFrameData();
//                Mat yuyv(cur_height, cur_width, CV_8UC2, buf.data()), bgr;
//                cvtColor(yuyv, bgr, COLOR_YUV2BGR_YUYV);
//                frame = onlyValidRange ? bgr(valid_row_range, valid_col_range) : bgr;
//            }
//
//            if (rotationCorrect) {
//                Mat frame2;
//                correctImageRotation(frame, frame2);
//                return frame2;
//            } else {
//                return frame;
//            }
//        }

        Mat decodeBufferFrameHA(bool onlyValidRange, bool rotationCorrect, std::vector<uchar>* p_buf = nullptr, size_t bytesused = 0) {
            Mat frame;
            if (cur_width == width) {
                if (pMppRgaDecoder != nullptr && p_buf != nullptr) {
                    Mat frame2;

//                    long t = common_utils::currentTimeMilliseconds();
                    if (!pMppRgaDecoder->decode(p_buf->data(), bytesused, frame2)) {
                        spdlog::info("mpp decode fail, replace by opencv\n");
                        frame2 = imdecode(p_buf == nullptr ? buf : (*p_buf), ImreadModes::IMREAD_COLOR);
                    }
//                    spdlog::info("image decoded, cost:{}s.\n", (common_utils::currentTimeMilliseconds()-t)/1000.0f);
                    frame = onlyValidRange ? frame2(valid_row_range, valid_col_range) : frame2;
                } else {
                    frame = onlyValidRange ? imdecode(p_buf == nullptr ? buf : (*p_buf), ImreadModes::IMREAD_COLOR)(valid_row_range, valid_col_range) : imdecode(p_buf == nullptr ? buf : (*p_buf), ImreadModes::IMREAD_COLOR);
                }
            } else {
                Mat yuyv(cur_height, cur_width, CV_8UC2, p_buf == nullptr ? buf.data() : p_buf->data()), bgr;
                cvtColor(yuyv, bgr, COLOR_YUV2BGR_YUYV);
                frame = onlyValidRange ? bgr(valid_row_range, valid_col_range) : bgr;
            }

            if (rotationCorrect) {
                Mat frame2;
                correctImageRotation(frame, frame2);
                return frame2;
            } else {
                return frame;
            }
        }

        inline cv::Mat shootAutoToCacheMatRealtime(std::vector<uchar>* p_buf = nullptr) {
            setAutoExposure(true, false);
            queueFrameBuffer();
            size_t bytesused = dequeueFrameBuffer();
            if (p_buf != nullptr) {
                p_buf->resize(bytesused);
                memcpy(p_buf->data(), mptr[0], bytesused);
            }
            cv::Mat mat = decodeBufferFrameHASync(true, false, bytesused);
            if (!mat.empty()) {
                std::lock_guard<std::mutex> lock(camera_mutex);
                cached_image = mat;
            }
            return mat;
        }

        inline void cachedImage(cv::Mat& mat) {
            {
                std::lock_guard<std::mutex> lock(camera_mutex);
                mat = cached_image.clone();
            }
        }

        inline void shootAutoToCacheRealtime() {
            setAutoExposure(true, false);
            queueFrameBuffer();
            bytesused_image_raw = dequeueFrameBuffer();
            timestamp_image_raw = common_utils::currentTimeMilliseconds();
            {
                std::lock_guard<std::mutex> lock(camera_mutex);
                copyFrameData(&cached_image_raw);
            }
        }

        inline bool decodeRealtimeImage(cv::Mat& image, long& timestamp) {
            std::vector<uint8_t> cache;
            size_t bytesused;
            {
                std::lock_guard<std::mutex> lock(camera_mutex);
                if(cached_image_raw.empty()){
                    return false;
                }
                cache = cached_image_raw;
                bytesused = bytesused_image_raw;
                timestamp = timestamp_image_raw;
            }
            long t = common_utils::currentTimeMilliseconds();
//            image = decodeBufferFrame(true, false, &cache);
            image = decodeBufferFrameHA(true, false, &cache, bytesused);
            spdlog::info("decodeBufferFrame end, cost:{}s.\n", (common_utils::currentTimeMilliseconds()-t)/1000.0f);
            return true;
        }

    private:
        void init() {
            spdlog::info("{} init start.\n", comName.c_str());

            //read camera params
            if (!cameraParamsPath.empty()) {
                FileStorage fs(cameraParamsPath, FileStorage::READ);
                if (!fs.isOpened()) {
                    throw std::runtime_error("Open camera params file failed!");
                }
                fs["row_range"]["start"] >> cameraParamsMain.row_range.start;
                fs["row_range"]["end"] >> cameraParamsMain.row_range.end;
                fs["col_range"]["start"] >> cameraParamsMain.col_range.start;
                fs["col_range"]["end"] >> cameraParamsMain.col_range.end;
                fs["width"] >> cameraParamsMain.width;
                fs["height"] >> cameraParamsMain.height;
                fs["centerX"] >> cameraParamsMain.centerX;
                fs["centerY"] >> cameraParamsMain.centerY;
                fs["radius"] >> cameraParamsMain.radius;
                fs["intrinsic_matrix"] >> cameraParamsMain.k;
                fs["distortion_coeffs"] >> cameraParamsMain.d;
                fs.release();

                //根据实际应用的像素调整内参，需要与标定时同比例
                if (std::abs(1.0*width/height - 1.0*cameraParamsMain.width/cameraParamsMain.height) < 0.001) {
                    double scale = 1.0*width/cameraParamsMain.width;
                    cameraParamsMain.k(0, 0) *= scale;//fx
                    cameraParamsMain.k(1, 1) *= scale;//fy
                    cameraParamsMain.k(0, 2) *= scale;//cx
                    cameraParamsMain.k(1, 2) *= scale;//cy
                    cameraParamsMain.width *= scale;
                    cameraParamsMain.height *= scale;
                    cameraParamsMain.centerX *= scale;
                    cameraParamsMain.centerY *= scale;
                    cameraParamsMain.radius *= scale;
                    cameraParamsMain.row_range.start *= scale;
                    cameraParamsMain.row_range.end *= scale;
                    cameraParamsMain.col_range.start *= scale;
                    cameraParamsMain.col_range.end *= scale;
                    std::cout << "adjusted intrinsic scale:" << scale << std::endl << "row_range=[" << cameraParamsMain.row_range.start << "," << cameraParamsMain.row_range.end << "], col_range=["
                    << cameraParamsMain.col_range.start << "," << cameraParamsMain.col_range.end << "], width=" << cameraParamsMain.width << ", height=" << cameraParamsMain.height << ", centerX="
                    << cameraParamsMain.centerX << ", centerY=" << cameraParamsMain.centerY << ", radius=" << cameraParamsMain.radius << ", k=" << cameraParamsMain.k << ", d=" << cameraParamsMain.d << std::endl;
                } else {
                    throw std::runtime_error("Camera resolution is not proportional to the intrinsic!");
                }

                double ratio = 1.0*width_little/width;
                cameraParamsLittle.row_range.start = ratio*cameraParamsMain.row_range.start;
                cameraParamsLittle.row_range.end = ratio*cameraParamsMain.row_range.end;
                cameraParamsLittle.col_range.start = ratio*cameraParamsMain.col_range.start;
                cameraParamsLittle.col_range.end = ratio*cameraParamsMain.col_range.end;
                cameraParamsLittle.width = ratio*cameraParamsMain.width;
                cameraParamsLittle.height = ratio*cameraParamsMain.height;
                cameraParamsLittle.centerX = ratio*cameraParamsMain.centerX;
                cameraParamsLittle.centerY = ratio*cameraParamsMain.centerY;
                cameraParamsLittle.radius = ratio*cameraParamsMain.radius;
                cameraParamsLittle.k = cameraParamsMain.k;
                cameraParamsLittle.d = cameraParamsMain.d;

            } else {
                throw std::runtime_error("Camera params main filepath is empty!");
            }

            fd = open(comName.c_str(),O_RDWR);
            if (fd < 0)
            {
                throw std::runtime_error("打开设备失败");
            }

            requestBuffer.type = type;
            requestBuffer.memory = V4L2_MEMORY_MMAP;

            frameBuffer.type = type;
            frameBuffer.memory = V4L2_MEMORY_MMAP;

            setSolutionInternal(true);
            requestKernelSpace();

            //sleep(2);
            spdlog::info("{} init end.\n", comName.c_str());
        }

        void setSolutionInternal(bool main) {
            CameraParams cameraParams = main ? cameraParamsMain : cameraParamsLittle;
            cur_width = main ? width : width_little;
            cur_height = main ? height : height_little;
            //获取摄像头支持格式 ioctl(文件描述符,命令，与命令对应的结构体)
            fmt.type = type; //摄像头采集
            fmt.fmt.pix.width = cur_width; //设置摄像头采集参数，不可以任意设置
            fmt.fmt.pix.height = cur_height;
            fmt.fmt.pix.pixelformat = main ? V4L2_PIX_FMT_MJPEG : V4L2_PIX_FMT_YUYV; //设置为mjpg格式，则我可以直接写入文件保存，YUYV格式保存后需要转换格式才能查看
            if (ioctl(fd,VIDIOC_S_FMT,&fmt) < 0) {
                throw std::runtime_error("设置格式失败:"+comName);
            } else {
                buf.resize(fmt.fmt.pix.sizeimage);
            }

            {
                valid_row_range = Range(cameraParams.row_range.start, cameraParams.row_range.end);
                valid_col_range = Range(cameraParams.col_range.start, cameraParams.col_range.end);
                std::cout << comName << " valid_row_range:" << valid_row_range << std::endl;
                std::cout << comName << " valid_col_range:" << valid_col_range << std::endl;

                //生成有效图像范围掩码
                valid_range_mask = Mat(cur_height, cur_width, CV_8UC1, cv::Scalar(0));
                cv::Point center(cameraParams.centerX, cameraParams.centerY);
                cv::circle(valid_range_mask, center, cameraParams.radius, cv::Scalar(255), cv::FILLED);
                std::cout << comName << " point:" << center << ", radius:" << std::to_string(cameraParams.radius) << std::endl;
            }
        }

        void requestKernelSpace() {
            if (kernelSpaceRequested) {
                return;
            }
            kernelSpaceRequested = true;
            //申请内核空间
            requestBuffer.count = 1;

            if (ioctl(fd,VIDIOC_REQBUFS,&requestBuffer) < 0) {
                throw std::runtime_error("申请空间失败");
            }

            for(int i = 0; i <1;i++) {
                frameBuffer.index = i;
                if (ioctl(fd,VIDIOC_QUERYBUF,&frameBuffer) < 0) {//从内核空间中查询一个空间作映射
                    throw std::runtime_error("查询内核空间失败");
                }
                //映射到用户空间
                mptr[i] = (unsigned char *)mmap(NULL,frameBuffer.length,PROT_READ|PROT_WRITE,MAP_SHARED, fd,frameBuffer.m.offset);
                size[i] = frameBuffer.length; //保存映射长度用于后期释放
                /*
                //查询后通知内核已经放回
                if (ioctl(fd,VIDIOC_QBUF, &frameBuffer) < 0) {
                    throw std::runtime_error("放回失败");
                }*/
                //初始化硬件解码和颜色空间转换器
                if (pMppRgaDecoder != nullptr) {
                    delete pMppRgaDecoder;
                    pMppRgaDecoder = nullptr;
                }
                pMppRgaDecoder = new MppRgaDecoder(width, height);
            }
        }

        void releaseKernelSpace() {
            if (!kernelSpaceRequested) {
                return;
            }
            kernelSpaceRequested = false;
            if (fd >= 0) {
                //释放映射
                for(int i=0; i<1; i++) {
                    munmap(mptr[i], size[i]);
                }

                requestBuffer.count = 0;
                if (ioctl(fd,VIDIOC_REQBUFS, &requestBuffer) < 0) {
                    throw std::runtime_error("注销空间失败");
                }
            }
        }

        void closeFd() {
            if (fd >= 0) {
                close(fd); //关闭文件
                fd = -1;
            }
        }

        void initShootingParams() {
            spdlog::info("{} initShootingParams start.\n", comName.c_str());
            ctrl.id = V4L2_CID_AUTO_WHITE_BALANCE;
            ctrl.value = V4L2_WHITE_BALANCE_AUTO;
            if (ioctl(fd, VIDIOC_S_CTRL, &ctrl) < 0) {
                throw std::runtime_error("设置自动白平衡失败");
            }

            ctrl.id = V4L2_CID_EXPOSURE_AUTO;
            ctrl.value = V4L2_EXPOSURE_APERTURE_PRIORITY;
            if (ioctl(fd, VIDIOC_S_CTRL, &ctrl) < 0) {
                throw std::runtime_error("设置自动曝光失败");
            }
            cur_is_auto = true;
            spdlog::info("{} initShootingParams end.\n", comName.c_str());
        }

    public:
        int v4l2SetControl(int control, int value) {
            ctrl.id = control;
            ctrl.value = value;
            if (ioctl(fd, VIDIOC_S_CTRL, &ctrl) < 0) {
                spdlog::info("ioctl set control error\n");
                return -1;
            }
        }

        int v4l2QueryControl(int control, struct v4l2_queryctrl *queryctrl)
        {
            int err =0;
            queryctrl->id = control;
            if ((err= ioctl(fd, VIDIOC_QUERYCTRL, queryctrl)) < 0) {
                spdlog::info("ioctl querycontrol error {},{} \n",errno,control);
            } else if (queryctrl->flags & V4L2_CTRL_FLAG_DISABLED) {
                spdlog::info("control {} disabled \n", (char *) queryctrl->name);
            } else if (queryctrl->flags & V4L2_CTRL_TYPE_BOOLEAN) {
                return 1;
            } else if (queryctrl->type & V4L2_CTRL_TYPE_INTEGER) {
                return 0;
            } else {
                spdlog::info("contol {} unsupported  \n", (char *) queryctrl->name);
            }
            return -1;
        }

        int v4l2GetControl(int control)
        {
            struct v4l2_queryctrl queryctrl;
            struct v4l2_control control_s;
            int err;
            if (v4l2QueryControl(control, &queryctrl) < 0)
                return -1;
            control_s.id = control;
            if ((err = ioctl(fd, VIDIOC_G_CTRL, &control_s)) < 0) {
                spdlog::info("ioctl get control error\n");
                return -1;
            }
            return control_s.value;
        }
    };

    class Camera {

    private:
        CameraInternal* cameraInternal1 = nullptr;
        CameraInternal* cameraInternal2 = nullptr;
        std::vector<Mat> cached_images[2];
        std::vector<std::vector<uchar>> cached_images_raw[2];
        std::vector<std::vector<std::vector<uchar>>> cached_images_raw_multi_angles[2];
    public:
        Camera(std::string camera_configs_filepath) {

            FileStorage fs(camera_configs_filepath, FileStorage::READ);
            if (!fs.isOpened()) {
                throw std::runtime_error("Open camera configs file failed!");
            }
            std::string left_camera_path, left_camera_params_path, right_camera_path, right_camera_params_path;
            int width, height, width_metering, height_metering;

            fs["left_camera_path"] >> left_camera_path;
            fs["left_camera_params_path"] >> left_camera_params_path;
            fs["right_camera_path"] >> right_camera_path;
            fs["right_camera_params_path"] >> right_camera_params_path;
            fs["width"] >> width;
            fs["height"] >> height;
            fs["width_metering"] >> width_metering;
            fs["height_metering"] >> height_metering;

            fs.release();

            cameraInternal1 = new CameraInternal(left_camera_path, width, height, width_metering, height_metering, left_camera_params_path);
            cameraInternal2 = new CameraInternal(right_camera_path, width, height, width_metering, height_metering, right_camera_params_path);

            spdlog::info("streamOn start.\n");
            cameraInternal1->streamOn();
            cameraInternal2->streamOn();
            spdlog::info("streamOn end.\n");
        }
        ~Camera() {
            cameraInternal1->streamOff();
            cameraInternal2->streamOff();
            delete cameraInternal1;
            delete cameraInternal2;
        }

        void shootManualImages(Mat& fisheye1, Mat& fisheye2, int ev_lv) {
            spdlog::info("Start shooting auto images.\n");
            omp_set_num_threads(2);
            CameraInternal* cameraInternals[2];
            cameraInternals[0] = cameraInternal1;
            cameraInternals[1] = cameraInternal2;
            Mat* mats[2];
            mats[0] = std::addressof(fisheye1);
            mats[1] = std::addressof(fisheye2);
#pragma omp parallel for default(none) shared(cameraInternals, mats)
            for (int i=0; i<2; i++) {
                cameraInternals[i]->shootManualExposure(*mats[i], ev_lv);
            }
            spdlog::info("Shooting auto images finished.\n");
        }

        void shootAutoImages(Mat& fisheye1, Mat& fisheye2) {
            spdlog::info("Start shooting auto images.\n");
            omp_set_num_threads(2);
            CameraInternal* cameraInternals[2];
            cameraInternals[0] = cameraInternal1;
            cameraInternals[1] = cameraInternal2;
            Mat* mats[2];
            mats[0] = std::addressof(fisheye1);
            mats[1] = std::addressof(fisheye2);
#pragma omp parallel for default(none) shared(cameraInternals, mats)
            for (int i=0; i<2; i++) {
                cameraInternals[i]->shootAutoExposure(*mats[i]);
            }
            spdlog::info("Shooting auto images finished.\n");
        }

        void shootSrcImagesToCache() {
            spdlog::info("Start shooting cache images.\n");
            omp_set_num_threads(2);
            CameraInternal* cameraInternals[2];
            cameraInternals[0] = cameraInternal1;
            cameraInternals[1] = cameraInternal2;
#pragma omp parallel for default(none) shared(cameraInternals, cached_images_raw)
            for (int i=0; i<2; i++) {
                cached_images_raw[i].clear();
                cameraInternals[i]->shootExposureBracketing(cached_images_raw[i]);
            }
            spdlog::info("Shooting src finished, wait for merge to panorama.\n");
        }

        void shootSrcImagesToCacheMultiAngles() {
            spdlog::info("Start shooting cache images.\n");
            omp_set_num_threads(2);
            CameraInternal* cameraInternals[2];
            cameraInternals[0] = cameraInternal1;
            cameraInternals[1] = cameraInternal2;
#pragma omp parallel for default(none) shared(cameraInternals, cached_images_raw_multi_angles)
            for (int i=0; i<2; i++) {
                size_t count = cached_images_raw_multi_angles[i].size();
                cached_images_raw_multi_angles[i].resize(count+1);
                cameraInternals[i]->shootExposureBracketing(cached_images_raw_multi_angles[i][count]);
            }
            spdlog::info("Shooting src finished, wait for merge to panorama.\n");
        }

        void shootSrcImagesToLocal(std::string cache_dir_path) {
            spdlog::info("Start shooting cache images.\n");
            omp_set_num_threads(2);
            CameraInternal* cameraInternals[2];
            cameraInternals[0] = cameraInternal1;
            cameraInternals[1] = cameraInternal2;
#pragma omp parallel for default(none) shared(cameraInternals, cached_images_raw, cache_dir_path)
            for (int i=0; i<2; i++) {
                cached_images_raw[i].clear();
                cameraInternals[i]->shootExposureBracketing(cached_images_raw[i]);
                for (int j=0; j<cached_images_raw[i].size(); j++) {
                    std::string cache_filepath(cache_dir_path + "/" + to_string(i) + "_" + to_string(j) + ".dat");
                    std::ofstream outFile(cache_filepath, std::ios::out | std::ios::binary);
                    if (!outFile) {
                        throw std::runtime_error("Cannot open image cache file: " + cache_filepath);
                    }
                    outFile.write(reinterpret_cast<const char*>(cached_images_raw[i][j].data()), cached_images_raw[i][j].size());
                    outFile.close();
                }
            }
            spdlog::info("Shooting src finished, wait for merge to panorama.\n");
        }

        std::vector<Mat>& cachedImages(int index) {
            return cached_images[index];
        }

        //先曝光融合再合成全景图
        void mergeCachedImages(Mat& img, Mat& fisheye1, Mat& fisheye2, float contrast_weight = 1.0f, float saturation_weight = 1.0f, float exposure_weight = 0.0f, const Eigen::Matrix4d& correct = Eigen::Matrix4d::Identity()) {
            if (cached_images_raw[0].empty()) {
                return;
            }
            omp_set_num_threads(2);
            CameraInternal* cameraInternals[2];
            cameraInternals[0] = cameraInternal1;
            cameraInternals[1] = cameraInternal2;
            Mat fusions[2], fusions_median_blur[2];
#pragma omp parallel for default(none) shared(cameraInternals, cached_images, cached_images_raw, fusions, fusions_median_blur, contrast_weight, saturation_weight, exposure_weight)
            for (int i=0; i<2; i++) {
                cached_images[i].clear();
                for (int j=0; j<cached_images_raw[i].size(); j++) {
                    cached_images[i].emplace_back(cameraInternals[i]->decodeBufferFrame(true, false, &(cached_images_raw[i][j])));
                }
                if (cached_images[i].size() > 1) {
                    Mat mat[2];
                    Ptr<MergeMertens> mergeMertens = createMergeMertens(contrast_weight, saturation_weight, exposure_weight);
                    mergeMertens->process(cached_images[i], mat[i]);
                    //fusions[i] = mat[i] * 255;
                    mat[i].convertTo(fusions[i], CV_8UC3, 255.0);
                    //中值滤波
                    medianBlur(fusions[i], fusions_median_blur[i], 3);
                } else {
                    fusions_median_blur[i] = cached_images[i][0];
                }
            }
            spdlog::info("Exposure fusion finished, start merge to panorama.\n");

            Mat sharpenKernel = (Mat_<float>(3, 3) <<
                                                   0, -1, 0,
                    -1, 5, -1,
                    0, -1, 0);

            Mat tmp1, tmp2;

            GaussianBlur(fusions_median_blur[0], tmp1, Size(3, 3), 1.5);
            GaussianBlur(fusions_median_blur[1], tmp2, Size(3, 3), 1.5);
            filter2D(tmp1, fisheye1, fusions_median_blur[0].depth(), sharpenKernel);
            filter2D(tmp2, fisheye2, fusions_median_blur[1].depth(), sharpenKernel);

            merge_to_panorama_with_correct(fisheye1, fisheye2, img, true, correct);

            spdlog::info("Merge to panorama done.\n");
            //fisheye1 = fusions_median_blur[0];
            //fisheye2 = fusions_median_blur[1];
            spdlog::info("Cached images processing finished.\n");
        }

        void mergeCachedImagesMultiAngles(Mat& panorama, std::vector<Mat>& fisheye1, std::vector<Mat>& fisheye2, size_t panorama_composite_index = 0,
                                          float contrast_weight = 1.0f, float saturation_weight = 1.0f, float exposure_weight = 0.0f, const Eigen::Matrix4d& correct = Eigen::Matrix4d::Identity()) {
            if (cached_images_raw_multi_angles[0].empty() || cached_images_raw_multi_angles[0][0].empty()) {
                return;
            }
            if (cached_images_raw_multi_angles[0].size()-1 < panorama_composite_index) {
                throw std::runtime_error("The panorama_composite_index is out of range.");
            }
            omp_set_num_threads(2);
            CameraInternal* cameraInternals[2];
            cameraInternals[0] = cameraInternal1;
            cameraInternals[1] = cameraInternal2;
            std::vector<Mat> fusions[2], fusions_median_blur[2];
#pragma omp parallel for default(none) shared(cameraInternals, cached_images, cached_images_raw_multi_angles, fusions, fusions_median_blur, contrast_weight, saturation_weight, exposure_weight)
            for (int i=0; i<2; i++) {
                fusions[i].resize(cached_images_raw_multi_angles[i].size());
                fusions_median_blur[i].resize(cached_images_raw_multi_angles[i].size());
                for (int angle=0; angle<cached_images_raw_multi_angles[i].size(); angle++) {
                    cached_images[i].clear();
                    for (int j=0; j<cached_images_raw_multi_angles[i][angle].size(); j++) {
                        cached_images[i].emplace_back(cameraInternals[i]->decodeBufferFrame(true, false, &(cached_images_raw_multi_angles[i][angle][j])));
                    }
                    if (cached_images[i].size() > 1) {
                        Mat mat[2];
                        Ptr<MergeMertens> mergeMertens = createMergeMertens(contrast_weight, saturation_weight, exposure_weight);
                        mergeMertens->process(cached_images[i], mat[i]);
                        //fusions[i] = mat[i] * 255;
                        mat[i].convertTo(fusions[i][angle], CV_8UC3, 255.0);
                        //中值滤波
                        medianBlur(fusions[i][angle], fusions_median_blur[i][angle], 3);
                    } else {
                        fusions_median_blur[i][angle] = cached_images[i][0];
                    }
                }
            }
            spdlog::info("Exposure fusion finished, start merge to panorama.\n");

            Mat sharpenKernel = (Mat_<float>(3, 3) <<
                                                   0, -1, 0,
                    -1, 5, -1,
                    0, -1, 0);

            size_t angle_count = cached_images_raw_multi_angles[0].size();

            fisheye1.clear();
            fisheye2.clear();
            fisheye1.resize(angle_count);
            fisheye2.resize(angle_count);
            for (int angle=0; angle<angle_count; angle++) {
                Mat tmp1, tmp2;

                GaussianBlur(fusions_median_blur[0][angle], tmp1, Size(3, 3), 1.5);
                GaussianBlur(fusions_median_blur[1][angle], tmp2, Size(3, 3), 1.5);
                filter2D(tmp1, fisheye1[angle], fusions_median_blur[0][angle].depth(), sharpenKernel);
                filter2D(tmp2, fisheye2[angle], fusions_median_blur[1][angle].depth(), sharpenKernel);

                if (angle == panorama_composite_index) {
                    merge_to_panorama_with_correct(fisheye1[angle], fisheye2[angle], panorama, true, correct);
                }
            }

            cached_images_raw_multi_angles[0].clear();
            cached_images_raw_multi_angles[1].clear();

            spdlog::info("Merge to panorama done.\n");
            spdlog::info("Cached images processing finished.\n");
        }

        void mergeCachedImages(Mat& fisheye1, Mat& fisheye2, float contrast_weight = 1.0f, float saturation_weight = 1.0f, float exposure_weight = 0.0f, const Eigen::Matrix4d& correct = Eigen::Matrix4d::Identity()) {
            if (cached_images_raw[0].empty()) {
                return;
            }
            omp_set_num_threads(2);
            CameraInternal* cameraInternals[2];
            cameraInternals[0] = cameraInternal1;
            cameraInternals[1] = cameraInternal2;
            Mat fusions[2], fusions_median_blur[2];
#pragma omp parallel for default(none) shared(cameraInternals, cached_images, cached_images_raw, fusions, fusions_median_blur, contrast_weight, saturation_weight, exposure_weight)
            for (int i=0; i<2; i++) {
                cached_images[i].clear();
                for (int j=0; j<cached_images_raw[i].size(); j++) {
                    cached_images[i].emplace_back(cameraInternals[i]->decodeBufferFrame(true, false, &(cached_images_raw[i][j])));
                }
                if (cached_images[i].size() > 1) {
                    Mat mat[2];
                    Ptr<MergeMertens> mergeMertens = createMergeMertens(contrast_weight, saturation_weight, exposure_weight);
                    mergeMertens->process(cached_images[i], mat[i]);
                    //fusions[i] = mat[i] * 255;
                    mat[i].convertTo(fusions[i], CV_8UC3, 255.0);
                    //中值滤波
                    medianBlur(fusions[i], fusions_median_blur[i], 3);
                } else {
                    fusions_median_blur[i] = cached_images[i][0];
                }
            }
            spdlog::info("Exposure fusion finished, start merge to panorama.\n");

            Mat sharpenKernel = (Mat_<float>(3, 3) <<
                                                   0, -1, 0,
                    -1, 5, -1,
                    0, -1, 0);

            Mat tmp1, tmp2;

            GaussianBlur(fusions_median_blur[0], tmp1, Size(3, 3), 1.5);
            GaussianBlur(fusions_median_blur[1], tmp2, Size(3, 3), 1.5);
            filter2D(tmp1, fisheye1, fusions_median_blur[0].depth(), sharpenKernel);
            filter2D(tmp2, fisheye2, fusions_median_blur[1].depth(), sharpenKernel);
            spdlog::info("Cached images processing finished.\n");
        }

        void mergeCachedImagesInLocal(std::string cache_dir_path, Mat& img, Mat& fisheye1, Mat& fisheye2, float contrast_weight = 1.0f, float saturation_weight = 1.0f, float exposure_weight = 0.0f, const Eigen::Matrix4d& correct = Eigen::Matrix4d::Identity()) {
            omp_set_num_threads(2);
            CameraInternal* cameraInternals[2];
            cameraInternals[0] = cameraInternal1;
            cameraInternals[1] = cameraInternal2;
#pragma omp parallel for default(none) shared(cameraInternals, cached_images_raw, cache_dir_path)
            for (int i=0; i<2; i++) {
                cached_images_raw[i].clear();
                cached_images_raw[i].resize(3);

                for (int j=0; j<3; j++) {
                    std::string cache_filepath(cache_dir_path + "/" + to_string(i) + "_" + to_string(j) + ".dat");
                    std::ifstream inFile(cache_filepath, std::ios::in | std::ios::binary);
                    if (!inFile) {
                        cached_images_raw[i].resize(j);
                        break;
                    }
                    // 确定文件大小
                    inFile.seekg(0, std::ios::end);
                    std::streampos fileSize = inFile.tellg();
                    inFile.seekg(0, std::ios::beg);
                    cached_images_raw[i][j].resize(fileSize);

                    inFile.read(reinterpret_cast<char*>(cached_images_raw[i][j].data()), fileSize);
                    inFile.close();
                }
            }
            mergeCachedImages(img, fisheye1, fisheye2, contrast_weight, saturation_weight, exposure_weight, correct);
        }

        void mergeCachedImagesInLocal(std::string cache_dir_path, Mat& fisheye1, Mat& fisheye2, float contrast_weight = 1.0f, float saturation_weight = 1.0f, float exposure_weight = 0.0f, const Eigen::Matrix4d& correct = Eigen::Matrix4d::Identity()) {
            omp_set_num_threads(2);
            CameraInternal* cameraInternals[2];
            cameraInternals[0] = cameraInternal1;
            cameraInternals[1] = cameraInternal2;
#pragma omp parallel for default(none) shared(cameraInternals, cached_images_raw, cache_dir_path)
            for (int i=0; i<2; i++) {
                cached_images_raw[i].clear();
                cached_images_raw[i].resize(3);

                for (int j=0; j<3; j++) {
                    std::string cache_filepath(cache_dir_path + "/" + to_string(i) + "_" + to_string(j) + ".dat");
                    std::ifstream inFile(cache_filepath, std::ios::in | std::ios::binary);
                    if (!inFile) {
                        cached_images_raw[i].resize(j);
                        break;
                    }
                    // 确定文件大小
                    inFile.seekg(0, std::ios::end);
                    std::streampos fileSize = inFile.tellg();
                    inFile.seekg(0, std::ios::beg);
                    cached_images_raw[i][j].resize(fileSize);

                    inFile.read(reinterpret_cast<char*>(cached_images_raw[i][j].data()), fileSize);
                    inFile.close();
                }
            }
            mergeCachedImages(fisheye1, fisheye2, contrast_weight, saturation_weight, exposure_weight, correct);
        }

        inline bool projectPointsLeft(std::vector<Point3f>& objPoints, std::vector<Point2f>& imgPoints) {
            return cameraInternal1->projectPoints(objPoints, imgPoints);
        }

        inline bool projectPointsRight(std::vector<Point3f>& objPoints, std::vector<Point2f>& imgPoints) {
            return cameraInternal2->projectPoints(objPoints, imgPoints);
        }

        inline std::array<double,4> undistortLeft(cv::Mat& distorted_img, cv::Mat& undistorted_img, cv::Mat& r) {//return intrinsic, order:fx,fy,cx,cy
            return cameraInternal1->undistort(distorted_img, undistorted_img, r);
        }

        inline std::array<double,4> undistortRight(cv::Mat& distorted_img, cv::Mat& undistorted_img, cv::Mat& r) {//return intrinsic, order:fx,fy,cx,cy
            return cameraInternal2->undistort(distorted_img, undistorted_img, r);
        }

        void merge_to_panorama(Mat& left, Mat& right, Mat& target, bool left2right) {
            using namespace std;
            int height = 4096;//min(left.rows, right.rows);//5376*2688
            int width = height*2;

            vector<Point3f> objPoints;
            vector<Point2f> imgPointsLeft, imgPointsRight;
            vector<Point2i> targetPoints;

            double r=1.0, h, v;

            for (int i=0; i<height; i++) {//cols
                for (int j=0; j<height; j++) {//rows
                    h=i*180.0/height-90, v=j*180.0/height-90;
                    double r_xz = r*cos(v*M_PI/180.0);
                    double x = r_xz*sin(h*M_PI/180.0);
                    double y = r*sin(v*M_PI/180.0);
                    double z = r_xz*cos(h*M_PI/180.0);

                    objPoints.emplace_back(Point3f(x, y, z));
                    targetPoints.emplace_back(i,j);
                }
            }

            if (left2right) {
                cameraInternal1->projectPoints(objPoints, imgPointsLeft);
                cameraInternal2->projectPoints(objPoints, imgPointsRight);
            } else {
                cameraInternal2->projectPoints(objPoints, imgPointsLeft);
                cameraInternal1->projectPoints(objPoints, imgPointsRight);
            }

            target = Mat(height, width, left.type());
#define INTERPOLATION
#ifdef INTERPOLATION
            if (left.type() == CV_8UC3) {
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3b>(point.y, point.x+height) = interpolate<Vec3b>(left, imgPointsLeft[i], Vec3b(0, 0, 0));//left.at<Vec3b>(imgPointsLeft[i]);
                }
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3b>(point) = interpolate<Vec3b>(right, imgPointsRight[i], Vec3b(0, 0, 0));//right.at<Vec3b>(imgPointsLeft[i]);
                }
            } else if (left.type() == CV_32FC3) {
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3f>(point.y, point.x+height) = interpolate<Vec3f>(left, imgPointsLeft[i], Vec3b(0, 0, 0));//left.at<Vec3f>(imgPointsLeft[i]);
                }
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3f>(point) = interpolate<Vec3f>(right, imgPointsRight[i], Vec3b(0, 0, 0));//right.at<Vec3f>(imgPointsLeft[i]);
                }
            }
#else
            if (left.type() == CV_8UC3) {
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3b>(point.y, point.x+height) = left.at<Vec3b>(imgPointsLeft[i]);
                }
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3b>(point) = right.at<Vec3b>(imgPointsLeft[i]);
                }
            } else if (left.type() == CV_32FC3) {
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3f>(point.y, point.x+height) = left.at<Vec3f>(imgPointsLeft[i]);
                }
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3f>(point) = right.at<Vec3f>(imgPointsLeft[i]);
                }
            }
#endif
        }

        void merge_to_panorama_with_correct(Mat& left, Mat& right, Mat& target, bool left2right, const Eigen::Matrix4d& correct) {
            using namespace std;
            int height = 4096;//min(left.rows, right.rows);//5376*2688
            int width = height*2;

            vector<Point3f> objPoints;
            vector<Point2f> imgPointsLeft, imgPointsRight;
            vector<Point2i> targetPoints;

            double r=1.0, h, v;

            for (int i=0; i<height; i++) {//cols
                for (int j=0; j<height; j++) {//rows
                    h=i*180.0/height-90.0, v=j*180.0/height-90.0;
                    double r_xz = r*cos(v*M_PI/180.0);
                    double x = r_xz*sin(h*M_PI/180.0);
                    double y = r*sin(v*M_PI/180.0);
                    double z = r_xz*cos(h*M_PI/180.0);

                    Eigen::Vector4d np = correct*Eigen::Vector4d(x, y, z, 1);
                    x = np.x();
                    y = np.y();
                    z = np.z();
                    r = pow(x*x+y*y+z*z, 0.5), h = atan(x/z)*180.0/CV_PI, v = asin(y/r)*180.0/CV_PI;
                    if (h < -90.0 || h > 90.0 || v < -90.0 || v > 90.0) {
                        continue;
                    }

                    objPoints.emplace_back(Point3f(x, y, z));
                    targetPoints.emplace_back(i,j);
                }
            }

            if (left2right) {
                cameraInternal1->projectPoints(objPoints, imgPointsLeft);
                cameraInternal2->projectPoints(objPoints, imgPointsRight);
            } else {
                cameraInternal2->projectPoints(objPoints, imgPointsLeft);
                cameraInternal1->projectPoints(objPoints, imgPointsRight);
            }

            target = Mat(height, width, left.type());
#define INTERPOLATION
#ifdef INTERPOLATION
            if (left.type() == CV_8UC3) {
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3b>(point.y, point.x+height) = interpolate<Vec3b>(left, imgPointsLeft[i], Vec3b(0, 0, 0));//left.at<Vec3b>(imgPointsLeft[i]);
                }
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3b>(point) = interpolate<Vec3b>(right, imgPointsRight[i], Vec3b(0, 0, 0));//right.at<Vec3b>(imgPointsLeft[i]);
                }
            } else if (left.type() == CV_32FC3) {
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3f>(point.y, point.x+height) = interpolate<Vec3f>(left, imgPointsLeft[i], Vec3b(0, 0, 0));//left.at<Vec3f>(imgPointsLeft[i]);
                }
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3f>(point) = interpolate<Vec3f>(right, imgPointsRight[i], Vec3b(0, 0, 0));//right.at<Vec3f>(imgPointsLeft[i]);
                }
            }
#else
            if (left.type() == CV_8UC3) {
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3b>(point.y, point.x+height) = left.at<Vec3b>(imgPointsLeft[i]);
                }
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3b>(point) = right.at<Vec3b>(imgPointsLeft[i]);
                }
            } else if (left.type() == CV_32FC3) {
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3f>(point.y, point.x+height) = left.at<Vec3f>(imgPointsLeft[i]);
                }
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3f>(point) = right.at<Vec3f>(imgPointsLeft[i]);
                }
            }
#endif
        }

        void shootIntrinsicMarkImages(Mat& left, Mat& right) {
            shootAutoImages(left, right);
            vector<Point3f> objPoints, objPoints2, objPointsX, objPointsY, objPointsZ;
            double r=1.0, h, v, step=0.5;
            h=-92.5, v=0.0;
            while (h<=92.5) {
                double r_xz = r*cos(v*M_PI/180.0);
                double x = r_xz*sin(h*M_PI/180.0);
                double y = r*sin(v*M_PI/180.0);
                double z = r_xz*cos(h*M_PI/180.0);
                objPoints.emplace_back(Point3f(x, y, z));
                h+=step;
            }

            h=0, v=-92.5;
            while (v<=92.5) {
                double r_xz = r*cos(v*M_PI/180.0);
                double x = r_xz*sin(h*M_PI/180.0);
                double y = r*sin(v*M_PI/180.0);
                double z = r_xz*cos(h*M_PI/180.0);
                objPoints.emplace_back(Point3f(x, y, z));
                v+=step;
            }

            h=45, v=-92.5;
            while (v<=92.5) {
                double r_xz = r*cos(v*M_PI/180.0);
                double x = r_xz*sin(h*M_PI/180.0);
                double y = r*sin(v*M_PI/180.0);
                double z = r_xz*cos(h*M_PI/180.0);
                objPoints.emplace_back(Point3f(x, y, z));
                v+=step;
            }

            double incidence = 90, surround = 0.0;
            while (surround < 360) {
                double r_xy = r*sin(incidence*M_PI/180.0);
                double x = r_xy*cos(surround*M_PI/180.0);
                double y = r_xy*sin(surround*M_PI/180.0);
                double z = r*cos(incidence*M_PI/180.0);
                objPoints2.emplace_back(Point3f(x, y, z));
                surround+=step;
            }

            float value = 0;
            while (value < 20) {
                objPointsX.emplace_back(Point3f(value, 0, 0));
                value+=0.2f;
            }
            value = 0;
            while (value < 20) {
                objPointsY.emplace_back(Point3f(0, value, 0));
                value+=0.2f;
            }
            value = 0;
            while (value < 20) {
                objPointsZ.emplace_back(Point3f(0, 0, value));
                value+=0.2f;
            }

            for (int i=0; i<2; i++) {
                CameraInternal* cameraInternal = i==0 ? cameraInternal1 : cameraInternal2;
                Mat& img = i==0 ? left : right;

                vector<Point2f> imgPoints, imgPoints2, imgPointsX, imgPointsY, imgPointsZ;
                cameraInternal->projectPoints(objPoints, imgPoints);
                cameraInternal->projectPoints(objPoints2, imgPoints2);
                cameraInternal->projectPoints(objPointsX, imgPointsX);
                cameraInternal->projectPoints(objPointsY, imgPointsY);
                cameraInternal->projectPoints(objPointsZ, imgPointsZ);
                for (auto& p : imgPoints) {
                    if (p.x < 0 || p.y < 0) {
                        continue;
                    }
                    fillAround(img, p.x, p.y, Vec3b(255, 0, 255));
                }
                for (auto& p : imgPoints2) {
                    if (p.x < 0 || p.y < 0) {
                        continue;
                    }
                    fillAround(img, p.x, p.y, Vec3b(255, 255, 0));
                }

                for (auto& p : imgPointsX) {
                    if (p.x < 0 || p.y < 0) {
                        continue;
                    }
                    fillAround(img, p.x, p.y, Vec3b(0, 0, 255));
                }
                for (auto& p : imgPointsY) {
                    if (p.x < 0 || p.y < 0) {
                        continue;
                    }
                    fillAround(img, p.x, p.y, Vec3b(0, 255, 0));
                }
                for (auto& p : imgPointsZ) {
                    if (p.x < 0 || p.y < 0) {
                        continue;
                    }
                    fillAround(img, p.x, p.y, Vec3b(255, 0, 0));
                }
            }
        }
    private:
        template <typename T>
        inline T interpolate(Mat& mat, Point2f& point, T default_value) {
#if 1
            int x_floor = static_cast<int>(std::floor(point.x));
            int y_floor = static_cast<int>(std::floor(point.y));

            if (x_floor < 0 || x_floor >= mat.cols - 1 || y_floor < 0 || y_floor >= mat.rows - 1) {
                return default_value;
            }

            auto& a = mat.at<T>(y_floor+1, x_floor);
            auto& b = mat.at<T>(y_floor, x_floor);
            auto& c = mat.at<T>(y_floor, x_floor+1);
            auto& d = mat.at<T>(y_floor+1, x_floor+1);

            float x_rate = point.x - x_floor;
            float y_rate = point.y - y_floor;

            T e = a + (d - a) * x_rate;
            T f = b + (c - b) * x_rate;

            return f + (e - f) * y_rate;
#else
            int x = static_cast<int>(point.x);
            int y = static_cast<int>(point.y);

            if (x < 0 || y < 0 || x >= mat.cols - 1 || y >= mat.rows - 1) {
                return default_value;
            }

            float u = point.x - x;
            float v = point.y - y;

            float inv_u = 1.0f - u;
            float inv_v = 1.0f - v;

            const T* row0 = mat.ptr<T>(y);
            const T* row1 = mat.ptr<T>(y + 1);

            using InternalType = cv::Vec3f;

            InternalType p00 = static_cast<InternalType>(row0[x]);
            InternalType p10 = static_cast<InternalType>(row0[x + 1]);
            InternalType p01 = static_cast<InternalType>(row1[x]);
            InternalType p11 = static_cast<InternalType>(row1[x + 1]);

            InternalType result = (p00 * inv_u + p10 * u) * inv_v +
                                  (p01 * inv_u + p11 * u) * v;

            return T(result);
#endif
        }

        void fillAround(cv::Mat& img, int x, int y, cv::Vec3b&& color) {
            using namespace cv;
            if (x >=0 && x < img.cols && y >= 0 && y < img.rows) {
                img.at<Vec3b>(y, x) = color;
            }
            bool left = x > 0, right = x < img.cols-1, up = y > 0, down = y < img.rows-1;
            if (left) {
                img.at<Vec3b>(y, x-1) = color;
            }
            if (right) {
                img.at<Vec3b>(y, x+1) = color;
            }
            if (up) {
                img.at<Vec3b>(y-1, x) = color;
            }
            if (down) {
                img.at<Vec3b>(y+1, x) = color;
            }
            if (up && left) {
                img.at<Vec3b>(y-1, x-1) = color;
            }
            if (up && right) {
                img.at<Vec3b>(y-1, x+1) = color;
            }
            if (down && left) {
                img.at<Vec3b>(y+1, x-1) = color;
            }
            if (down && left) {
                img.at<Vec3b>(y+1, x+1) = color;
            }
        }
    };

    class CameraSingle {

    private:
        CameraInternal* cameraInternal = nullptr;
        std::vector<Mat> cached_images;
        std::vector<std::vector<std::vector<uchar>>> cached_images_raw_multi_angles;

    public:
        CameraSingle(std::string camera_path, std::string camera_params_path, int width, int height) {

            cameraInternal = new CameraInternal(camera_path, width, height, width/10, height/10, camera_params_path);

            spdlog::info("streamOn start.\n");
            cameraInternal->streamOn();
            spdlog::info("streamOn end.\n");
        }
        ~CameraSingle() {
            cameraInternal->streamOff();
            delete cameraInternal;
        }

        void shootManualImages(Mat& fisheye, int ev_lv) {
            spdlog::info("Start shooting manual images.\n");
            cameraInternal->shootManualExposure(fisheye, ev_lv);
            spdlog::info("Shooting manual images finished.\n");
        }

        void shootAutoImages(Mat& fisheye) {
            spdlog::info("Start shooting auto images.\n");
            cameraInternal->shootAutoExposure(fisheye);
            spdlog::info("Shooting auto images finished.\n");
        }

        inline Mat shootAutoToCacheMatRealtime(std::vector<uchar>* p_buf = nullptr) {
            return cameraInternal->shootAutoToCacheMatRealtime(p_buf);
        }

        inline void cachedImage(cv::Mat& mat) {
            cameraInternal->cachedImage(mat);
        }

        inline void shootAutoToCacheRealtime() {
            cameraInternal->shootAutoToCacheRealtime();
        }

        inline bool decodeRealtimeImage(cv::Mat& image, long& timestamp) {
            return cameraInternal->decodeRealtimeImage(image, timestamp);
        }

        void shootSrcImagesToCacheMultiAngles() {
            spdlog::info("Start shooting cache images.\n");
            size_t count = cached_images_raw_multi_angles.size();
            cached_images_raw_multi_angles.resize(count+1);
            cameraInternal->shootExposureBracketing(cached_images_raw_multi_angles[count]);
            spdlog::info("Shooting src finished, wait for merge to panorama.\n");
        }

        std::vector<Mat>& cachedImages() {
            return cached_images;
        }

        void mergeCachedImagesMultiAngles(Mat& panorama, std::vector<Mat>& fisheye, size_t panorama_composite_index_left = 0, size_t panorama_composite_index_right = 1,
                                          float contrast_weight = 1.0f, float saturation_weight = 1.0f, float exposure_weight = 0.0f, const Eigen::Matrix4d& correct = Eigen::Matrix4d::Identity()) {
            if (cached_images_raw_multi_angles.size() < 2) {
                return;
            }
            if (cached_images_raw_multi_angles.size()-1 < panorama_composite_index_left || cached_images_raw_multi_angles.size()-1 < panorama_composite_index_right) {
                throw std::runtime_error("The panorama_composite_index is out of range.");
            }

            std::vector<Mat> fusions, fusions_median_blur;
            fusions.resize(cached_images_raw_multi_angles.size());
            fusions_median_blur.resize(cached_images_raw_multi_angles.size());
            for (int angle=0; angle<cached_images_raw_multi_angles.size(); angle++) {
                cached_images.clear();
                for (int j=0; j<cached_images_raw_multi_angles[angle].size(); j++) {
                    cached_images.emplace_back(cameraInternal->decodeBufferFrame(true, false, &(cached_images_raw_multi_angles[angle][j])));
                }
                if (cached_images.size() > 1) {
                    Mat mat;
                    Ptr<MergeMertens> mergeMertens = createMergeMertens(contrast_weight, saturation_weight, exposure_weight);
                    mergeMertens->process(cached_images, mat);
                    //fusions[i] = mat[i] * 255;
                    mat.convertTo(fusions[angle], CV_8UC3, 255.0);
                    //中值滤波
                    medianBlur(fusions[angle], fusions_median_blur[angle], 3);
                } else {
                    fusions_median_blur[angle] = cached_images[0];
                }
            }

            spdlog::info("Exposure fusion finished, start merge to panorama.\n");

            Mat sharpenKernel = (Mat_<float>(3, 3) <<
                                                   0, -1, 0,
                    -1, 5, -1,
                    0, -1, 0);

            size_t angle_count = cached_images_raw_multi_angles.size();

            fisheye.clear();
            fisheye.resize(angle_count);
            for (int angle=0; angle<angle_count; angle++) {
                Mat tmp1;

                GaussianBlur(fusions_median_blur[angle], tmp1, Size(3, 3), 1.5);
                filter2D(tmp1, fisheye[angle], fusions_median_blur[angle].depth(), sharpenKernel);
            }
            spdlog::info("Merge to panorama start.\n");

            merge_to_panorama_with_correct(fisheye[panorama_composite_index_left], fisheye[panorama_composite_index_right], panorama, correct);

            cached_images_raw_multi_angles.clear();

            spdlog::info("Merge to panorama done.\n");
            spdlog::info("Cached images processing finished.\n");
        }

        inline bool projectPoints(std::vector<Point3f>& objPoints, std::vector<Point2f>& imgPoints) {
            return cameraInternal->projectPoints(objPoints, imgPoints);
        }

        CameraParams getMainCameraParams() {
            return cameraInternal->getMainCameraParams();
        }

        inline std::array<double,4> undistort(cv::Mat& distorted_img, cv::Mat& undistorted_img, cv::Mat& r) {//return intrinsic, order:fx,fy,cx,cy
            return cameraInternal->undistort(distorted_img, undistorted_img, r);
        }

        void merge_to_panorama(Mat& left, Mat& right, Mat& target) {
            using namespace std;
            int height = 4096;//min(left.rows, right.rows);//5376*2688
            int width = height*2;

            vector<Point3f> objPoints;
            vector<Point2f> imgPoints;
            vector<Point2i> targetPoints;

            double r=1.0, h, v;

            for (int i=0; i<height; i++) {//cols
                for (int j=0; j<height; j++) {//rows
                    h=i*180.0/height-90, v=j*180.0/height-90;
                    double r_xz = r*cos(v*M_PI/180.0);
                    double x = r_xz*sin(h*M_PI/180.0);
                    double y = r*sin(v*M_PI/180.0);
                    double z = r_xz*cos(h*M_PI/180.0);

                    objPoints.emplace_back(Point3f(x, y, z));
                    targetPoints.emplace_back(i,j);
                }
            }

            cameraInternal->projectPoints(objPoints, imgPoints);

            target = Mat(height, width, left.type());
#define INTERPOLATION
#ifdef INTERPOLATION
            if (left.type() == CV_8UC3) {
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3b>(point) = interpolate<Vec3b>(left, imgPoints[i], Vec3b(0, 0, 0));//right.at<Vec3b>(imgPointsLeft[i]);
                }
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3b>(point.y, point.x+height) = interpolate<Vec3b>(right, imgPoints[i], Vec3b(0, 0, 0));//left.at<Vec3b>(imgPointsLeft[i]);
                }
            } else if (left.type() == CV_32FC3) {
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3f>(point) = interpolate<Vec3f>(left, imgPoints[i], Vec3b(0, 0, 0));//right.at<Vec3f>(imgPointsLeft[i]);
                }
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3f>(point.y, point.x+height) = interpolate<Vec3f>(right, imgPoints[i], Vec3b(0, 0, 0));//left.at<Vec3f>(imgPointsLeft[i]);
                }
            }
#else
            if (left.type() == CV_8UC3) {
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3b>(point.y, point.x+height) = right.at<Vec3b>(imgPoints[i]);
                }
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3b>(point) = left.at<Vec3b>(imgPoints[i]);
                }
            } else if (left.type() == CV_32FC3) {
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3f>(point.y, point.x+height) = right.at<Vec3f>(imgPoints[i]);
                }
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3f>(point) = left.at<Vec3f>(imgPoints[i]);
                }
            }
#endif
        }

        void merge_to_panorama_with_correct(Mat& left, Mat& right, Mat& target, const Eigen::Matrix4d& correct) {
            using namespace std;
            int height = 4096;//min(left.rows, right.rows);//5376*2688
            int width = height*2;

            vector<Point3f> objPoints;
            vector<Point2f> imgPoints;
            vector<Point2i> targetPoints;

            double r=1.0, h, v;

            for (int i=0; i<height; i++) {//cols
                for (int j=0; j<height; j++) {//rows
                    h=i*180.0/height-90.0, v=j*180.0/height-90.0;
                    double r_xz = r*cos(v*M_PI/180.0);
                    double x = r_xz*sin(h*M_PI/180.0);
                    double y = r*sin(v*M_PI/180.0);
                    double z = r_xz*cos(h*M_PI/180.0);

                    Eigen::Vector4d np = correct*Eigen::Vector4d(x, y, z, 1);
                    x = np.x();
                    y = np.y();
                    z = np.z();
                    r = pow(x*x+y*y+z*z, 0.5), h = atan(x/z)*180.0/CV_PI, v = asin(y/r)*180.0/CV_PI;
                    if (h < -90.0 || h > 90.0 || v < -90.0 || v > 90.0) {
                        continue;
                    }

                    objPoints.emplace_back(Point3f(x, y, z));
                    targetPoints.emplace_back(i,j);
                }
            }

            cameraInternal->projectPoints(objPoints, imgPoints);

            target = Mat(height, width, left.type());
#define INTERPOLATION
#ifdef INTERPOLATION
            if (left.type() == CV_8UC3) {
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3b>(point.y, point.x+height) = interpolate<Vec3b>(right, imgPoints[i], Vec3b(0, 0, 0));//left.at<Vec3b>(imgPointsLeft[i]);
                }
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3b>(point) = interpolate<Vec3b>(left, imgPoints[i], Vec3b(0, 0, 0));//right.at<Vec3b>(imgPointsLeft[i]);
                }
            } else if (left.type() == CV_32FC3) {
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3f>(point.y, point.x+height) = interpolate<Vec3f>(right, imgPoints[i], Vec3b(0, 0, 0));//left.at<Vec3f>(imgPointsLeft[i]);
                }
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3f>(point) = interpolate<Vec3f>(left, imgPoints[i], Vec3b(0, 0, 0));//right.at<Vec3f>(imgPointsLeft[i]);
                }
            }
#else
            if (left.type() == CV_8UC3) {
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3b>(point.y, point.x+height) = right.at<Vec3b>(imgPoints[i]);
                }
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3b>(point) = left.at<Vec3b>(imgPoints[i]);
                }
            } else if (left.type() == CV_32FC3) {
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3f>(point.y, point.x+height) = right.at<Vec3f>(imgPoints[i]);
                }
                for (size_t i=0; i<targetPoints.size(); i++) {
                    auto& point = targetPoints[i];
                    target.at<Vec3f>(point) = left.at<Vec3f>(imgPoints[i]);
                }
            }
#endif
        }
    private:
        template <typename T>
        inline T interpolate(Mat& mat, Point2f& point, T default_value) {
#if 1
            int x_floor = static_cast<int>(std::floor(point.x));
            int y_floor = static_cast<int>(std::floor(point.y));

            if (x_floor < 0 || x_floor >= mat.cols - 1 || y_floor < 0 || y_floor >= mat.rows - 1) {
                return default_value;
            }

            auto& a = mat.at<T>(y_floor+1, x_floor);
            auto& b = mat.at<T>(y_floor, x_floor);
            auto& c = mat.at<T>(y_floor, x_floor+1);
            auto& d = mat.at<T>(y_floor+1, x_floor+1);

            float x_rate = point.x - x_floor;
            float y_rate = point.y - y_floor;

            T e = a + (d - a) * x_rate;
            T f = b + (c - b) * x_rate;

            return f + (e - f) * y_rate;
#else
            int x = static_cast<int>(point.x);
            int y = static_cast<int>(point.y);

            if (x < 0 || y < 0 || x >= mat.cols - 1 || y >= mat.rows - 1) {
                return default_value;
            }

            float u = point.x - x;
            float v = point.y - y;

            float inv_u = 1.0f - u;
            float inv_v = 1.0f - v;

            const T* row0 = mat.ptr<T>(y);
            const T* row1 = mat.ptr<T>(y + 1);

            using InternalType = cv::Vec3f;

            InternalType p00 = static_cast<InternalType>(row0[x]);
            InternalType p10 = static_cast<InternalType>(row0[x + 1]);
            InternalType p01 = static_cast<InternalType>(row1[x]);
            InternalType p11 = static_cast<InternalType>(row1[x + 1]);

            InternalType result = (p00 * inv_u + p10 * u) * inv_v +
                                  (p01 * inv_u + p11 * u) * v;

            return T(result);
#endif
        }
    };
}
#endif //CAMERA_TEST_TOOL_CAMERA_H
