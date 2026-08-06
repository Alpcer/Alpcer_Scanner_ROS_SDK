//
// Created by Zachary on 2026/2/28.
//

#ifndef CAMERA_TEST_TOOL_CAMERA_2EYE_H
#define CAMERA_TEST_TOOL_CAMERA_2EYE_H
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

namespace CAMERA2EYE {
    using namespace cv;
    using namespace std;

    const int EV_LEVEL_ARRAY[] = {2047, 1536, 768, 384, 192, 96, 48, 24, 12, 6, 3};
    const int EV_LEVEL_COUNT = 11;
    const int LOW_TO_HIGH_EV_LEVEL_MAP[] = {0, 2, 6, 7, 8, 8, 9, 9, 10, 10, 10};
    const double HIGH_TO_LOW_EV_LEVEL_MAP[] = {0, 0.1304, 1.0033, 1.1678, 1.2960, 1.5799, 1.9413, 2.7692, 4.2380, 6.0282, 7.8153};

    struct CameraParams {
        int width = 0;
        int height = 0;
        cv::Matx33d k;    /*    摄像机内参数矩阵    */
        cv::Vec4d d;     /* 摄像机的4个畸变系数：k1,k2,k3,k4*/
    };

    class CameraInternal {

    public:
        const static int INITIAL_WIDTH = 1280;
        const static int INITIAL_HEIGHT = 480;
        const uint32_t width;
        const uint32_t height;
        const uint32_t width_little;
        const uint32_t height_little;
        std::vector<uchar> buf;
        const std::string comName;

        bool useRgaResize = false;
    private:
        uint32_t cur_width = INITIAL_WIDTH;
        uint32_t cur_height = INITIAL_HEIGHT;
        bool cur_is_auto = true;

        const std::string cameraParamsPathLeft;
        const std::string cameraParamsPathRight;
        const int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        volatile int fd = -1;
        unsigned char *mptr[4];//保存映射后用户空间的首地址
        unsigned int size[4];
        struct v4l2_requestbuffers requestBuffer;
        struct v4l2_buffer frameBuffer;
        struct v4l2_control ctrl;
        struct v4l2_format fmt;

        volatile bool kernelSpaceRequested = false;

        CameraParams cameraParamsLeft;
        CameraParams cameraParamsRight;

        MppRgaDecoder* pMppRgaDecoder = nullptr;
        std::mutex camera_mutex;
        std::vector<uchar> cached_image_raw;
        size_t bytesused_image_raw;
        long timestamp_image_raw;
        cv::Mat cached_image;

    public:
        CameraInternal(std::string comName, uint32_t width, uint32_t height, uint32_t width_little, uint32_t height_little, std::string cameraParamsPathLeft, std::string cameraParamsPathRight) : comName(std::move(comName)), width(width), height(height), width_little(width_little), height_little(height_little), cameraParamsPathLeft(std::move(cameraParamsPathLeft)), cameraParamsPathRight(std::move(cameraParamsPathRight)) {
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

        void setAutoExposure(bool isAuto, bool wait_for_frame_update = false) {
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

        void acquireFrame(std::vector<uchar>* p_buf = nullptr) {
            queueFrameBuffer();
            dequeueFrameBuffer();
            copyFrameData(p_buf);
        }

        static inline void correctImageRotation(Mat& src, Mat& dst) {
            rotate(src, dst, RotateFlags::ROTATE_180);
        }

        Mat decodeBufferFrame(bool onlyValidRange, bool rotationCorrect = false, std::vector<uchar>* p_buf = nullptr) {
            Mat frame;
            frame = imdecode(p_buf == nullptr ? buf : (*p_buf), ImreadModes::IMREAD_COLOR);
            if (rotationCorrect) {
                Mat frame2;
                correctImageRotation(frame, frame2);
                return frame2;
            } else {
                return frame;
            }
        }

        Mat decodeBufferFrameHASync(bool onlyValidRange, bool rotationCorrect, size_t bytesused) {//同步解码
            Mat frame;
            if (cur_width == width) {
                if (pMppRgaDecoder != nullptr) {

//                    long t = common_utils::currentTimeMilliseconds();
                    if (!pMppRgaDecoder->decode(mptr[0], bytesused, frame)) {
                        spdlog::info("mpp decode fail, replace by opencv\n");
                        copyFrameData();
                        frame = imdecode(buf, ImreadModes::IMREAD_COLOR);
                    }
//                    spdlog::info("image decoded, cost:{}s.\n", (common_utils::currentTimeMilliseconds()-t)/1000.0f);
                } else {
                    copyFrameData();
                    frame = imdecode(buf, ImreadModes::IMREAD_COLOR);
                }
            } else {
                copyFrameData();
                frame = imdecode(buf, ImreadModes::IMREAD_COLOR);
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
            frame = imdecode(p_buf == nullptr ? buf : (*p_buf), ImreadModes::IMREAD_GRAYSCALE);
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

        void queueFrameBuffer() {
            if(ioctl(fd, VIDIOC_QBUF, &frameBuffer) < 0) {
                throw std::runtime_error("放回队列失败");
            }
        }

        double decodeBufferFrameGrayValue(std::vector<uchar>* p_buf = nullptr) {
            return mean(decodeBufferFrameGray(false, false, p_buf)).val[0];
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

        inline bool projectPoints(std::vector<Point3f>& objPoints, std::vector<Point2f>& imgPoints, bool isLeft) {
            auto& cameraParams = isLeft ? cameraParamsLeft : cameraParamsRight;
            cv::Vec3d rotation, translation(0,0,0);
            Rodrigues(Mat::eye(3, 3, CV_64F), rotation);
            fisheye::projectPoints(objPoints, imgPoints, rotation, translation, cameraParams.k, cameraParams.d);
            float max_x = cameraParams.width - 1.0f;
            float max_y = cameraParams.height - 1.0f;
            for (auto& p2f : imgPoints) {
                if (p2f.x < 0 || p2f.y < 0 || p2f.x > max_x || p2f.y > max_y) {
                    return false;
                }
            }
            return true;
        }

        inline CameraParams getLeftCameraParams() {
            return cameraParamsLeft;
        }

        inline CameraParams getRightCameraParams() {
            return cameraParamsRight;
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
            //双目相机不分档
            int ev_l = EV_LEVEL_ARRAY[static_cast<int>(std::floor(lv_f))];
            int ev_r = EV_LEVEL_ARRAY[static_cast<int>(std::ceil(lv_f))];
            int ev = ev_l - (lv_f-std::floor(lv_f))*(ev_l-ev_r);

            //获取低曝过曝
            {
                images_raw.resize(3);
                int ev_over = ev*4;
                int ev_under = ev/4;
                if (ev_over > EV_LEVEL_ARRAY[0]) {
                    ev_over = EV_LEVEL_ARRAY[0];
                }
                if (ev_under < EV_LEVEL_ARRAY[EV_LEVEL_COUNT-1]) {
                    ev_under = EV_LEVEL_ARRAY[EV_LEVEL_COUNT-1];
                }
                spdlog::info("{} ev under {}, over {}, base little lv {}\n", comName.c_str(), ev_under, ev_over, lv_f);

                long time = common_utils::currentTimeMilliseconds();
                setExposure(ev_under);
                skipFrames(SKIP_FRAMES);
                acquireFrame(&(images_raw[1]));
                spdlog::info("{} under cost time: {}\n", comName.c_str(), common_utils::currentTimeMilliseconds() - time);

                time = common_utils::currentTimeMilliseconds();
                setExposure(ev_over);
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

        inline void shootAutoRaw(std::vector<uchar>* p_buf = nullptr) {
            setAutoExposure(true, false);
            queueFrameBuffer();
            size_t bytesused = dequeueFrameBuffer();
            if (p_buf != nullptr) {
                p_buf->resize(bytesused);
                memcpy(p_buf->data(), mptr[0], bytesused);
            }
        }

        inline void cachedImage(cv::Mat& mat) {
            {
                std::lock_guard<std::mutex> lock(camera_mutex);
                mat = cached_image.clone();
            }
        }

    private:
        void init() {
            spdlog::info("{} init start.\n", comName.c_str());
            //read camera params
            if (!cameraParamsPathLeft.empty()) {
                FileStorage fs(cameraParamsPathLeft, FileStorage::READ);
                if (!fs.isOpened()) {
                    throw std::runtime_error("Open camera params file failed!");
                }
                fs["width"] >> cameraParamsLeft.width;
                fs["height"] >> cameraParamsLeft.height;
                fs["intrinsic_matrix"] >> cameraParamsLeft.k;
                fs["distortion_coeffs"] >> cameraParamsLeft.d;
                fs.release();

            } else {
                throw std::runtime_error("Camera params left filepath is empty!");
            }

            if (!cameraParamsPathRight.empty()) {
                FileStorage fs(cameraParamsPathRight, FileStorage::READ);
                if (!fs.isOpened()) {
                    throw std::runtime_error("Open camera params file failed!");
                }
                fs["width"] >> cameraParamsRight.width;
                fs["height"] >> cameraParamsRight.height;
                fs["intrinsic_matrix"] >> cameraParamsRight.k;
                fs["distortion_coeffs"] >> cameraParamsRight.d;
                fs.release();

            } else {
                throw std::runtime_error("Camera params right filepath is empty!");
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
            cur_width = main ? width : width_little;
            cur_height = main ? height : height_little;
            //获取摄像头支持格式 ioctl(文件描述符,命令，与命令对应的结构体)
            fmt.type = type; //摄像头采集
            fmt.fmt.pix.width = cur_width; //设置摄像头采集参数，不可以任意设置
            fmt.fmt.pix.height = cur_height;
            fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG; //设置为mjpg格式，则我可以直接写入文件保存，YUYV格式保存后需要转换格式才能查看
            if (ioctl(fd,VIDIOC_S_FMT,&fmt) < 0) {
                throw std::runtime_error("设置格式失败:"+comName);
            } else {
                buf.resize(fmt.fmt.pix.sizeimage);
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

    class CameraSingle {

    private:
        CameraInternal* cameraInternal = nullptr;
        std::vector<Mat> cached_images;
        std::vector<std::vector<std::vector<uchar>>> cached_images_raw_multi_angles;
    public:
        CameraSingle(std::string camera_path, int width, int height, int width_little, int height_little, std::string camera_params_path_left, std::string camera_params_path_right) {

            cameraInternal = new CameraInternal(camera_path, width, height, width_little, height_little, camera_params_path_left, camera_params_path_right);

            spdlog::info("streamOn start.\n");
            cameraInternal->streamOn();
            spdlog::info("streamOn end.\n");
        }
        ~CameraSingle() {
            cameraInternal->streamOff();
            delete cameraInternal;
        }

        void shootAutoImages(Mat& fisheye) {
            spdlog::info("Start shooting auto images.\n");
            cameraInternal->shootAutoExposure(fisheye);
            spdlog::info("Shooting auto images finished.\n");
        }

        void shootSrcImagesToCacheMultiAngles() {
            spdlog::info("Start shooting cache images.\n");
            size_t count = cached_images_raw_multi_angles.size();
            cached_images_raw_multi_angles.resize(count+1);
            cameraInternal->shootExposureBracketing(cached_images_raw_multi_angles[count]);
            spdlog::info("Shooting src finished, wait for merge to panorama.\n");
        }

        inline Mat shootAutoToCacheMatRealtime(std::vector<uchar>* p_buf = nullptr) {
            return cameraInternal->shootAutoToCacheMatRealtime(p_buf);
        }

        inline void shootAutoRaw(std::vector<uchar>* p_buf = nullptr) {
            cameraInternal->shootAutoRaw(p_buf);
        }

        inline void cachedImage(cv::Mat& mat) {
            cameraInternal->cachedImage(mat);
        }

        std::vector<Mat>& cachedImages() {
            return cached_images;
        }

        void mergeCachedImagesMultiAngles(std::vector<Mat>& fisheye, float contrast_weight = 1.0f, float saturation_weight = 1.0f, float exposure_weight = 0.0f, const Eigen::Matrix4d& correct = Eigen::Matrix4d::Identity()) {
            if (cached_images_raw_multi_angles.size() < 1) {
                return;
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

            cached_images_raw_multi_angles.clear();

            spdlog::info("Merge to panorama done.\n");
            spdlog::info("Cached images processing finished.\n");
        }

        inline bool projectPoints(std::vector<Point3f>& objPoints, std::vector<Point2f>& imgPoints, bool isLeft) {
            return cameraInternal->projectPoints(objPoints, imgPoints, isLeft);
        }

        CameraParams getLeftCameraParams() {
            return cameraInternal->getLeftCameraParams();
        }

        CameraParams getRightCameraParams() {
            return cameraInternal->getRightCameraParams();
        }
    };
}
#endif //CAMERA_TEST_TOOL_CAMERA_2EYE_H
