//
// Created by Zachary on 2024/2/28.
//

#ifndef STEPPERMOTORCONTROLLER_SERIAL_PORT_H
#define STEPPERMOTORCONTROLLER_SERIAL_PORT_H
#include <string>
#include <iostream>
#include <utility>
#define LINUX
#ifdef LINUX
#include <cstring>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <termios.h>
#include <stdbool.h>
#include <fcntl.h>
#include <sys/select.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif
#define UART_ACCESS_LOG

namespace SMC {
    class SerialPort {
    private:
        const static int WAIT_TIME_SECONDS = 2;
        const std::string comName;//串口文件描述名
        const speed_t baud;
#ifdef LINUX
        int fd;
#else
        HANDLE handle = INVALID_HANDLE_VALUE;//串口句柄
#endif

    public:
        explicit SerialPort(std::string comName, speed_t baud = B115200) : comName(std::move(comName)), baud(baud) {}
        ~SerialPort() {
            closeUart();
        }
        bool openUart() {
#ifdef LINUX
            fd = open(comName.c_str(), O_RDWR);
            if (fd == -1) {
                std::cout << comName << " open failed." << std::endl;
                return false;
            }
#else
            handle = CreateFile(comName.c_str(), GENERIC_READ|GENERIC_WRITE, 0, nullptr,OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL|FILE_FLAG_OVERLAPPED, nullptr);
            if (handle == INVALID_HANDLE_VALUE) {
                std::cout << comName << " open failed, code = " << GetLastError() << "." << std::endl;
                return false;
            }
#endif
            std::cout << comName << " open success." << std::endl;
            initBus();
            return true;
        }

        void closeUart() {
#ifdef LINUX
            if (fd != -1) {
                if (close(fd) == 0) {
                    std::cout << comName << " closed." << std::endl;
                } else {
                    std::cout << comName << " close failed." << std::endl;
                }
            }
#else
            if (handle != INVALID_HANDLE_VALUE) {
                if (CloseHandle(handle) == 0) {
                    std::cout << comName << " closed." << std::endl;
                } else {
                    std::cout << comName << " close failed, code = " << GetLastError() << "." << std::endl;
                }
            }
#endif
            std::cout << comName << " closed." << std::endl;
        }

        bool writeUart(const char* buff, size_t len) {
#ifdef LINUX
            fd_set write_set;
            struct timeval tv;
            FD_ZERO(&write_set);
            FD_SET(fd, &write_set);
            tv.tv_sec = WAIT_TIME_SECONDS;
            tv.tv_usec = 0;
            int ret = select(fd+1, 0, &write_set, 0, &tv);
            if (ret != 0 && ret != -1 && FD_ISSET(fd, &write_set)) {
                int doneLen = write(fd, buff, len);
                if (doneLen >= 0) {
#ifdef UART_ACCESS_LOG
                    std::cout << comName << " write success, length = " << doneLen << std::endl;
                    for (int i = 0; i < len; i++) {
                        printf("%02X ", buff[i]);
                    }
                    printf("\n");
#endif
                    return true;
                } else {
                    std::cout << comName << " write failed." << std::endl;
                }
            }
#else
            DWORD doneLen;
            if (WriteFile(handle, buff, len, &doneLen, nullptr)) {
                std::cout << comName << " write success, length = " << doneLen << std::endl;
                return true;
            } else {
                std::cout << comName << " write failed, code = " << GetLastError() << "." << std::endl;
            }
#endif
            return false;
        }

        bool readUart(char* buff, size_t len) {
#ifdef LINUX
            fd_set read_set;
            struct timeval tv;
            FD_ZERO(&read_set);
            FD_SET(fd, &read_set);
            tv.tv_sec = WAIT_TIME_SECONDS;
            tv.tv_usec = 0;
            int ret = select(fd+1, &read_set, 0, 0, &tv);
            if (ret != 0 && ret != -1 && FD_ISSET(fd, &read_set)) {
                int doneLen = 0;
                do {
                    doneLen += read(fd, buff+doneLen, len);
                } while (doneLen < len);
#ifdef UART_ACCESS_LOG
                std::cout << comName << " read success, length = " << doneLen << std::endl;
                for (int i = 0; i < doneLen; i++) {
                    printf("%02X ", buff[i]);
                }
                printf("\n");
#endif
                return true;
            }
#else
            DWORD doneLen;
            if (ReadFile(handle, buff, len, &doneLen, nullptr)) {
                std::cout << comName << " read success, length = " << doneLen << std::endl;
                return true;
            } else {
                std::cout << comName << " read failed, code = " << GetLastError() << "." << std::endl;
            }
#endif
            return false;
        }

        int readUartVariableLength(char* buff, size_t buffLen) {
#ifdef LINUX
            fd_set read_set;
            struct timeval tv;
            FD_ZERO(&read_set);
            FD_SET(fd, &read_set);
            tv.tv_sec = WAIT_TIME_SECONDS;
            tv.tv_usec = 0;
            int ret = select(fd+1, &read_set, 0, 0, &tv);
            if (ret != 0 && ret != -1 && FD_ISSET(fd, &read_set)) {
                int doneLen = read(fd, buff, buffLen);
#ifdef UART_ACCESS_LOG
                if (doneLen > 0) {
                    std::cout << comName << " read success, length = " << doneLen << std::endl;
                    for (int i = 0; i < doneLen; i++) {
                        printf("%02X ", buff[i]);
                    }
                    printf("\n");
                }
#endif
                return doneLen;
            }
#else
            DWORD doneLen;
            if (ReadFile(handle, buff, len, &doneLen, nullptr)) {
                std::cout << comName << " read success, length = " << doneLen << std::endl;
                return true;
            } else {
                std::cout << comName << " read failed, code = " << GetLastError() << "." << std::endl;
            }
#endif
            return 0;
        }

        bool readUartFilter(char* buff, size_t len, char header) {
#ifdef LINUX
            fd_set read_set;
            struct timeval tv;
            FD_ZERO(&read_set);
            FD_SET(fd, &read_set);
            tv.tv_sec = WAIT_TIME_SECONDS;
            tv.tv_usec = 0;
            int ret = select(fd+1, &read_set, NULL, NULL, &tv);
            if (ret != 0 && ret != -1 && FD_ISSET(fd, &read_set)) {
                bool match = false;
                while(read(fd, buff, 1) == 1) {
                    if ((*buff) == header) {
                        match = true;
                        break;
                    }
                }
                if (match) {
                    int doneLen = read(fd, buff+1, len-1)+1;
                    if (doneLen == len) {
#ifdef UART_ACCESS_LOG
                        std::cout << comName << " read success, length = " << doneLen << std::endl;
#endif
                        return true;
                    }
                }
            }
#endif
            std::cout << comName << " read failed." << std::endl;
            return false;
        }

        bool testRead() {
            char buff[2];
            fd_set read_set;
            struct timeval tv;
            FD_ZERO(&read_set);
            FD_SET(fd, &read_set);
            tv.tv_sec = WAIT_TIME_SECONDS;
            tv.tv_usec = 0;
            int ret = select(fd+1, &read_set, NULL, NULL, &tv);
            if (ret != 0 && ret != -1 && FD_ISSET(fd, &read_set)) {
                while(read(fd, buff, 1) == 1) {
                    std::cout << comName << " read success, value = " << buff[0] << std::endl;
                }
            }
            return true;
        }

    private:
        void initBus() {
            std::cout << comName << " init bus start." << std::endl;
#ifdef LINUX
            struct termios ios;
            memset(&ios, 0, sizeof(ios));
            ios.c_cflag |= CLOCAL | CREAD;//设置串口工作在本地模式
            cfmakeraw(&ios);
            cfsetispeed(&ios, baud);//波特率.B115200, B460800
            cfsetospeed(&ios, baud);
            ios.c_cflag &= ~CSIZE;//用数据位掩码清空数据位设置
            ios.c_cflag |= CS8;
            ios.c_cflag &= ~PARENB;//无奇偶校验位
            ios.c_cflag &= ~CSTOPB;//一个停止位
            ios.c_cc[VTIME] = 0;//最小等待时间
            ios.c_cc[VMIN] = 1;//最少字符数量
            tcflush(fd, TCIOFLUSH);//刷新串口
            tcsetattr(fd, TCSANOW, &ios);//配置生效
#else
            DCB dcb;
            GetCommState(handle, &dcb);
            dcb.Parity = NOPARITY;//校验方式
            dcb.ByteSize = 8;//数据位
            dcb.StopBits = ONESTOPBIT;//停止位
            dcb.BaudRate = CBR_115200;//波特率
            SetCommState(handle, &dcb);//设置串口参数
            SetupComm(handle, 1024, 1024);//设置串口读写缓冲区大小
            PurgeComm(handle, PURGE_TXCLEAR|PURGE_RXCLEAR);//清空读写缓冲
#endif
            std::cout << comName << " init bus end." << std::endl;
        }
    };
}
#endif //STEPPERMOTORCONTROLLER_SERIAL_PORT_H
