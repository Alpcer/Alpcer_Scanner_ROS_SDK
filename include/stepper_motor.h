//
// Created by Zachary on 2024/3/4.
//

#ifndef STEPPERMOTORCONTROLLER_STEPPER_MOTOR_H
#define STEPPERMOTORCONTROLLER_STEPPER_MOTOR_H
#include "serial_port.h"
#include <utility>
#include <vector>
#include <termios.h>

namespace SMC {
    using namespace SMC;

    struct PidParams {
        uint16_t anglePidKp;
        uint16_t anglePidKi;
        uint16_t speedPidKp;
        uint16_t speedPidKi;
        uint8_t currentPidKp;
        uint8_t currentPidKi;
    };

    struct ProductInfo {
        uint8_t driverName[20];//驱动名称
        uint8_t motoName[20];//电机名称
        uint8_t hardwareVersion;//驱动硬件版本
        uint8_t firmwareVersion;//固件版本
    };

    class StepperMotor {
    public:
        //电机系列
        const static int SERIES_MS = 0;
        const static int SERIES_MG = 1;
        const static int SERIES_MG_CUS = 2;
        const static int SERIES_MG_V3 = 3;
    private:
        SerialPort uart;
        char buff[200];
        int series;
    public:
        StepperMotor(std::string uartComName, int series = SERIES_MS) : uart(std::move(uartComName), series == SERIES_MG_CUS ? B460800 : B115200), series(series) {
            memset(buff, 0, sizeof(buff));
            uart.openUart();
        }

        //-1. 重启(5byte)
        bool reboot(char id) {//不可用
            return sendSimpleCmd(id, 0x07, 0);
        }

        //0. 连接(5byte)
        bool connect(char id) {//不可用
            return sendSimpleCmd(id, 0x1f, 8);
        }

        //1. 读取PID参数(5byte,r:110byte)
        bool readPidParams(char id, PidParams& pidParams) {
            buff[0] = 0x3e;//头字节
            buff[1] = 0x14;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x00;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            if (uart.writeUart(buff, 5)) {
                if (uart.readUart(buff, 110)) {
                    pidParams.anglePidKp = *(uint16_t*)(buff+55);//buff[55];
                    pidParams.anglePidKi = *(uint16_t*)(buff+57);//buff[57];
                    pidParams.speedPidKp = *(uint16_t*)(buff+61);//buff[61];
                    pidParams.speedPidKi = *(uint16_t*)(buff+63);//buff[63];
                    pidParams.currentPidKp = buff[67];
                    pidParams.currentPidKi = buff[69];
                    return true;
                }
            }
            return false;
        }

        //2. 发送简约指令(5byte)
        bool sendSimpleCmd(char id, char cmd, size_t replyBytes) {//不可用
            buff[0] = 0x3e;//头字节
            buff[1] = cmd;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x00;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            if (uart.writeUart(buff, 5)) {
                if (replyBytes > 0) {
                    return uart.readUart(buff, replyBytes);
                } else {
                    return true;
                }
            }
            return false;
        }

        //3. 写入PID参数到ROM(110byte)
        bool writePidParamsToROM(char id, PidParams& pidParams) {
            PidParams params;
            if (readPidParams(id, params)) {
                buff[1] = 0x15;
                buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
                buff[55] = *((uint8_t*)(&pidParams.anglePidKp));//angle kp 低字节
                buff[56] = *((uint8_t*)(&pidParams.anglePidKp)+1);//angle kp 高字节
                buff[57] = *((uint8_t*)(&pidParams.anglePidKi));//angle ki 低字节
                buff[58] = *((uint8_t*)(&pidParams.anglePidKi)+1);//angle ki 高字节
                buff[61] = *((uint8_t*)(&pidParams.speedPidKp));//speed kp 低字节
                buff[62] = *((uint8_t*)(&pidParams.speedPidKp)+1);//speed kp 高字节
                buff[63] = *((uint8_t*)(&pidParams.speedPidKi));//speed ki 低字节
                buff[64] = *((uint8_t*)(&pidParams.speedPidKi)+1);//speed ki 高字节
                buff[67] = pidParams.currentPidKp;//转矩环P参数
                buff[69] = pidParams.currentPidKi;//转矩环I参数
                buff[109] = checkSum(5, 104);
                if (uart.writeUart(buff, 110)) {
                    return uart.readUart(buff, 5);
                }
            }
            return false;
        }

        //4. 读取加速度(5byte,r:10byte)
        bool readAcceleratedSpeed(char id, uint32_t& acceleratedSpeed) {
            buff[0] = 0x3e;//头字节
            buff[1] = 0x33;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x00;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            if (uart.writeUart(buff, 5)) {
                if (uart.readUart(buff, 10)) {
                    if (series == SERIES_MG) {
                        acceleratedSpeed = (*(uint32_t*)(buff+5))/8;
                    } else if (series == SERIES_MG_V3) {
                        acceleratedSpeed = (*(uint32_t*)(buff+5))/36;
                    } else {
                        acceleratedSpeed = *(uint32_t*)(buff+5);
                    }
                    return true;
                }
            }
            return false;
        }

        //5. 写入加速度到RAM(10byte)
        bool writeAcceleratedSpeedToRAM(char id, uint32_t acceleratedSpeed) {
            if (series == SERIES_MG) {
                acceleratedSpeed *= 8;
            } else if (series == SERIES_MG_V3) {
                acceleratedSpeed *= 36;
            }
            buff[0] = 0x3e;//头字节
            buff[1] = 0x34;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x04;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            buff[5] = *((uint8_t*)(&acceleratedSpeed));//加速度低字节1
            buff[6] = *((uint8_t*)(&acceleratedSpeed)+1);//加速度字节2
            buff[7] = *((uint8_t*)(&acceleratedSpeed)+2);//加速度字节3
            buff[8] = *((uint8_t*)(&acceleratedSpeed)+3);//加速度字节4
            buff[9] = checkSum(5, 4);//帧头校验字节，[5]~[10]字节校验和
            if (uart.writeUart(buff, 10)) {
                return uart.readUart(buff, 10);
            }
            return false;
        }

        //6. 读取编码器(5byte,r:12byte)
        bool readEncoder(char id, uint16_t& encoder, uint16_t& encoderRaw, uint16_t& encoderOffset) {
            buff[0] = 0x3e;//头字节
            buff[1] = 0x90;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x00;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            if (uart.writeUart(buff, 5)) {
                if (uart.readUart(buff, 12)) {
                    encoder = *(uint16_t*)(buff+5);
                    encoderRaw = *(uint16_t*)(buff+7);
                    encoderOffset = *(uint16_t*)(buff+9);
                }
            }
            return false;
        }

        //7. 写入编码器值作为电机零点(8byte)
        bool writeAcceleratedSpeedToRAM(char id, uint16_t encoderOffset) {
            buff[0] = 0x3e;//头字节
            buff[1] = 0x91;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x02;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            buff[5] = *((uint8_t*)(&encoderOffset));//编码器零偏低字节
            buff[6] = *((uint8_t*)(&encoderOffset)+1);//编码器零偏高字节
            buff[7] = checkSum(5, 2);//帧头校验字节，[5]~[10]字节校验和
            if (uart.writeUart(buff, 8)) {
                return uart.readUart(buff, 8);
            }
            return false;
        }

        //8. 写入当前位置到ROM作为电机零点(5byte,r:26byte)   该命令暂未开放
        bool writeCurrentPositionToRomAsZeroPoint(char id) {
            buff[0] = 0x3e;//头字节
            buff[1] = 0x19;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x00;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            if (uart.writeUart(buff, 5)) {
                return uart.readUart(buff, 26);
            }
            return false;
        }

        //9. 读取多圈角度(5byte,r:14byte)
        bool readMultiCircleAngle(char id, int64_t& motorAngle) {
            buff[0] = 0x3e;//头字节
            buff[1] = 0x92;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x00;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            if (uart.writeUart(buff, 5)) {
                if (uart.readUart(buff, 14)) {
                    if (series == SERIES_MG) {
                        motorAngle = (*(int64_t*)(buff+5))/8;
                    } else if (series == SERIES_MG_V3) {
                        motorAngle = (*(int64_t*)(buff+5))/36;
                    } else {
                        motorAngle = *(int64_t*)(buff+5);
                    }
                    return true;
                }
            }
            return false;
        }

        //10. 读取单圈角度(5byte,r:10byte)
        bool readSingleCircleAngle(char id, uint32_t& motorAngle) {
            buff[0] = 0x3e;//头字节
            buff[1] = 0x94;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x00;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            if (uart.writeUart(buff, 5)) {
                if (uart.readUart(buff, 10)) {
                    if (series == SERIES_MG) {
                        motorAngle = (*(uint32_t*)(buff+5))/8;
                    } else if (series == SERIES_MG_V3) {
                        motorAngle = (*(uint32_t*)(buff+5))/36;
                    } else {
                        motorAngle = *(uint32_t*)(buff+5);
                    }
                    return true;
                }
            }
            return false;
        }

        //11. 读取电机状态1和错误标志(5byte,r:13byte)
        bool readMotorState1(char id, uint8_t& temperature, uint16_t& voltage, uint8_t& errorState) {
            buff[0] = 0x3e;//头字节
            buff[1] = 0x9a;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x00;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            if (uart.writeUart(buff, 5)) {
                if (uart.readUart(buff, 13)) {
                    temperature = *(uint8_t*)(buff+5);
                    voltage = *(uint16_t*)(buff+7);
                    errorState = *(uint8_t*)(buff+11);
                    return true;
                }
            }
            return false;
        }

        //12. 清除电机错误标志(5byte,r:13byte)
        bool clearMotorErrorState(char id, uint8_t& temperature, uint16_t& voltage, uint8_t& errorState) {
            buff[0] = 0x3e;//头字节
            buff[1] = 0x9b;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x00;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            if (uart.writeUart(buff, 5)) {
                if (uart.readUart(buff, 13)) {
                    temperature = *(uint8_t*)(buff+5);
                    voltage = *(uint16_t*)(buff+7);
                    errorState = *(uint8_t*)(buff+11);
                    return true;
                }
            }
            return false;
        }

        //13. 读取电机状态2(5byte,r:13byte)
        bool readMotorState2(char id, uint8_t& temperature, uint16_t& power, uint16_t& speed, uint16_t& encoder) {
            buff[0] = 0x3e;//头字节
            buff[1] = 0x9c;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x00;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            if (uart.writeUart(buff, 5)) {
                if (uart.readUart(buff, 13)) {
                    temperature = *(uint8_t*)(buff+5);
                    power = *(uint16_t*)(buff+6);
                    speed = *(uint16_t*)(buff+8);
                    encoder = *(uint16_t*)(buff+10);
                    return true;
                }
            }
            return false;
        }

        //14. 读取电机状态3(5byte,r:13byte)
        bool readMotorState3(char id, uint8_t& temperature, uint16_t& iA, uint16_t& iB, uint16_t& iC) {
            buff[0] = 0x3e;//头字节
            buff[1] = 0x9d;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x00;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            if (uart.writeUart(buff, 5)) {
                if (uart.readUart(buff, 13)) {
                    temperature = *(uint8_t*)(buff+5);
                    iA = *(uint16_t*)(buff+6);
                    iB = *(uint16_t*)(buff+8);
                    iC = *(uint16_t*)(buff+10);
                    return true;
                }
            }
            return false;
        }

        //15. 电机关闭, 同时清除状态(5byte)
        bool closeMotor(char id) {
            buff[0] = 0x3e;//头字节
            buff[1] = 0x80;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x00;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            if (uart.writeUart(buff, 5)) {
                return uart.readUart(buff, 5);
            }
            return false;
        }

        //16. 电机停止, 不清除状态(5byte)
        bool stopMotor(char id) {
            buff[0] = 0x3e;//头字节
            buff[1] = 0x81;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x00;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            if (uart.writeUart(buff, 5)) {
                return uart.readUart(buff, 5);
            }
            return false;
        }

        //17. 电机运行(5byte)
        bool runMotor(char id) {
            buff[0] = 0x3e;//头字节
            buff[1] = 0x88;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x00;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            if (uart.writeUart(buff, 5)) {
                return uart.readUart(buff, 5);
            }
            return false;
        }

        //18. 开环控制(8byte,r:13byte) 仅在MS上实现
        bool openLoopControl(char id, uint16_t powerControl) {
            if (series != SERIES_MS) {
                throw std::runtime_error("Implement in series MS only.");
            }
            buff[0] = 0x3e;//头字节
            buff[1] = 0xa0;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x02;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            buff[5] = *((uint8_t*)(&powerControl));//输出功率控制值低字节
            buff[6] = *((uint8_t*)(&powerControl)+1);//输出功率控制值高字节
            buff[7] = checkSum(5, 2);//帧头校验字节，[5]~[6]字节校验和
            if (uart.writeUart(buff, 8)) {
                if (uart.readUart(buff, 13)) {
                    return true;
                }
            }
            return false;
        }

        //20. 速度闭环控制(10byte,r:13byte)
        bool speedClosedLoopControl(char id, uint32_t speedControl) {
            if (series == SERIES_MG) {
                speedControl *= 8;
            } else if (series == SERIES_MG_V3) {
                speedControl *= 36;
            }
            buff[0] = 0x3e;//头字节
            buff[1] = 0xa2;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x04;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            buff[5] = *((uint8_t*)(&speedControl));//电机速度低字节1
            buff[6] = *((uint8_t*)(&speedControl)+1);//电机速度字节2
            buff[7] = *((uint8_t*)(&speedControl)+2);//电机速度字节3
            buff[8] = *((uint8_t*)(&speedControl)+3);//电机速度字节4
            buff[9] = checkSum(5, 4);//帧头校验字节，[5]~[8]字节校验和
            if (uart.writeUart(buff, 10)) {
                if (uart.readUart(buff, 13)) {
                    return true;
                }
            }
            return false;
        }

        //21. 多圈位置闭环控制1(14byte,r:13byte)
        bool multiCircleAngleClosedLoopControl1(char id, int64_t angleControl) {
            if (series == SERIES_MG) {
                angleControl *= 8;
            } else if (series == SERIES_MG_V3) {
                angleControl *= 36;
            }
            buff[0] = 0x3e;//头字节
            buff[1] = 0xa3;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x08;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            buff[5] = *((uint8_t*)(&angleControl));//位置控制低字节1
            buff[6] = *((uint8_t*)(&angleControl)+1);//位置控制字节2
            buff[7] = *((uint8_t*)(&angleControl)+2);//位置控制字节3
            buff[8] = *((uint8_t*)(&angleControl)+3);//位置控制字节4
            buff[9] = *((uint8_t*)(&angleControl)+4);//位置控制字节5
            buff[10] = *((uint8_t*)(&angleControl)+5);//位置控制字节6
            buff[11] = *((uint8_t*)(&angleControl)+6);//位置控制字节7
            buff[12] = *((uint8_t*)(&angleControl)+7);//位置控制字节8
            buff[13] = checkSum(5, 8);//帧头校验字节，[5]~[12]字节校验和
            if (uart.writeUart(buff, 14)) {
                if (uart.readUart(buff, 13)) {
                    return true;
                }
            }
            return false;
        }

        //22. 多圈位置闭环控制2(18byte,r:13byte)
        bool multiCircleAngleClosedLoopControl2(char id, int64_t angleControl, uint32_t maxSpeed, bool autoAdjustKp = false) {//speed:0.01dps/LSB, 即36000代表360°/s
            if (autoAdjustKp) {
                connect(id);
                PidParams pidParams{};
                if (readPidParams(id, pidParams)) {
                    printf("Read PID settings => angle Kp:%d, Ki:%d; speed Kp:%d, Ki:%d; current Kp:%d, Ki:%d\n", pidParams.anglePidKp, pidParams.anglePidKi, pidParams.speedPidKp, pidParams.speedPidKi, pidParams.currentPidKp, pidParams.currentPidKi);
                } else {
                    throw std::runtime_error("Reading PID settings failed!");
                }
                bool pid_changed = false;
                if (maxSpeed > 400 && pidParams.speedPidKp > 150) {
                    pidParams.speedPidKp = 150;
                    pidParams.speedPidKi = 60;
                    pid_changed = true;
                } else if (maxSpeed < 400 && pidParams.speedPidKp <= 150) {
                    pidParams.speedPidKp = 200;
                    pidParams.speedPidKi = 10;
                    pid_changed = true;
                }
                if (pid_changed) {
                    if (writePidParamsToROM(id, pidParams)) {
                        if (reboot(id)) {
                            printf("Rebooting...\n");
                            sleep(2);
                        } else {
                            throw std::runtime_error("Reboot failed!");
                        }
                    } else {
                        throw std::runtime_error("Writing PID settings failed!");
                    }
                }
            }

            if (series == SERIES_MG) {
                angleControl *= 8;
                maxSpeed *= 8;
            } else if (series == SERIES_MG_V3) {
                angleControl *= 36;
                maxSpeed *= 36;
            }
            buff[0] = 0x3e;//头字节
            buff[1] = 0xa4;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x0c;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            buff[5] = *((uint8_t*)(&angleControl));//位置控制低字节1
            buff[6] = *((uint8_t*)(&angleControl)+1);//位置控制字节2
            buff[7] = *((uint8_t*)(&angleControl)+2);//位置控制字节3
            buff[8] = *((uint8_t*)(&angleControl)+3);//位置控制字节4
            buff[9] = *((uint8_t*)(&angleControl)+4);//位置控制字节5
            buff[10] = *((uint8_t*)(&angleControl)+5);//位置控制字节6
            buff[11] = *((uint8_t*)(&angleControl)+6);//位置控制字节7
            buff[12] = *((uint8_t*)(&angleControl)+7);//位置控制字节8
            buff[13] = *((uint8_t*)(&maxSpeed));//速度限制低字节1
            buff[14] = *((uint8_t*)(&maxSpeed)+1);//速度限制字节2
            buff[15] = *((uint8_t*)(&maxSpeed)+2);//速度限制字节3
            buff[16] = *((uint8_t*)(&maxSpeed)+3);//速度限制字节4
            buff[17] = checkSum(5, 12);//帧头校验字节，[5]~[16]字节校验和
            if (uart.writeUart(buff, 18)) {
                if (uart.readUart(buff, 13)) {
                    return true;
                }
            }
            return false;
        }

        //23. 单圈位置闭环控制1(10byte,r:13byte)
        bool singleCircleAngleClosedLoopControl1(char id, uint8_t spinDirection, uint16_t angleControl) {
            if (series == SERIES_MG) {
                angleControl *= 8;
            } else if (series == SERIES_MG_V3) {
                angleControl *= 36;
            }
            buff[0] = 0x3e;//头字节
            buff[1] = 0xa5;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x04;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            buff[5] = spinDirection;//转动方向字节
            buff[6] = *((uint8_t*)(&angleControl));//位置控制低字节
            buff[7] = *((uint8_t*)(&angleControl)+1);//位置控制高字节
            buff[8] = 0x00;//NULL
            buff[9] = checkSum(5, 4);//帧头校验字节，[5]~[8]字节校验和
            if (uart.writeUart(buff, 10)) {
                if (uart.readUart(buff, 13)) {
                    return true;
                }
            }
            return false;
        }

        //24. 单圈位置闭环控制2(14byte,r:13byte)
        bool singleCircleAngleClosedLoopControl2(char id, uint8_t spinDirection, uint16_t angleControl, uint32_t maxSpeed) {
            if (series == SERIES_MG) {
                angleControl *= 8;
                maxSpeed *= 8;
            } else if (series == SERIES_MG_V3) {
                angleControl *= 36;
                maxSpeed *= 36;
            }
            buff[0] = 0x3e;//头字节
            buff[1] = 0xa6;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x08;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            buff[5] = spinDirection;//转动方向字节
            buff[6] = *((uint8_t*)(&angleControl));//位置控制低字节
            buff[7] = *((uint8_t*)(&angleControl)+1);//位置控制高字节
            buff[8] = 0x00;//NULL
            buff[9] = *((uint8_t*)(&maxSpeed));//速度限制低字节1
            buff[10] = *((uint8_t*)(&maxSpeed)+1);//速度限制字节2
            buff[11] = *((uint8_t*)(&maxSpeed)+2);//速度限制字节3
            buff[12] = *((uint8_t*)(&maxSpeed)+3);//速度限制字节4
            buff[13] = checkSum(5, 8);//帧头校验字节，[5]~[12]字节校验和
            if (uart.writeUart(buff, 14)) {
                if (uart.readUart(buff, 13)) {
                    return true;
                }
            }
            return false;
        }

        //25. 增量位置闭环控制1(10byte,r:13byte)
        bool incrementAngleClosedLoopControl1(char id, uint32_t angleIncrement) {
            if (series == SERIES_MG) {
                angleIncrement *= 8;
            } else if (series == SERIES_MG_V3) {
                angleIncrement *= 36;
            }
            buff[0] = 0x3e;//头字节
            buff[1] = 0xa7;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x04;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            buff[5] = *((uint8_t*)(&angleIncrement));//增量位置控制低字节1
            buff[6] = *((uint8_t*)(&angleIncrement)+1);//增量位置控制字节2
            buff[7] = *((uint8_t*)(&angleIncrement)+2);//增量位置控制字节3
            buff[8] = *((uint8_t*)(&angleIncrement)+3);//增量位置控制字节4
            buff[9] = checkSum(5, 4);//帧头校验字节，[5]~[8]字节校验和
            if (uart.writeUart(buff, 10)) {
                if (uart.readUart(buff, 13)) {
                    return true;
                }
            }
            return false;
        }

        //26. 增量位置闭环控制2(14byte,r:13byte)
        bool incrementAngleClosedLoopControl2(char id, uint32_t angleIncrement, uint32_t maxSpeed) {
            if (series == SERIES_MG) {
                angleIncrement *= 8;
                maxSpeed *= 8;
            } else if (series == SERIES_MG_V3) {
                angleIncrement *= 36;
                maxSpeed *= 36;
            }
            buff[0] = 0x3e;//头字节
            buff[1] = 0xa8;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x08;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            buff[5] = *((uint8_t*)(&angleIncrement));//增量位置控制低字节1
            buff[6] = *((uint8_t*)(&angleIncrement)+1);//增量位置控制字节2
            buff[7] = *((uint8_t*)(&angleIncrement)+2);//增量位置控制字节3
            buff[8] = *((uint8_t*)(&angleIncrement)+3);//增量位置控制字节4
            buff[9] = *((uint8_t*)(&maxSpeed));//速度限制低字节1
            buff[10] = *((uint8_t*)(&maxSpeed)+1);//速度限制字节2
            buff[11] = *((uint8_t*)(&maxSpeed)+2);//速度限制字节3
            buff[12] = *((uint8_t*)(&maxSpeed)+3);//速度限制字节4
            buff[13] = checkSum(5, 8);//帧头校验字节，[5]~[12]字节校验和
            if (uart.writeUart(buff, 14)) {
                if (uart.readUart(buff, 13)) {
                    return true;
                }
            }
            return false;
        }

        //27. 读取驱动和电机型号(5byte,r:48byte)
        bool readDriverAndMotorType(char id, uint32_t& acceleratedSpeed) {
            buff[0] = 0x3e;//头字节
            buff[1] = 0x12;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x00;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            if (uart.writeUart(buff, 5)) {
                if (uart.readUart(buff, 48)) {
                    if (series == SERIES_MG) {
                        acceleratedSpeed = (*(uint32_t*)(buff+5))/8;
                    } else if (series == SERIES_MG_V3) {
                        acceleratedSpeed = (*(uint32_t*)(buff+5))/36;
                    } else {
                        acceleratedSpeed = *(uint32_t*)(buff+5);
                    }
                    return true;
                }
            }
            return false;
        }

        //28. 开启或关闭串口角度定时输出命令（主机发送）(9byte,r:8byte)
        bool enableUartTimestamp(char id, bool enable, uint16_t output_frequency) {
            buff[0] = 0x3e;//头字节
            buff[1] = 0xc0;//命令字节
            buff[2] = id;//ID字节, 0x01~0x20
            buff[3] = 0x03;//数据长度字节
            buff[4] = checkSum(0, 4);//帧头校验字节，[0]~[3]字节校验和
            buff[5] = enable ? 0x01 : 0x00;//输出控制字节
            buff[6] = *((uint8_t*)(&output_frequency));//输出频率低字节
            buff[7] = *((uint8_t*)(&output_frequency)+1);//输出频率高字节
            buff[8] = checkSum(5, 3);//帧头校验字节，[5]~[7]字节校验和

            printBuffData(0, 9);
            if (uart.writeUart(buff, 9)) {
                return true;
            }
            return false;
        }

        //29. 读取串口角度及时间戳输出（电机发送）
        bool readTimestamp(char id, uint16_t& angle, uint32_t& timestamp) {
            if (uart.readUartFilter(buff, 12, 0x3e)) {
                angle = (*(uint16_t*)(buff+5));
                timestamp = *(uint32_t*)(buff+7);
                //printBuffData(0, 12);
                printBuffData(0, 12);
                return true;
            }
            return false;
        }
    private:
        char checkSum(int startIndex, size_t length) {
            //TODO check sum
            char sum = 0;
            for (int i= 0; i < length; i++) {
                sum += buff[startIndex + i];
            }
            return sum;
        }

        void printBuffData(int startIndex, int len) {
            for (int i = startIndex; i < (startIndex + len); i++) {
                printf("%02X ", buff[i]);
            }
            printf("\n");
        }
    };
}
#endif //STEPPERMOTORCONTROLLER_STEPPER_MOTOR_H
