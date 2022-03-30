#pragma once

#ifndef PREPROCESS_MNN_UTILS_H
#define PREPROCESS_MNN_UTILS_H

//#include <ctime>
#include "time.h"
#include <iostream>
#include  <direct.h> 
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

clock_t  start_record();
clock_t stop_record(clock_t t);

void pwd();	//print current work path

void cur_thread(); // print current thread id

void imshow(const std::string& name, cv::Mat& mat);

/*****************************************************************************/

// 去除图片噪点
void remove_small_connected_area(cv::Mat& alpha_pred);	

// clipHistPercent 剪枝（剪去总像素的多少百分比）
// histSize 最后将所有的灰度值归到多大的范围
// lowhist 最小的灰度值
void BrightnessAndContrastAuto(const cv::Mat& src, cv::Mat& dst, float clipHistPercent = 0, int histSize = 255, int lowhist = 0);

//增强图片的亮度和对比度（固定系数）
void addBrightness(const cv::Mat& src);

#endif //PREPROCESS_MNN_UTILS_H

struct Info {
	float alpha;
	float offset_x;
	float offset_y;
	float width;
	float height;
	Info(float alpha, float offset_x, float offset_y, float width, float height) 
	:alpha(alpha), offset_x(offset_x), offset_y(offset_y), width(width), height(height){}
};

void addWatermark(cv::Mat& src, cv::Mat& watermark, Info info);