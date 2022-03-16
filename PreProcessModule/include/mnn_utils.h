#pragma once

#ifndef PREPROCESS_MNN_UTILS_H
#define PREPROCESS_MNN_UTILS_H

#include <ctime>
#include "time.h"
#include <iostream>
#include  <direct.h> 
#include "opencv2/opencv.hpp"

clock_t  start_record();
clock_t stop_record(clock_t t);

void pwd();	//print current work path

void cur_thread(); // print current thread id

void imshow(const std::string& name, cv::Mat& mat);

void remove_small_connected_area(cv::Mat& alpha_pred);	// 去噪点的c++实现



#endif //PREPROCESS_MNN_UTILS_H