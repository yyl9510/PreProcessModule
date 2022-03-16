//
// Created by YileiYang on 2022/2/27.
//

#include "mnn_utils.h"

clock_t  start_record() {
	//std::cout << "start record...";
	return clock();
}

clock_t stop_record(clock_t t) {
	clock_t cost_time = clock() - t;
	//std::cout << "stop record... cost :" << cost_time << std::endl;
	return cost_time;
}

void pwd() {
	char ch[128];
	std::cout << _getcwd(ch, 128) << std::endl;
}

void cur_thread() {
	std::cout << "cur thread id: " << std::this_thread::get_id() << std::endl;
}

void imshow(const std::string& name, cv::Mat& mat) {
	cv::imshow(name, mat);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

void remove_small_connected_area(cv::Mat& alpha_pred)
{
	cv::Mat gray, binary;
	alpha_pred.convertTo(gray, CV_8UC1, 255.f);
	// 255 * 0.05 ~ 13
	// https://github.com/yucornetto/MGMatting/blob/main/code-base/utils/util.py#L209
	cv::threshold(gray, binary, 13, 255, cv::THRESH_BINARY);
	// morphologyEx with OPEN operation to remove noise first.
	auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
	cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
	// Computationally connected domain
	cv::Mat labels = cv::Mat::zeros(alpha_pred.size(), CV_32S);
	cv::Mat stats, centroids;
	int num_labels = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, 4);
	if (num_labels <= 1) return; // no noise, skip.
	// find max connected area, 0 is background
	int max_connected_id = 1; // 1,2,...
	int max_connected_area = stats.at<int>(max_connected_id, cv::CC_STAT_AREA);
	for (int i = 1; i < num_labels; ++i)
	{
		int tmp_connected_area = stats.at<int>(i, cv::CC_STAT_AREA);
		if (tmp_connected_area > max_connected_area)
		{
			max_connected_area = tmp_connected_area;
			max_connected_id = i;
		}
	}
	const int h = alpha_pred.rows;
	const int w = alpha_pred.cols;
	// remove small connected area.
	for (int i = 0; i < h; ++i)
	{
		int* label_row_ptr = labels.ptr<int>(i);
		float* alpha_row_ptr = alpha_pred.ptr<float>(i);
		for (int j = 0; j < w; ++j)
		{
			if (label_row_ptr[j] != max_connected_id)
				alpha_row_ptr[j] = 0.f;
		}
	}
}