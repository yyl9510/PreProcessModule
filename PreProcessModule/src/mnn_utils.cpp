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

void BrightnessAndContrastAuto(const cv::Mat& src, cv::Mat& dst, float clipHistPercent, int histSize, int lowhist)
{

    CV_Assert(clipHistPercent >= 0);
    CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));

    float alpha, beta;
    double minGray = 0, maxGray = 0;

    //to calculate grayscale histogram
    cv::Mat gray;
    if (src.type() == CV_8UC1) gray = src;
    else if (src.type() == CV_8UC3) cvtColor(src, gray, COLOR_BGR2GRAY);
    else if (src.type() == CV_8UC4) cvtColor(src, gray, COLOR_BGRA2GRAY);
    if (clipHistPercent == 0)
    {
        // keep full available range
        cv::minMaxLoc(gray, &minGray, &maxGray);
    }
    else
    {
        cv::Mat hist; //the grayscale histogram

        float range[] = { 0, 256 };
        const float* histRange = { range };
        bool uniform = true;
        bool accumulate = false;
        calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

        // calculate cumulative distribution from the histogram
        std::vector<float> accumulator(histSize);
        accumulator[0] = hist.at<float>(0);
        for (int i = 1; i < histSize; i++)
        {
            accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
        }

        // locate points that cuts at required value
        float max = accumulator.back();

        int clipHistPercent2;
        clipHistPercent2 = clipHistPercent * (max / 100.0); //make percent as absolute
        clipHistPercent2 /= 2.0; // left and right wings
        // locate left cut
        minGray = 0;
        while (accumulator[minGray] < clipHistPercent2)
            minGray++;

        // locate right cut
        maxGray = histSize - 1;
        while (accumulator[maxGray] >= (max - clipHistPercent2))
            maxGray--;
    }

    // current range
    float inputRange = maxGray - minGray;

    alpha = (histSize - 1) / inputRange;   // alpha expands current range to histsize range
    beta = -minGray * alpha + lowhist;             // beta shifts current range so that minGray will go to 0

    // Apply brightness and contrast normalization
    // convertTo operates with saurate_cast
    src.convertTo(dst, -1, alpha, beta);

    // restore alpha channel from source
    if (dst.type() == CV_8UC4)
    {
        int from_to[] = { 3, 3 };
        cv::mixChannels(&src, 4, &dst, 1, from_to, 1);
    }

	string input_vin = "src";
	namedWindow(input_vin, WINDOW_AUTOSIZE);
	imshow(input_vin, src);
	string output_vin = "dst";
	namedWindow(output_vin, WINDOW_AUTOSIZE);
	imshow(output_vin, src);
	waitKey(0);
}

void addBrightness(const cv::Mat& src) {

	if (src.empty()) {
		cout << "could not be found " << endl;
		return;
	}
	cvtColor(src, src, COLOR_BGRA2BGR);
	char input_vin[] = "input image";
	namedWindow(input_vin, WINDOW_AUTOSIZE);
	imshow(input_vin, src);

	int height = src.rows;
	int width = src.cols;
	float alpha = 1.2;  // 对比度
	float beta = 15;
	Mat dst = Mat::zeros(src.size(), src.type());

	Mat m1;

	src.convertTo(m1, CV_32F); // 转换为32位float

	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			if (src.channels() == 3) {
				// 获取三通道
				//float b = src.at<Vec3b>(row, col)[0];
				//float g = src.at<Vec3b>(row, col)[1];
				//float r = src.at<Vec3b>(row, col)[2];
				float b = m1.at<Vec3f>(row, col)[0];
				float g = m1.at<Vec3f>(row, col)[1];
				float r = m1.at<Vec3f>(row, col)[2];
				dst.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(b * alpha + beta);	//为了防止颜色溢出操作,相当于是对图像色彩变化时做的保护！ 
				dst.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(g * alpha + beta);
				dst.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(r * alpha + beta);
			}
			else if (src.channels() == 1) {
				float v = src.at<uchar>(row, col);
				dst.at<uchar>(row, col) = saturate_cast<uchar>(v * alpha + beta);
			}
		}
	}
	String str = "contrast and brightness changes demo";
	namedWindow(str, WINDOW_AUTOSIZE);
	cv::imshow(str, dst);

	waitKey(0);
}


void drawTransparency(Mat& frame, Mat& transp, int xPos, int yPos) {
	Mat mask;
	vector<Mat> layers;

	split(transp, layers); // seperate channels
	Mat rgb[3] = { layers[0],layers[1],layers[2] };
	mask = layers[3] / 255; // png's alpha channel used as mask

	merge(rgb, 3, transp);  // put together the RGB channels, now transp insn't transparent 
	transp.copyTo(frame.rowRange(yPos, yPos + transp.rows).colRange(xPos, xPos + transp.cols), mask);
}


void addWatermark(cv::Mat& src, cv::Mat& watermark, Info info) {
	cv::Rect roi(src.cols * info.offset_x, src.rows * info.offset_y, src.cols * info.width, src.rows * info.height);
	//cv::Mat frame_roi = src(roi);

	cv::Mat logo;
	cv::resize(watermark, logo,cv::Size(int(src.cols * info.width), int(src.rows * info.height)));
	

	//cv::addWeighted(frame_roi, 1, logo, info.alpha, 1, frame_roi);
	drawTransparency(src, logo, src.cols * info.offset_x, src.rows * info.offset_y);

	
}

