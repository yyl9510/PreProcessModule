//
// Created by YileiYang on 2022/2/27.
//

#include "mnn_rvm.h"
#include "mnn_utils.h"

static void test_mnn_pic(const std::string image_path, const std::string output_path)
{
	std::string mnn_path = "models/mnn/rvm_mobilenetv3_fp32-1080-1920.mnn";

	//auto* rvm = new mnn::cv::matting::RobustVideoMatting(mnn_path, 8, 0); // 8 threads, not save content
	auto* rvm = new mnn::cv::matting::RobustVideoMatting(mnn_path, 8, 0, "C:/Users/58454/Desktop/white.jpg"); // 8 threads, not save content
	//mnncv::MattingContent content;

	// 1. video matting.
	rvm->detect_pic(image_path, output_path);

	delete rvm;
}

static void test_mnn(std::string video_path, std::string output_path)
{
	std::string mnn_path = "models/mnn/rvm_mobilenetv3_fp32-480-480.mnn";
	//std::string mnn_path = "models/mnn/rvm_mobilenetv3_fp32-480-640.mnn";
	//std::string mnn_path = "models/mnn/rvm_mobilenetv3_fp32-1080-1920.mnn";
	
	//std::string mnn_path = "models/mnn/rvm_resnet50_fp32-480-480.mnn";
	//std::string mnn_path = "models/mnn/rvm_resnet50_fp32-480-640.mnn";
	//std::string mnn_path = "models/mnn/rvm_resnet50_fp32-1080-1920.mnn";
	auto* rvm = new mnn::cv::matting::RobustVideoMatting(mnn_path, 16, 0, "resources/background02.jpg"); // 16 threads, not save content

	// 1. video matting.
	rvm->detect_video(video_path, output_path);

	delete rvm;
}

static void test_mnn_capture() 
{
	std::string mnn_path = "models/mnn/rvm_mobilenetv3_fp32-480-480.mnn";	//7
	//std::string mnn_path = "models/mnn/rvm_mobilenetv3_fp32-480-640.mnn";	//5
	//std::string mnn_path = "models/mnn/rvm_mobilenetv3_fp32-1080-1920.mnn";	//4
	
	//std::string mnn_path = "models/mnn/rvm_resnet50_fp32-480-480.mnn";	//4
	//std::string mnn_path = "models/mnn/rvm_resnet50_fp32-480-640.mnn";	//3
	//std::string mnn_path = "models/mnn/rvm_resnet50_fp32-1080-1920.mnn";	//3
	
	
	auto* rvm = new mnn::cv::matting::RobustVideoMatting(mnn_path, 16, 0, "resources/background02.jpg"); // 8 threads, not save content

	rvm->mnn_capture();

	delete rvm;
}

static void test_preprocess()
{
	//test_mnn_capture();
	
	//test_mnn("resources/input.mp4",  "result/input_rvm.mp4");
	//test_mnn("resources/test_rvm_0.mp4",  "result/test_rvm_0_rvm.mp4");
	//test_mnn("resources/test_rvm_1.mp4",  "result/test_rvm_1_rvm.mp4");
	//test_mnn("resources/luoxiang1.mp4",  "result/luoxiang1_rvm.mp4");
	//test_mnn("resources/luoxiang2.mp4",  "result/luoxiang2_rvm.mp4");
	//test_mnn("resources/luoxiang3.mp4",  "result/luoxiang3_rvm.mp4");
	//test_mnn("resources/mobile_capture.mp4",  "result/mobile_capture_rvm.mp4");
	//test_mnn("resources/mobile_capture2.mp4",  "result/mobile_capture2_rvm.mp4");
	//test_mnn("resources/computer_capture.mp4",  "result/computer_capture_rvm.mp4");
	//test_mnn("resources/computer_record_0313.mp4",  "result/computer_record_0313_rvm.mp4");
	//test_mnn("resources/demo.mp4",  "result/demo_rvm.mp4");
	
	//test_mnn_pic("resources/test_lite_rvm.jpg", "result/test_lite_mg_matting_input_rvm.jpg");
	//test_mnn_pic("C:/Users/58454/Desktop/teacher1.jpg", "C:/Users/58454/Desktop/teacher1_result.jpg");
	//test_mnn_pic("C:/Users/58454/Desktop/teacher2.jpg", "C:/Users/58454/Desktop/teacher2_result.jpg");
	//test_mnn_pic("C:/Users/58454/Desktop/teacher3.jpg", "C:/Users/58454/Desktop/teacher3_result.jpg");
	//test_mnn_pic("C:/Users/58454/Desktop/teacher4.jpg", "C:/Users/58454/Desktop/teacher4_result.jpg");
	//test_mnn_pic("C:/Users/58454/Desktop/teacher5.jpg", "C:/Users/58454/Desktop/teacher5_result.jpg");
	//test_mnn_pic("C:/Users/58454/Desktop/teacher6.jpg", "C:/Users/58454/Desktop/teacher6_result.jpg");
	test_mnn_pic("C:/Users/58454/Desktop/teacher7.jpg", "C:/Users/58454/Desktop/teacher7_result.jpg");
	//test_mnn_pic("C:/Users/58454/Desktop/teacher8.jpg", "C:/Users/58454/Desktop/teacher8_result.jpg");
}

void test_image() {
	//string backpath = "C:/Users/58454/Desktop/pptbackground.jpg";
	//string backpath = "C:/Users/58454/Desktop/backwatermark.jpg";
	//string backpath = "C:/Users/58454/Desktop/teacher2_result.jpg";
	string backpath = "C:/Users/58454/Desktop/class.jpg";
	cv::Mat backmat = cv::imread(backpath);
	
	string logopath = "C:/Users/58454/Desktop/logo.png";
	//string logopath = "C:/Users/58454/Desktop/teacher1.jpg";
	//string logopath = "C:/Users/58454/Desktop/teacher7_foreground.png";
	cv::Mat logomat = cv::imread(logopath, IMREAD_UNCHANGED);	//4 channels
	//std::cout << logomat << std::endl;
	//imshow("logomat", logomat);

	Info info(0.6, 0.03, 0.05, 0.05, 0.03);	//watermark info
	//Info info(1, 0.8, 0.8, 0.2, 0.2);	//watermark info
	addWatermark(backmat, logomat, info);

	//imshow("backmat", backmat);
	cv::imwrite("C:/Users/58454/Desktop/result_6.jpg", backmat);
}


int main(int argc, char** argv)
{	
	//test_preprocess();

	test_image();


	return 0;
}
