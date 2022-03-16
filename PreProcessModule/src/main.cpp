//
// Created by YileiYang on 2022/2/27.
//

#include "mnn_rvm.h"
#include "mnn_utils.h"

static void test_mnn_pic(const std::string image_path, const std::string output_path)
{
	std::string mnn_path = "models/mnn/rvm_mobilenetv3_fp32-1080-1920.mnn";

	auto* rvm = new mnn::cv::matting::RobustVideoMatting(mnn_path, 8, 0); // 8 threads, not save content
	//mnncv::MattingContent content;

	// 1. video matting.
	rvm->detect_pic(image_path, output_path);

	delete rvm;
}

static void test_mnn(std::string video_path, std::string output_path)
{
	std::string mnn_path = "models/mnn/rvm_mobilenetv3_fp32-1080-1920.mnn";
	auto* rvm = new mnn::cv::matting::RobustVideoMatting(mnn_path, 16, 0); // 16 threads, not save content

	// 1. video matting.
	rvm->detect_video(video_path, output_path);

	delete rvm;
}

static void test_mnn_capture() 
{
	//std::string mnn_path = "models/mnn/rvm_mobilenetv3_fp32-480-640.mnn";
	//std::string mnn_path = "models/mnn/rvm_mobilenetv3_fp32-1080-1920.mnn";
	std::string mnn_path = "models/mnn/rvm_resnet50_fp32-1080-1920.mnn";
	//std::string mnn_path = "models/mnn/rvm_resnet50_fp32-480-640.mnn";
	auto* rvm = new mnn::cv::matting::RobustVideoMatting(mnn_path, 8, 1); // 8 threads, not save content

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
	test_mnn("resources/demo.mp4",  "result/demo_rvm.mp4");
	
	//test_mnn_pic("resources/test_lite_rvm.jpg", "result/test_lite_mg_matting_input_rvm.jpg");
}

#include <thread>
int main(int argc, char** argv)
{	
	test_preprocess();
	//std::thread t(f);

	return 0;
}
