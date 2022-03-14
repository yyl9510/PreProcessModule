//
// Created by YileiYang on 2022/2/27.
//

#include "mnn_rvm.h"
#include "mnn_utils.h"

static void test_mnn()
{
	std::string mnn_path = "models/mnn/rvm_mobilenetv3_fp32-480-640.mnn";
	std::string video_path = "resources/test_lite_rvm_0.mp4";
	std::string output_path = "result/test_lite_rvm_0_mnn.mp4";

	auto* rvm = new mnn::cv::matting::RobustVideoMatting(mnn_path, 16, 0); // 16 threads, not save content
	std::vector<mnncv::MattingContent> contents;

	// 1. video matting.
	rvm->detect_video(video_path, output_path, contents, false);

	delete rvm;
}

static void test_mnn_pic(const std::string image_path, const std::string output_path)
{
	std::string mnn_path = "models/mnn/rvm_mobilenetv3_fp32-480-640.mnn";

	auto* rvm = new mnn::cv::matting::RobustVideoMatting(mnn_path, 1, 0); // 1 threads, not save content
	mnncv::MattingContent content;

	// 1. video matting.
	cv::Mat mat = cv::imread(image_path);
	rvm->detect(mat, content, false);
	imshow("result", content.merge_mat);

	delete rvm;
}

static void test_mnn(std::string video_path, std::string output_path)
{
	std::string mnn_path = "models/mnn/rvm_mobilenetv3_fp32-1080-1920.mnn";
	auto* rvm = new mnn::cv::matting::RobustVideoMatting(mnn_path, 8, 0); // 8 threads, not save content
	std::vector<mnncv::MattingContent> contents;

	// 1. video matting.
	rvm->detect_video(video_path, output_path, contents, false);

	delete rvm;
}

static void test_mnn_capture() {
	//std::string mnn_path = "models/mnn/rvm_mobilenetv3_fp32-480-640.mnn";
	std::string mnn_path = "models/mnn/rvm_resnet50_fp32-1080-1920.mnn";
	auto* rvm = new mnn::cv::matting::RobustVideoMatting(mnn_path, 8, 0); // 8 threads, not save content

	rvm->mnn_capture();

	delete rvm;
}

static void test_preprocess()
{
	//test_default();
	//test_onnxruntime();
	//test_tnn();
	//test_ncnn();
	//test_mnn();

	//test_mnn_capture();
	
	//test_mnn("resources/input.mp4",  "result/input_mnn.mp4");
	//test_mnn("resources/test_rvm_0.mp4",  "result/test_rvm_0_mnn.mp4");
	//test_mnn("resources/test_rvm_1.mp4",  "result/test_rvm_1_mnn.mp4");
	//test_mnn("resources/luoxiang1.mp4",  "result/luoxiang1_mnn.mp4");
	//test_mnn("resources/luoxiang2.mp4",  "result/luoxiang2_mnn.mp4");
	//test_mnn("resources/luoxiang3.mp4",  "result/luoxiang3_mnn.mp4");
	//test_mnn("resources/mobile_capture.mp4",  "result/mobile_capture_mnn.mp4");
	//test_mnn("resources/mobile_capture2.mp4",  "result/mobile_capture2_mnn.mp4");
	//test_mnn("resources/computer_capture.mp4",  "result/computer_capture_mnn.mp4");
	//test_mnn("resources/computer_record_0313.mp4",  "result/computer_record_0313_mnn.mp4");
	
	test_mnn_pic("resources/test_lite_rvm.jpg", "result/test_lite_mg_matting_input_mnn.jpg");
}

int main(int argc, char** argv)
{
	test_preprocess();
	return 0;
}



//static void test_onnxruntime()
//{
//#ifdef ENABLE_ONNXRUNTIME
//    std::string onnx_path = "../../../hub/onnx/cv/rvm_mobilenetv3_fp32.onnx";
//    std::string video_path = "../../../examples/lite/resources/test_lite_rvm_1.mp4";
//    std::string output_path = "../../../logs/test_lite_rvm_1_onnx.mp4";
//
//    auto* rvm = new lite::onnxruntime::cv::matting::RobustVideoMatting(onnx_path, 16); // 16 threads
//    std::vector<lite::types::MattingContent> contents;
//
//    // 1. video matting.
//    rvm->detect_video(video_path, output_path, contents, false, 0.4f);
//
//    delete rvm;
//#endif
//}
