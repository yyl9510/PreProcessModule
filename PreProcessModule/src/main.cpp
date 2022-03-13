//
// Created by YileiYang on 2022/2/27.
//

#include "lite/lite.h"

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

static void test_mnn()
{
#ifdef ENABLE_MNN
	std::string mnn_path = "models/mnn/rvm_mobilenetv3_fp32-480-640.mnn";
	std::string video_path = "resources/test_lite_rvm_0.mp4";
	std::string output_path = "result/test_lite_rvm_0_mnn.mp4";

	auto* rvm = new lite::mnn::cv::matting::RobustVideoMatting(mnn_path, 16, 0); // 16 threads
	std::vector<lite::types::MattingContent> contents;

	// 1. video matting.
	rvm->detect_video(video_path, output_path, contents, false);

	delete rvm;
#endif
}

static void test_mnn(std::string video_path, std::string output_path)
{
#ifdef ENABLE_MNN
	std::string mnn_path = "models/mnn/rvm_mobilenetv3_fp32-1080-1920.mnn";
	auto* rvm = new lite::mnn::cv::matting::RobustVideoMatting(mnn_path, 8, 0); // 8 threads
	std::vector<lite::types::MattingContent> contents;

	// 1. video matting.
	rvm->detect_video(video_path, output_path, contents, false);

	delete rvm;
#endif
}

//static void test_tnn()
//{
//#ifdef ENABLE_TNN
//
//    std::string proto_path = "../../../hub/tnn/cv/rvm_mobilenetv3_fp32-480-480-sim.opt.tnnproto";
//    std::string model_path = "../../../hub/tnn/cv/rvm_mobilenetv3_fp32-480-480-sim.opt.tnnmodel";
//    std::string video_path = "../../../examples/lite/resources/test_lite_rvm_1.mp4";
//    std::string output_path = "../../../logs/test_lite_rvm_1_tnn.mp4";
//
//    auto* rvm = new lite::tnn::cv::matting::RobustVideoMatting(
//        proto_path, model_path, 16); // 16 threads
//    std::vector<lite::types::MattingContent> contents;
//
//    // 1. video matting.
//    rvm->detect_video(video_path, output_path, contents, false);
//
//    delete rvm;
//#endif
//}

//static void test_ncnn()
//{
//#ifdef ENABLE_NCNN
//
//    // WARNING: Test Failed!
//
//    std::string param_path = "../../../hub/ncnn/cv/rvm_mobilenetv3_fp32-640-480-opt.param";
//    std::string bin_path = "../../../hub/ncnn/cv/rvm_mobilenetv3_fp32-640-480-opt.bin";
//    std::string video_path = "../../../examples/lite/resources/test_lite_rvm_1.mp4";
//    std::string output_path = "../../../logs/test_lite_rvm_1_ncnn.mp4";
//
//    auto* rvm = new lite::ncnn::cv::matting::RobustVideoMatting(
//        param_path, bin_path, 1, 480, 480, 0); // 1 threads
//    std::vector<lite::types::MattingContent> contents;
//
//    // 1. video matting.
//    rvm->detect_video(video_path, output_path, contents, false);
//
//    delete rvm;
//#endif
//}

static void test_mnn_capture() {
#ifdef ENABLE_MNN
	//std::string mnn_path = "models/mnn/rvm_mobilenetv3_fp32-480-640.mnn";
	std::string mnn_path = "models/mnn/rvm_resnet50_fp32-1080-1920.mnn";
	auto* rvm = new lite::mnn::cv::matting::RobustVideoMatting(mnn_path, 8, 0); // 8 threads
	
	rvm->mnn_capture();

	delete rvm;
#endif
}

static void test_lite()
{
	//test_default();
	//test_onnxruntime();
	//test_tnn();
	//test_ncnn();
	//test_mnn();

	//test_mnn_capture();
	//test_mnn("resources/input.mp4",  "result/input_mnn.mp4");
	test_mnn("resources/test_lite_rvm_0.mp4",  "result/test_lite_rvm_0_mnn.mp4");
	//test_mnn("resources/test_lite_rvm_1.mp4",  "result/test_lite_rvm_1_mnn.mp4");
	//test_mnn("resources/test_lite_rvm_0.mp4",  "result/test_lite_rvm_0_mnn.mp4");
	//test_mnn("resources/mobile_capture2.mp4",  "result/mobile_capture2_mnn.mp4");
}

int main(__unused int argc, __unused char* argv[])
{
	test_lite();
	return 0;
}
