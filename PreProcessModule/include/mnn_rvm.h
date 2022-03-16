//
// Created by YileiYang on 2022/2/27.
//

#ifndef PREPROCESS_MNN_RVM_H
#define PREPROCESS_MNN_RVM_H

#include <memory>
#include "pre_process.h"

#include "MNN/Interpreter.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"
#include "MNN/ImageProcess.hpp"

namespace mnncv
{
	class PREPROCESS_EXPORTS MNNRobustVideoMatting
	{
	public:
		explicit MNNRobustVideoMatting(const std::string& _mnn_path,
			unsigned int _num_threads = 1,
			unsigned int _variant_type = 0,
			const std::string& _background_path = "resources/background01.jpg");
		~MNNRobustVideoMatting();

	private:
		std::shared_ptr<MNN::Interpreter> mnn_interpreter;
		MNN::Session* mnn_session = nullptr;
		MNN::ScheduleConfig schedule_config;
		std::shared_ptr<MNN::CV::ImageProcess> pretreat; // init at runtime, transfer data from mat to tensor
		const char* log_id = nullptr;
		const char* mnn_path = nullptr;

	private:
		const float mean_vals[3] = { 0.f, 0.f, 0.f }; // RGB
		const float norm_vals[3] = { 1.f / 255.f, 1.f / 255.f, 1.f / 255.f };
		// hardcode input node names, hint only.
		// downsample_ratio has been freeze while onnx exported
		// and the input size of each input has been freeze also.
		std::vector<const char*> input_node_names = { "src", "r1i", "r2i", "r3i", "r4i" };
		// hardcode output node names, hint only.
		std::vector<const char*> output_node_names = { "fgr", "pha", "r1o", "r2o", "r3o", "r4o" };
		//bool context_is_update = false;
		bool context_is_initialized = false;

	private:
		enum VARIANT
		{
			MOBILENETV3 = 0,
			RESNET50 = 1
		};

		const unsigned int num_threads; // initialize at runtime.
		// multi inputs, rxi will be update inner video matting process.
		MNN::Tensor* src_tensor = nullptr;
		MNN::Tensor* r1i_tensor = nullptr;
		MNN::Tensor* r2i_tensor = nullptr;
		MNN::Tensor* r3i_tensor = nullptr;
		MNN::Tensor* r4i_tensor = nullptr;
		// input size & variant_type, initialize at runtime.
		const unsigned int variant_type;
		int img_h;	// video 's image height
		int img_w;	// video 's image width
		int tensor_height;	// tensor's height of the model 
		int tensor_width;		// tensor's width of the model 
		int dimension_type; // hint only
		unsigned int src_size;
		unsigned int r1i_size;
		unsigned int r2i_size;
		unsigned int r3i_size;
		unsigned int r4i_size;

		//std::shared_ptr<cv::Mat> foremat;	//forground image,	CV_32FC3, 1/255.f
		//std::shared_ptr<cv::Mat> backmat;	//background image, CV_32FC3, 1/255.f
		//std::shared_ptr<cv::Mat> alpha;		//alpha mat, CV_32FC1
		//std::shared_ptr<cv::Mat> alpha_new;	//alpha mat, CV_32FC1, update alpha_new and use alpha
		
		cv::Mat foremat;	//forground image,	CV_8UC3
		cv::Mat backmat;	//background image, CV_32FC3
		cv::Mat alpha;		//alpha mat, CV_32FC1
		cv::Mat alpha_new;	//alpha mat, CV_32FC1, update alpha_new and use alpha
		bool alpha_is_update = false;	//signal for whether alpha_new is ok
		std::string backmat_path;
		

		// un-copyable
	protected:
		MNNRobustVideoMatting(const MNNRobustVideoMatting&) = delete; //
		MNNRobustVideoMatting(MNNRobustVideoMatting&&) = delete; //
		MNNRobustVideoMatting& operator=(const MNNRobustVideoMatting&) = delete; //
		MNNRobustVideoMatting& operator=(MNNRobustVideoMatting&&) = delete; //

	private:
		void print_debug_string();

	private:
		void transform(const cv::Mat& mat_rs); // without resize

		void initialize_interpreter();

		void initialize_context();

		void initialize_pretreat();

		void set_background_image(const std::string& backpath);

		void update_alpha(const std::map<std::string, MNN::Tensor*>& output_tensors);
		void update_context(const std::map<std::string, MNN::Tensor*>& output_tensors);

		/**
		 * 2. Get the Matting Using RVM(https://github.com/PeterL1n/RobustVideoMatting)
		 * @param video_path: eg. xxx/xxx/input.mp4
		 * @param output_path: eg. xxx/xxx/output.mp4
		 * See https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/inference_zh_Hans.md
		 * @param writer_fps: FPS for VideoWriter, 20 by default.
		 */
		void alpha_matting(bool video_mode = true);

		/**
		 *	3. merge the foremat , backmat and alphamat to merge_mat
		 */
		void alpha_merge(cv::Mat& merge_mat);

	public:
		/**
		 * 1 .detect one picture
		 * @param output_path: eg. xxx/xxx/output.jpg
		 */
		void detect_pic(const std::string& img_path, const std::string& output_path = nullptr);
		/**
		 * 1 .capture live stream from camera
		 */
		void mnn_capture();

		/**
		 * 1. detect video
		 * @param video_path: eg. xxx/xxx/input.mp4
		 * @param output_path: eg. xxx/xxx/output.mp4
		 */
		void detect_video(const std::string& video_path, const std::string& output_path);


	};
}


#endif //PREPROCESS_MNN_RVM_H
