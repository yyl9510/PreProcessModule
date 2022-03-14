//
// Created by YileiYang on 2022/2/27.
//

#ifndef PREPROCESS_MNN_RVM_H
#define PREPROCESS_MNN_RVM_H

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
			unsigned int _variant_type = 0); 
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
		bool context_is_update = false;
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
		int input_height;	// tensor's height of the model 
		int input_width;		// tensor's width of the model 
		int dimension_type; // hint only
		unsigned int src_size;
		unsigned int r1i_size;
		unsigned int r2i_size;
		unsigned int r3i_size;
		unsigned int r4i_size;

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

		void initialize_pretreat(); //

		/**
		*  alpha matting from mnn matting result, combines foreground and background together, according to alpha matting
		*  @param output_tensors contains the result of mnn rvm,
		*      while fgr is the foreground data, which has three channels, 
		*      and pha is the alpha matting data, which has one channel
		*  @param content the result container
		*	@param img_h real image height
		*  @param img_w real image width
		*/
		void generate_matting(const std::map<std::string, MNN::Tensor*>& output_tensors,
			mnncv::MattingContent& content,
			int img_h, int img_w);

		void update_context(const std::map<std::string, MNN::Tensor*>& output_tensors);

	public:
		/**
		 * Image Matting Using RVM(https://github.com/PeterL1n/RobustVideoMatting)
		 * @param mat: cv::Mat BGR HWC
		 * @param content: mnncv::MattingContent to catch the detected results.
		 * @param video_mode: false by default.
		 * See https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/inference_zh_Hans.md
		 */
		void detect(const cv::Mat& mat, mnncv::MattingContent& content, bool video_mode = false);
		/**
		 * Video Matting Using RVM(https://github.com/PeterL1n/RobustVideoMatting)
		 * @param video_path: eg. xxx/xxx/input.mp4
		 * @param output_path: eg. xxx/xxx/output.mp4
		 * @param contents: vector of MattingContent to catch the detected results.
		 * @param save_contents: false by default, whether to save MattingContent.
		 * See https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/inference_zh_Hans.md
		 * @param writer_fps: FPS for VideoWriter, 20 by default.
		 */
		void detect_video(const std::string& video_path,
			const std::string& output_path,
			std::vector<mnncv::MattingContent>& contents,
			bool save_contents = false,
			unsigned int writer_fps = 20);
		/**
		* capture live stream from camera
		* @param output_path: eg. xxx/xxx/output.mp4
		*/
		void mnn_capture();
	};
}


#endif //PREPROCESS_MNN_RVM_H