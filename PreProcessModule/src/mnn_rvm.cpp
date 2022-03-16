//
// Created by YileiYang on 2022/2/27.
//

#include "mnn_rvm.h"
#include "mnn_utils.h"

using mnncv::MNNRobustVideoMatting;

#ifdef PREPROCESS_DEBUG
long total_time = 0;
long write_time = 0;
long runsession_time = 0;
long copy_time = 0;
long mat_matrix_time = 0;
#endif

MNNRobustVideoMatting::MNNRobustVideoMatting(
	const std::string& _mnn_path,
	unsigned int _num_threads,
	unsigned int _variant_type,
	const std::string& background_path
) :	log_id(_mnn_path.data()),
	mnn_path(_mnn_path.data()),
	num_threads(_num_threads),
	variant_type(_variant_type)
{
	initialize_interpreter();
	initialize_context();
	initialize_pretreat();
	set_background_image(background_path);
}

void MNNRobustVideoMatting::initialize_interpreter()
{
	// 1. init interpreter
	mnn_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path));
	mnn_interpreter->setSessionMode(MNN::Interpreter::Session_Release);//add
	// 2. init schedule_config
	schedule_config.numThread = (int)num_threads;
	schedule_config.type = MNN_FORWARD_VULKAN;
	schedule_config.mode = MNN_GPU_TUNING_NORMAL | MNN_GPU_MEMORY_IMAGE;
	schedule_config.backupType = MNN_FORWARD_VULKAN;
	MNN::BackendConfig backend_config;
	backend_config.precision = MNN::BackendConfig::Precision_High; // default Precision_High
	schedule_config.backendConfig = &backend_config;
	
	// 3. create session
	mnn_session = mnn_interpreter->createSession(schedule_config);
	// 4. init input tensor
	src_tensor = mnn_interpreter->getSessionInput(mnn_session, "src");	//it should be src here ,not NULL
	// 5. init input dims
	tensor_height = src_tensor->height();
	tensor_width = src_tensor->width();
	dimension_type = src_tensor->getDimensionType(); // CAFFE
	mnn_interpreter->resizeTensor(src_tensor, { 1, 3, tensor_height, tensor_width });
	mnn_interpreter->resizeSession(mnn_session);
	src_size = 1 * 3 * tensor_height * tensor_width;
	// 6. rxi
	r1i_tensor = mnn_interpreter->getSessionInput(mnn_session, "r1i");
	r2i_tensor = mnn_interpreter->getSessionInput(mnn_session, "r2i");
	r3i_tensor = mnn_interpreter->getSessionInput(mnn_session, "r3i");
	r4i_tensor = mnn_interpreter->getSessionInput(mnn_session, "r4i");
#ifdef PREPROCESS_DEBUG
	this->print_debug_string();
#endif
}

void MNNRobustVideoMatting::initialize_context()
{
	if (variant_type == VARIANT::MOBILENETV3)
	{
		if (tensor_width == 1920 && tensor_height == 1080)
		{
			mnn_interpreter->resizeTensor(r1i_tensor, { 1, 16, 135, 240 });
			mnn_interpreter->resizeTensor(r2i_tensor, { 1, 20, 68, 120 });
			mnn_interpreter->resizeTensor(r3i_tensor, { 1, 40, 34, 60 });
			mnn_interpreter->resizeTensor(r4i_tensor, { 1, 64, 17, 30 });
			r1i_size = 1 * 16 * 135 * 240;
			r2i_size = 1 * 20 * 68 * 120;
			r3i_size = 1 * 40 * 34 * 60;
			r4i_size = 1 * 64 * 17 * 30;
		} // hxw 480x640 480x480 640x480
		else
		{
			mnn_interpreter->resizeTensor(r1i_tensor, { 1, 16, tensor_height / 2, tensor_width / 2 });
			mnn_interpreter->resizeTensor(r2i_tensor, { 1, 20, tensor_height / 4, tensor_width / 4 });
			mnn_interpreter->resizeTensor(r3i_tensor, { 1, 40, tensor_height / 8, tensor_width / 8 });
			mnn_interpreter->resizeTensor(r4i_tensor, { 1, 64, tensor_height / 16, tensor_width / 16 });
			r1i_size = 1 * 16 * (tensor_height / 2) * (tensor_width / 2);
			r2i_size = 1 * 20 * (tensor_height / 4) * (tensor_width / 4);
			r3i_size = 1 * 40 * (tensor_height / 8) * (tensor_width / 8);
			r4i_size = 1 * 64 * (tensor_height / 16) * (tensor_width / 16);
		}
	}// RESNET50
	else
	{
		if (tensor_width == 1920 && tensor_height == 1080)
		{
			mnn_interpreter->resizeTensor(r1i_tensor, { 1, 16, 135, 240 });
			mnn_interpreter->resizeTensor(r2i_tensor, { 1, 32, 68, 120 });
			mnn_interpreter->resizeTensor(r3i_tensor, { 1, 64, 34, 60 });
			mnn_interpreter->resizeTensor(r4i_tensor, { 1, 128, 17, 30 });
			r1i_size = 1 * 16 * 135 * 240;
			r2i_size = 1 * 32 * 68 * 120;
			r3i_size = 1 * 64 * 34 * 60;
			r4i_size = 1 * 128 * 17 * 30;
		} // hxw 480x640 480x480 640x480
		else
		{
			mnn_interpreter->resizeTensor(r1i_tensor, { 1, 16, tensor_height / 2, tensor_width / 2 });
			mnn_interpreter->resizeTensor(r2i_tensor, { 1, 32, tensor_height / 4, tensor_width / 4 });
			mnn_interpreter->resizeTensor(r3i_tensor, { 1, 64, tensor_height / 8, tensor_width / 8 });
			mnn_interpreter->resizeTensor(r4i_tensor, { 1, 128, tensor_height / 16, tensor_width / 16 });
			r1i_size = 1 * 16 * (tensor_height / 2) * (tensor_width / 2);
			r2i_size = 1 * 32 * (tensor_height / 4) * (tensor_width / 4);
			r3i_size = 1 * 64 * (tensor_height / 8) * (tensor_width / 8);
			r4i_size = 1 * 128 * (tensor_height / 16) * (tensor_width / 16);
		}
	}
	// resize session
	mnn_interpreter->resizeSession(mnn_session);
	// init 0.
	//std::fill_n(r1i_tensor->host<float>(), r1i_size, 0.f);
	//std::fill_n(r2i_tensor->host<float>(), r2i_size, 0.f);
	//std::fill_n(r3i_tensor->host<float>(), r3i_size, 0.f);
	//std::fill_n(r4i_tensor->host<float>(), r4i_size, 0.f);

	context_is_initialized = true;
}

inline void MNNRobustVideoMatting::initialize_pretreat()
{
	pretreat = std::shared_ptr<MNN::CV::ImageProcess>(
		MNN::CV::ImageProcess::create( MNN::CV::BGR, MNN::CV::RGB, mean_vals, 3, norm_vals, 3 )
		);
}

void MNNRobustVideoMatting::set_background_image(const std::string& backpath) {
	if (backpath.empty() || backpath.length() == 0 ) {
		return;
	}
	if (backmat_path == backpath) return;
	backmat_path = backpath;
	backmat = cv::imread(backpath);
	//cv::resize(backmat, backmat, cv::Size(tensor_width, tensor_height));
	//backmat.convertTo(backmat, CV_32FC3);
}

inline void MNNRobustVideoMatting::transform(const cv::Mat& mat_rs)
{
	pretreat->convert(mat_rs.data, tensor_width, tensor_height, mat_rs.step[0], src_tensor);	//data from mat_rs to src_tensor
}

void MNNRobustVideoMatting::detect_pic(const std::string& img_path, const std::string& output_path) {
	foremat = cv::imread(img_path);
	if (foremat.empty())
	{
		std::cout << "Can not open image: " << img_path << std::endl;
		return;
	}
	img_w = foremat.cols;
	img_h = foremat.rows;
	
	this->alpha_matting(false);	// not video

	cv::Mat merge_mat = cv::Mat(foremat.size(), foremat.type());
	this->alpha_merge(merge_mat);

	imshow("mergemat", merge_mat);
}

void MNNRobustVideoMatting::detect_video(const std::string& video_path, const std::string& output_path)
{
#ifdef PREPROCESS_DEBUG
	auto total_t = start_record();
#endif	// PREPROCESS_DEBUG
	// 0. init video capture
	cv::VideoCapture video_capture(video_path);
	if (!video_capture.isOpened())
	{
		std::cout << "Can not open video: " << video_path << std::endl;
		return;
	}
	img_w = video_capture.get(cv::CAP_PROP_FRAME_WIDTH);
	img_h = video_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
	const unsigned int frame_count = video_capture.get(cv::CAP_PROP_FRAME_COUNT);

	// 1. init video writer
	cv::VideoWriter video_writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), video_capture.get(cv::CAP_PROP_FPS), cv::Size(img_w, img_h));
	if (!video_writer.isOpened())
	{
		std::cout << "Can not open writer: " << output_path << std::endl;
		return;
	}

	// 2. matting loop
	unsigned int i = 0;
	while (video_capture.read(foremat))
	{
		i += 1;

		// 3. detect alpha matting
		this->alpha_matting(true);	// video

		// 4. merge foremat backmat alphamat to mergemat
		cv::Mat merge_mat = cv::Mat(foremat.size(), foremat.type());
		this->alpha_merge(merge_mat);
		//imshow("mergemat", merge_mat);
#ifdef PREPROCESS_DEBUG
		auto write_t = start_record();
#endif	// PREPROCESS_DEBUG
		// 5. mergemat write to file
		if (!merge_mat.empty()) video_writer.write(merge_mat);
#ifdef PREPROCESS_DEBUG
		write_time += stop_record(write_t);
#endif	// PREPROCESS_DEBUG
		// 6. check context states.
		//if (!context_is_update) break;
#ifdef PREPROCESS_DEBUG
		std::cout << i << "/" << frame_count << " done!" << std::endl;
#endif
	}
#ifdef PREPROCESS_DEBUG
	std::cout << " total_time: " << stop_record(total_t) << std::endl;
	std::cout << " write_time: " << write_time << std::endl;
	std::cout << " runsession_time: " << runsession_time << std::endl;
	std::cout << " copy_time: " << copy_time << std::endl;
	std::cout << " mat_matrix_time: " << mat_matrix_time << std::endl;
#endif
	// 5. release
	video_capture.release();
	video_writer.release();
}

void MNNRobustVideoMatting::mnn_capture() {
	cv::VideoCapture video_capture(0);
	if (!video_capture.isOpened())
	{
		std::cout << "Can not open camera" << std::endl;
		return;
	}
	img_w = video_capture.get(cv::CAP_PROP_FRAME_WIDTH);
	img_h = video_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
	double fps = video_capture.get(cv::CAP_PROP_FPS);

	//get current time
	time_t cur_time = time(0);
	tm* local_cur_time = localtime(&cur_time);
	std::stringstream time_stream;
	time_stream << local_cur_time->tm_year + 1900 << local_cur_time->tm_mon + 1 \
		<< local_cur_time->tm_mday << local_cur_time->tm_hour \
		<< local_cur_time->tm_min << local_cur_time->tm_sec;

	std::string output_path = "result/capture_" + time_stream.str() + ".mp4";
	cv::VideoWriter video_writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(img_w, img_h));
	if (!video_writer.isOpened())
	{
		std::cout << "Can not open writer: " << output_path << std::endl;
		return;
	}

	int count = 0;
	// 2. matting loop
	while (video_capture.isOpened())
	{
		// Initialize a boolean to check if frames are there or not
		bool isSuccess = video_capture.read(foremat);

		// If frames are not there, close it
		if (isSuccess == false)
		{
			std::cout << "Stream disconnected" << std::endl;
			break;
		}
		else 
		{
			// 3. detect alpha matting
			this->alpha_matting(true);	// video

			// 4. merge foremat backmat alphamat to mergemat
			cv::Mat merge_mat = cv::Mat(foremat.size(), foremat.type());
			this->alpha_merge(merge_mat);
			//imshow("mergemat", merge_mat);
#ifdef PREPROCESS_DEBUG
			auto write_t = start_record();
#endif	// PREPROCESS_DEBUG
			// 5. mergemat write to file
			if (!merge_mat.empty()) video_writer.write(merge_mat);
#ifdef PREPROCESS_DEBUG
			write_time += stop_record(write_t);
#endif	// PREPROCESS_DEBUG
			cv::imshow("Frame", merge_mat);
			count++;
			// wait for 1 ms between successive frames and break        
			// the loop if key q is pressed
			int key = cv::waitKey(1);
			if (key == 'q')
			{
				std::cout << "Key q key is pressed by the user.  Stopping the video" << std::endl;
				break;
			}
		}

	}
#ifdef PREPROCESS_DEBUG
	std::cout << " write_time: " << write_time << std::endl;
	std::cout << " runsession_time: " << runsession_time << std::endl;
	std::cout << " copy_time: " << copy_time << std::endl;
	std::cout << " mat_matrix_time: " << mat_matrix_time << std::endl;
#endif
	std::cout << " total time :" << (time(0) - cur_time) << std::endl;
	std::cout << " total count :" << count << std::endl;
	// 5. release
	video_capture.release();
	video_writer.release();

}

void MNNRobustVideoMatting::alpha_matting(bool video_mode)
{
	if (foremat.empty()) return;
	if (!this->context_is_initialized) return;
	//this->alpha_is_update = false;
	//int img_h = mat.rows;
	//int img_w = mat.cols;

	//get foremat from MNNRobustVideoMatting instance
	cv::Mat fore_mat;
	foremat.copyTo(fore_mat);
	if (img_w != tensor_width || img_h != tensor_height) {
		cv::resize(fore_mat, fore_mat, cv::Size(tensor_width, tensor_height));
	}
	//*foremat = cv::Mat::zeros(foremat->size(), foremat->type());
	//imshow("fore_mat", fore_mat);
	//std::cout << fore_mat.type() << std::endl;
	
	//mat = cv::Mat::zeros(mat.size(), mat.type()); 
	//imshow("foremat mat", *foremat);
	// 1. make input tensor
	this->transform(fore_mat);
	//std::cout << foremat->type() << std::endl;
	// 2. inference & run session
#ifdef PREPROCESS_DEBUG
	auto runsession_t = start_record();
#endif // PREPROCESS_DEBUG
	mnn_interpreter->runSession(mnn_session);
	auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);	//now result get

#ifdef PREPROCESS_DEBUG
	runsession_time += stop_record(runsession_t);
#endif	// PREPROCESS_DEBUG
	// 3. update alpha matting
#ifdef PREPROCESS_DEBUG
	auto mat_matrix_t = start_record();
#endif // PREPROCESS_DEBUG
	//this->generate_matting(output_tensors, content, img_h, img_w);
	this->update_alpha(output_tensors);
	//imshow("alpha_new", alpha_new);
#ifdef PREPROCESS_DEBUG
	mat_matrix_time += stop_record(mat_matrix_t);
#endif // PREPROCESS_DEBUG
	// 4.  update context (needed for video matting)
	if (video_mode)
	{
		//context_is_update = false; // init state.
		this->update_context(output_tensors);
	}
}

void MNNRobustVideoMatting::update_alpha(const std::map<std::string, MNN::Tensor*>& output_tensors) {
	//auto device_fgr_ptr = output_tensors.at("fgr");
	auto device_pha_ptr = output_tensors.at("pha");

	//MNN::Tensor host_fgr_tensor(device_fgr_ptr, device_fgr_ptr->getDimensionType());  // NCHW
	MNN::Tensor host_pha_tensor(device_pha_ptr, device_pha_ptr->getDimensionType());  // NCHW

	//device_fgr_ptr->copyToHostTensor(&host_fgr_tensor);
	device_pha_ptr->copyToHostTensor(&host_pha_tensor);	//todo is it duplicated?

	//float* fgr_ptr = host_fgr_tensor.host<float>();
	float* pha_ptr = host_pha_tensor.host<float>();

	const unsigned int channel_step = tensor_height * tensor_width;		//	480*640=307200

	cv::Mat(tensor_height, tensor_width, CV_32FC1, pha_ptr).copyTo(alpha_new);

	cv::threshold(alpha_new, alpha_new, 0.35f, 1.f, cv::THRESH_BINARY);	//dealing with alpha_new

	alpha_new.convertTo(alpha_new, CV_8UC1);
	cv::resize(alpha_new, alpha_new, cv::Size(img_w, img_h));
	
	//alpha_new *= 255;
	//imshow("alpha_new", alpha_new);

	this->alpha_is_update = true;
}

void MNNRobustVideoMatting::update_context(const std::map<std::string, MNN::Tensor*>& output_tensors)
{
	auto device_r1o_ptr = output_tensors.at("r1o");
	auto device_r2o_ptr = output_tensors.at("r2o");
	auto device_r3o_ptr = output_tensors.at("r3o");
	auto device_r4o_ptr = output_tensors.at("r4o");

#ifdef PREPROCESS_DEBUG
	auto copy_t = start_record();
#endif	// PREPROCESS_DEBUG

	device_r1o_ptr->copyToHostTensor(r1i_tensor);
	device_r2o_ptr->copyToHostTensor(r2i_tensor);
	device_r3o_ptr->copyToHostTensor(r3i_tensor);
	device_r4o_ptr->copyToHostTensor(r4i_tensor);

#ifdef PREPROCESS_DEBUG
	copy_time += stop_record(copy_t);
#endif	// PREPROCESS_DEBUG

	//context_is_update = true;
}


void MNNRobustVideoMatting::alpha_merge(cv::Mat& merge_mat) {
	if (alpha_is_update)
	{
		alpha_new.copyTo(alpha);
		alpha_is_update = false;
	}
	if (alpha.empty())
	{
		merge_mat = foremat.clone();
	}
	else
	{
		if (backmat.cols != img_w || backmat.rows != img_h)
		{
			cv::resize(backmat, backmat, cv::Size(img_w, img_h));
		}

		// Get floating point pointers to the data matrices
		char* fptr = reinterpret_cast<char*>(foremat.data);
		char* bptr = reinterpret_cast<char*>(backmat.data);
		char* alp_ptr = reinterpret_cast<char*>(alpha.data);
		char* outImagePtr = reinterpret_cast<char*>(merge_mat.data);
		//std::cout << foremat.type() << backmat.type() << alpha.type() << outImage.type() << std::endl;

		const unsigned int channel_step = foremat.rows * foremat.cols;

		for (int pixel = 0; pixel < channel_step; pixel++, alp_ptr++) {
			for (int channel = 0; channel < 3; channel++) {
				*outImagePtr++ = (*fptr++) * (*alp_ptr) + (*bptr++) * (1 - *alp_ptr);
			}
		}
	}
	//outImage.convertTo(outImage, CV_8UC3);
	//cv::resize(outImage, outImage, cv::Size(img_w, img_h));
}



void MNNRobustVideoMatting::print_debug_string()
{
	std::cout << "PREPROCESS_MNN_DEBUG LogId: " << log_id << "\n";
	std::cout << "=============== Input-Dims ==============\n";
	if (src_tensor) src_tensor->printShape();
	if (r1i_tensor) r1i_tensor->printShape();
	if (r2i_tensor) r2i_tensor->printShape();
	if (r3i_tensor) r3i_tensor->printShape();
	if (r4i_tensor) r4i_tensor->printShape();
	if (dimension_type == MNN::Tensor::CAFFE)
		std::cout << "Dimension Type: (CAFFE/PyTorch/ONNX)NCHW" << "\n";
	else if (dimension_type == MNN::Tensor::TENSORFLOW)
		std::cout << "Dimension Type: (TENSORFLOW)NHWC" << "\n";
	else if (dimension_type == MNN::Tensor::CAFFE_C4)
		std::cout << "Dimension Type: (CAFFE_C4)NC4HW4" << "\n";
	std::cout << "=============== Output-Dims ==============\n";
	auto tmp_output_map = mnn_interpreter->getSessionOutputAll(mnn_session);
	std::cout << "getSessionOutputAll done!\n";
	for (auto it = tmp_output_map.cbegin(); it != tmp_output_map.cend(); ++it)
	{
		std::cout << "Output: " << it->first << ": ";
		it->second->printShape();
	}
	std::cout << "========================================\n";
}

MNNRobustVideoMatting::~MNNRobustVideoMatting()
{
	mnn_interpreter->releaseModel();
	if (mnn_session)
		mnn_interpreter->releaseSession(mnn_session);
}

