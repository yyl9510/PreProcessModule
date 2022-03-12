//
// Created by DefTruth on 2021/10/10.
//

#include <ctime>
#include "mnn_rvm.h"


using mnncv::MNNRobustVideoMatting;

void imshow(const std::string &name, cv::Mat &mat) {
	cv::imshow(name, mat);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

MNNRobustVideoMatting::MNNRobustVideoMatting(
	const std::string& _mnn_path,
	unsigned int _num_threads,
	unsigned int _variant_type
) : log_id(_mnn_path.data()),
mnn_path(_mnn_path.data()),
num_threads(_num_threads),
variant_type(_variant_type)
{
	initialize_interpreter();
	initialize_context();
	initialize_pretreat();
}

MNNRobustVideoMatting::~MNNRobustVideoMatting()
{
	mnn_interpreter->releaseModel();
	if (mnn_session)
		mnn_interpreter->releaseSession(mnn_session);
}

void MNNRobustVideoMatting::initialize_interpreter()
{
	// 1. init interpreter
	mnn_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path));
	// 2. init schedule_config
	schedule_config.numThread = (int)num_threads;
	MNN::BackendConfig backend_config;
	backend_config.precision = MNN::BackendConfig::Precision_High; // default Precision_High
	schedule_config.backendConfig = &backend_config;
	// 3. create session
	mnn_session = mnn_interpreter->createSession(schedule_config);
	// 4. init input tensor
	src_tensor = mnn_interpreter->getSessionInput(mnn_session, "src");
	// 5. init input dims
	input_height = src_tensor->height();
	input_width = src_tensor->width();
	dimension_type = src_tensor->getDimensionType(); // CAFFE
	mnn_interpreter->resizeTensor(src_tensor, { 1, 3, input_height, input_width });
	mnn_interpreter->resizeSession(mnn_session);
	src_size = 1 * 3 * input_height * input_width;
	// 6. rxi
	r1i_tensor = mnn_interpreter->getSessionInput(mnn_session, "r1i");
	r2i_tensor = mnn_interpreter->getSessionInput(mnn_session, "r2i");
	r3i_tensor = mnn_interpreter->getSessionInput(mnn_session, "r3i");
	r4i_tensor = mnn_interpreter->getSessionInput(mnn_session, "r4i");
#ifdef LITEMNN_DEBUG
	this->print_debug_string();
#endif
}

void MNNRobustVideoMatting::print_debug_string()
{
	std::cout << "LITEMNN_DEBUG LogId: " << log_id << "\n";
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

void MNNRobustVideoMatting::initialize_context()
{
	if (variant_type == VARIANT::MOBILENETV3)
	{
		if (input_width == 1920 && input_height == 1080)
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
			mnn_interpreter->resizeTensor(r1i_tensor, { 1, 16, input_height / 2, input_width / 2 });
			mnn_interpreter->resizeTensor(r2i_tensor, { 1, 20, input_height / 4, input_width / 4 });
			mnn_interpreter->resizeTensor(r3i_tensor, { 1, 40, input_height / 8, input_width / 8 });
			mnn_interpreter->resizeTensor(r4i_tensor, { 1, 64, input_height / 16, input_width / 16 });
			r1i_size = 1 * 16 * (input_height / 2) * (input_width / 2);
			r2i_size = 1 * 20 * (input_height / 4) * (input_width / 4);
			r3i_size = 1 * 40 * (input_height / 8) * (input_width / 8);
			r4i_size = 1 * 64 * (input_height / 16) * (input_width / 16);
		}
	}// RESNET50
	else
	{
		if (input_width == 1920 && input_height == 1080)
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
			mnn_interpreter->resizeTensor(r1i_tensor, { 1, 16, input_height / 2, input_width / 2 });
			mnn_interpreter->resizeTensor(r2i_tensor, { 1, 32, input_height / 4, input_width / 4 });
			mnn_interpreter->resizeTensor(r3i_tensor, { 1, 64, input_height / 8, input_width / 8 });
			mnn_interpreter->resizeTensor(r4i_tensor, { 1, 128, input_height / 16, input_width / 16 });
			r1i_size = 1 * 16 * (input_height / 2) * (input_width / 2);
			r2i_size = 1 * 32 * (input_height / 4) * (input_width / 4);
			r3i_size = 1 * 64 * (input_height / 8) * (input_width / 8);
			r4i_size = 1 * 128 * (input_height / 16) * (input_width / 16);
		}
	}
	// resize session
	mnn_interpreter->resizeSession(mnn_session);
	// init 0.
	std::fill_n(r1i_tensor->host<float>(), r1i_size, 0.f);
	std::fill_n(r2i_tensor->host<float>(), r2i_size, 0.f);
	std::fill_n(r3i_tensor->host<float>(), r3i_size, 0.f);
	std::fill_n(r4i_tensor->host<float>(), r4i_size, 0.f);

	context_is_initialized = true;
}

inline void MNNRobustVideoMatting::initialize_pretreat()
{
	pretreat = std::shared_ptr<MNN::CV::ImageProcess>(
		MNN::CV::ImageProcess::create(
			MNN::CV::BGR,
			MNN::CV::RGB,
			mean_vals, 3,
			norm_vals, 3
		)
		);
}

inline void MNNRobustVideoMatting::transform(const cv::Mat& mat_rs)
{
	pretreat->convert(mat_rs.data, input_width, input_height, mat_rs.step[0], src_tensor);
}

void MNNRobustVideoMatting::detect(const cv::Mat& mat, types::MattingContent& content, bool video_mode)
{
	if (mat.empty()) return;
	int img_h = mat.rows;
	int img_w = mat.cols;
	if (!context_is_initialized) return;

	cv::Mat mat_rs;
	cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
	// 1. make input tensor
	this->transform(mat_rs);

	// 2. inference & run session
	mnn_interpreter->runSession(mnn_session);

	auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);	//now result get
	// 3. generate matting
	this->generate_matting(output_tensors, content, img_h, img_w);
	// 4.  update context (needed for video matting)
	if (video_mode)
	{
		context_is_update = false; // init state.
		this->update_context(output_tensors);
	}
}

void MNNRobustVideoMatting::detect_video(
	const std::string& video_path, const std::string& output_path,
	std::vector<types::MattingContent>& contents, bool save_contents,
	unsigned int writer_fps)
{
	// 0. init video capture
	cv::VideoCapture video_capture(video_path);
	const unsigned int width = video_capture.get(cv::CAP_PROP_FRAME_WIDTH);
	const unsigned int height = video_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
	const unsigned int frame_count = video_capture.get(cv::CAP_PROP_FRAME_COUNT);
	if (!video_capture.isOpened())
	{
		std::cout << "Can not open video: " << video_path << "\n";
		return;
	}
	// 1. init video writer
	cv::VideoWriter video_writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), writer_fps, cv::Size(width, height));
	if (!video_writer.isOpened())
	{
		std::cout << "Can not open writer: " << output_path << "\n";
		return;
	}

	// 2. matting loop
	cv::Mat mat;
	unsigned int i = 0;
	while (video_capture.read(mat))
	{
		i += 1;
		types::MattingContent content;
		this->detect(mat, content, true); // video_mode true
		// 3. save contents and writing out.
		if (content.flag)
		{
			if (save_contents) contents.push_back(content);
			if (!content.merge_mat.empty()) video_writer.write(content.merge_mat);
		}
		// 4. check context states.
		if (!context_is_update) break;
#ifdef LITEMNN_DEBUG
		std::cout << i << "/" << frame_count << " done!" << "\n";
#endif
	}

	// 5. release
	video_capture.release();
	video_writer.release();
}

void MNNRobustVideoMatting::generate_matting(
	const std::map<std::string,  MNN::Tensor*>& output_tensors,
	types::MattingContent& content,
	int img_h, int img_w)
{
	auto device_fgr_ptr = output_tensors.at("fgr");
	auto device_pha_ptr = output_tensors.at("pha");

	MNN::Tensor host_fgr_tensor(device_fgr_ptr, device_fgr_ptr->getDimensionType());  // NCHW
	MNN::Tensor host_pha_tensor(device_pha_ptr, device_pha_ptr->getDimensionType());  // NCHW

	device_fgr_ptr->copyToHostTensor(&host_fgr_tensor);
	device_pha_ptr->copyToHostTensor(&host_pha_tensor);

	float* fgr_ptr = host_fgr_tensor.host<float>();
	float* pha_ptr = host_pha_tensor.host<float>();

	const unsigned int channel_step = input_height * input_width;		//	480*640=307200
	
	//foreground mat
	cv::Mat foremat(input_height, input_width, CV_32FC3);
	cv::Mat rmat(input_height, input_width, CV_32FC1, fgr_ptr);
	cv::Mat gmat(input_height, input_width, CV_32FC1, fgr_ptr + channel_step);
	cv::Mat bmat(input_height, input_width, CV_32FC1, fgr_ptr + 2 * channel_step);
	std::vector<cv::Mat> fgr_mats;
	fgr_mats.push_back(bmat);
	fgr_mats.push_back(gmat);
	fgr_mats.push_back(rmat);
	cv::merge(fgr_mats, foremat);
	//content.fgr_mat.convertTo(content.fgr_mat, CV_8UC3);
	//imshow("foremat", foremat);
	//std::cout << foremat.channels() << "-"<< foremat.rows << "-" << foremat.cols << "-" << foremat.size << "-" << foremat.dims  
	//	<< "-"<< foremat.elemSize() << "-" << foremat.elemSize1() <<  std::endl;

	//background mat
	cv::Mat backmat = cv::imread("resources/background01.jpg");
	cv::resize(backmat, backmat, cv::Size(input_width, input_height));
	backmat.convertTo(backmat, CV_32FC3);
	backmat /= 255.f;
	//std::cout << backmat.channels() << "-" << backmat.rows << "-" << backmat.cols << "-" << backmat.size << std::endl;

	//alpha mat
	cv::Mat alpha(input_height, input_width, CV_32FC1, pha_ptr);
	cv::threshold(alpha, alpha, 0.3f, 1.f, cv::THRESH_BINARY);
	//std::cout << alpha.channels() << "-" << alpha.rows << "-" << alpha.cols << "-" << alpha.size << std::endl;
	//imshow("alpha", alpha);

	//result mat
	cv::Mat out_img = cv::Mat::zeros(backmat.size(), backmat.type());
	//std::cout << out_img.channels() << "-" << out_img.rows << "-" << out_img.cols << "-" << out_img.size << std::endl;
	
	// Find number of pixels.
	//int numberOfPixels = foreground.rows * foreground.cols * foreground.channels();

	// Get floating point pointers to the data matrices
	float* fptr = reinterpret_cast<float*>(foremat.data);
	float* bptr = reinterpret_cast<float*>(backmat.data);
	float* alp_ptr = reinterpret_cast<float*>(alpha.data);
	float* outImagePtr = reinterpret_cast<float*>(out_img.data);

	// Loop over all pixesl once, this is much more efficient than official multiply method, especially in big pictures
	//for (int channel = 0; channel < 3; channel++) {
	//	float *aptr = alp_ptr;
	//	for (int i = 0; i < channel_step; i++, outImagePtr++, fptr++, aptr++, bptr++)
	//	{
	//		*outImagePtr = (*fptr) * (*aptr) + (*bptr) * (1.0f - *aptr);
	//	}
	//}
	for (int pixel = 0; pixel < channel_step; pixel++, alp_ptr++) {
		for (int channel = 0; channel < 3; channel++ ) {
			*outImagePtr++ = (*fptr++) * (*alp_ptr) + (*bptr++) * (1.0f - *alp_ptr);
		}
	}
	out_img *= 255.f;
	out_img.convertTo(out_img, CV_8UC3);
	cv::resize(out_img, out_img, cv::Size(img_w, img_h));
	//cv::imshow("out_img", out_img);
	content.merge_mat = out_img;
	
	// fast assign & channel transpose(CHW->HWC).
	//cv::Mat rmat(input_height, input_width, CV_32FC1, fgr_ptr);
	//cv::Mat gmat(input_height, input_width, CV_32FC1, fgr_ptr + channel_step);
	//cv::Mat bmat(input_height, input_width, CV_32FC1, fgr_ptr + 2 * channel_step);
	//cv::Mat pmat(input_height, input_width, CV_32FC1, pha_ptr); // ref only, zero-copy.
	//imshow("rmat", rmat);
	//imshow("gmat", gmat);
	//imshow("bmat", bmat);
	//imshow("pmat", pmat);
	//cv::Mat rest = 1.f - pmat;
	//pmat *= 255.f;
	//cv::threshold(pmat, pmat, 0.3f, 1.f, cv::THRESH_BINARY);
	//pmat /= 255.f;

	//std::cout << "pmat = " << pmat.rowRange(200,201).colRange(100,500) << std::endl;

	////matting with pmat
	//cv::multiply(pmat, rmat, rmat);
	//cv::multiply(pmat, gmat, gmat);
	//cv::multiply(pmat, bmat, bmat);
	////combine three channels to one
	//std::vector<cv::Mat> fgr_channel_mats;
	//fgr_channel_mats.push_back(bmat);
	//fgr_channel_mats.push_back(gmat);
	//fgr_channel_mats.push_back(rmat);
	//cv::merge(fgr_channel_mats, content.fgr_mat);
	//content.fgr_mat.convertTo(content.fgr_mat, CV_8UC3);
	////show the fore-ground result
	//imshow("Frame0", content.fgr_mat);



	//cv::multiply(pmat, content.fgr_mat, content.fgr_mat);	// Multiply the foreground with the alpha matte
	//cv::multiply(cv::Scalar::all(1.0) - pmat, backmat, backmat);	// Multiply the background with ( 1 - alpha )
	//cv::add(content.fgr_mat, backmat, content.merge_mat);	// Add the masked foreground and background.

	//cv::Mat merge_mats[3];
	//split(backmat, merge_mats);
	//cv::add(bmat.mul(pmat), merge_mats[0].mul(rest), merge_mats[0]);
	//cv::add(gmat.mul(pmat), merge_mats[1].mul(rest), merge_mats[1]);
	//cv::add(rmat.mul(pmat), merge_mats[2].mul(rest), merge_mats[2]);

	//merge_mats[0] = bmat.mul(pmat) + merge_mats[0].mul(rest);
	//merge_mats[1] = gmat.mul(pmat) + merge_mats[1].mul(rest);
	//merge_mats[2] = rmat.mul(pmat) + merge_mats[2].mul(rest);

	//std::vector<cv::Mat> merge_channel_mats;
	//merge_channel_mats.push_back(merge_mats[0]);
	//merge_channel_mats.push_back(merge_mats[1]);
	//merge_channel_mats.push_back(merge_mats[2]);
	//cv::merge(merge_channel_mats, content.merge_mat);

	//cv::imshow("Frame0", content.fgr_mat);
	//cv::imshow("Frame1", backmat);
	//cv::waitKey(0);
	//cv::destroyAllWindows();
	
	// bmat * pmat + (1 - pmat) * 153 = 153 + (bmat - 153) * pmat;
	// pmat is 0 in background,  rgb(120,255,153) is green, when pmat is 0, green is shown
	//cv::Mat mbmat = bmat.mul(pmat) + rest * 153.f;
	//cv::Mat mgmat = gmat.mul(pmat) + rest * 255.f;
	//cv::Mat mrmat = rmat.mul(pmat) + rest * 120.f;

	//std::vector<cv::Mat> alphamatting_channel_mats;
	//cv::Mat *pmat0 = new cv::Mat(pmat), *pmat1 = new cv::Mat(pmat), *pmat2 = new cv::Mat(pmat);
	//merge_channel_mats.push_back(*pmat0);
	//merge_channel_mats.push_back(*pmat1);
	//merge_channel_mats.push_back(*pmat2);
	//cv::merge(alphamatting_channel_mats, content.merge_mat);
	//cv::Mat outImage = cv::Mat::zeros(pmat.size() *3, pmat.type());

	//content.pha_mat = pmat;
	//content.fgr_mat.convertTo(content.fgr_mat, CV_8UC3);
	//content.merge_mat.convertTo(content.merge_mat, CV_8UC3);
	

	if (img_w != input_width || img_h != input_height)
	{
		//cv::resize(content.pha_mat, content.pha_mat, cv::Size(img_w, img_h));
		//cv::resize(content.fgr_mat, content.fgr_mat, cv::Size(img_w, img_h));
		cv::resize(content.merge_mat, content.merge_mat, cv::Size(img_w, img_h));
	}

	content.flag = true;
}

void MNNRobustVideoMatting::update_context(const std::map<std::string, MNN::Tensor*>& output_tensors)
{
	auto device_r1o_ptr = output_tensors.at("r1o");
	auto device_r2o_ptr = output_tensors.at("r2o");
	auto device_r3o_ptr = output_tensors.at("r3o");
	auto device_r4o_ptr = output_tensors.at("r4o");

	device_r1o_ptr->copyToHostTensor(r1i_tensor);
	device_r2o_ptr->copyToHostTensor(r2i_tensor);
	device_r3o_ptr->copyToHostTensor(r3i_tensor);
	device_r4o_ptr->copyToHostTensor(r4i_tensor);

	context_is_update = true;
}

cv::Mat& alpha_blending(cv::Mat & foreground, cv::Mat& background, cv::Mat& alpha ) {
	int width = foreground.cols;
	int height = foreground.rows;
	//resize(foreground, foreground, cv::Size(width, height), cv::INTER_LINEAR);
	resize(background, background, cv::Size(width, height), cv::INTER_LINEAR);
	resize(alpha, alpha, cv::Size(width, height), cv::INTER_LINEAR);

	// Convert Mat to float data type
	foreground.convertTo(foreground, CV_32FC3);
	background.convertTo(background, CV_32FC3);
	// Normalize the alpha mask to keep intensity between 0 and 1
	alpha.convertTo(alpha, CV_32FC3); // 

	// Storage for output image
	cv::Mat outImage = cv::Mat::zeros(foreground.size(), foreground.type());

	//// Find number of pixels.
	//int numberOfPixels = foreground.rows * foreground.cols * foreground.channels();

	//// Get floating point pointers to the data matrices
	//float* fptr = reinterpret_cast<float*>(foreground.data);
	//float* bptr = reinterpret_cast<float*>(background.data);
	//float* aptr = reinterpret_cast<float*>(alpha.data);
	//float* outImagePtr = reinterpret_cast<float*>(outImage.data);

	//// Loop over all pixesl once, this is much more efficient than official multiply method, especially in big pictures
	//for (
	//	int i = 0;
	//	i < numberOfPixels;
	//	i++, outImagePtr++, fptr++, aptr++, bptr++
	//	)
	//{
	//	*outImagePtr = (*fptr) * (*aptr) + (*bptr) * (1 - *aptr);
	//}

	multiply(alpha, foreground, foreground);	// Multiply the foreground with the alpha matte
	multiply(cv::Scalar::all(1.0) - alpha, background, background);	// Multiply the background with ( 1 - alpha )
	add(foreground, background, outImage);	// Add the masked foreground and background.
	return outImage;
}

void MNNRobustVideoMatting::mnn_capture() {
	cv::VideoCapture video_capture(0);
	const unsigned int width = video_capture.get(cv::CAP_PROP_FRAME_WIDTH);
	const unsigned int height = video_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
	//const unsigned int frame_count = video_capture.get(cv::CAP_PROP_FRAME_COUNT);
	if (!video_capture.isOpened())
	{
		std::cout << "Can not open capture! " << std::endl;
		return;
	}

	//get current time
	unsigned int frame_count;
	time_t cur_time = time(0);
	tm* local_cur_time = localtime(&cur_time);
	std::stringstream time_stream;
	time_stream << local_cur_time->tm_year + 1900 << local_cur_time->tm_mon + 1 \
		<< local_cur_time->tm_mday << local_cur_time->tm_hour \
		<< local_cur_time->tm_min << local_cur_time->tm_sec;
	
	cv::VideoWriter video_writer("result/capture_" + time_stream.str() + ".mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 40, cv::Size(width, height));
	cv::Mat background_img = cv::imread("resources/HappyFish.jpg");
	while (video_capture.isOpened())
	{
		// Initialize frame matrix
		cv::Mat frame;

		// Initialize a boolean to check if frames are there or not
		bool isSuccess = video_capture.read(frame);

		// If frames are not there, close it
		if (isSuccess == false)
		{
			std::cout << "Stream disconnected" << std::endl;
			break;
		}
		else {	// If frames are present
			types::MattingContent content;
			this->detect(frame, content, true); // video_mode true

			//cv::Mat outImage = alpha_blending(content.fgr_mat, background_img, content.pha_mat);
			//display frames
			video_writer.write(content.merge_mat);

			// display frames
			cv::imshow("Frame", content.merge_mat);

			// wait for 20 ms between successive frames and break        
			// the loop if key q is pressed
			int key = cv::waitKey(25);
			if (key == 'q')
			{
				std::cout << "Key q key is pressed by the user.  Stopping the video" << std::endl;
				break;
			}
		}
	}
}
