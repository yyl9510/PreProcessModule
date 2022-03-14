#pragma once

#define ENABLE_MNN

#define PREPROCESS_DEBUG 1

#ifndef PREPROCESS_EXPORTS
# if (defined _WIN32 || defined WINCE || defined __CYGWIN__)
#   define PREPROCESS_EXPORTS __declspec(dllexport)
#	define PREPROCESS_WIN32
# elif defined __GNUC__ && __GNUC__ >= 4 && (defined(__APPLE__))
#   define PREPROCESS_EXPORTS __attribute__ ((visibility ("default")))
#	define PREPROCESS_UNIX
# endif
#endif


#ifdef PREPROCESS_WIN32
# define NONMINMAX
#endif

#ifndef PREPROCESS_EXPORTS
# define PREPROCESS_EXPORTS
#endif

#include <map>
#include <cmath>
#include <vector>
#include <cassert>
#include <locale.h>
#include <string>
#include <algorithm>
#include <memory>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <type_traits>
#include "opencv2/opencv.hpp"

#include "inference_backend.h"


namespace mnncv
{
	class PREPROCESS_EXPORTS MNNRobustVideoMatting;
	typedef struct PREPROCESS_EXPORTS MattingContentType;
}

namespace mnncv
{
	typedef struct PREPROCESS_EXPORTS mnncv::MattingContentType
	{
		cv::Mat fgr_mat; // fore ground mat 3 channel (R,G,B) 0.~1. or 0~255
		cv::Mat pha_mat; // alpha(matte) 0.~1.
		cv::Mat merge_mat; // merge bg and fg according pha
		bool flag;

		MattingContentType() : flag(false)
		{};
	} MattingContent;
}




#ifdef ENABLE_MNN
namespace mnn
{
	namespace cv
	{
		// matting
		namespace matting
		{
			typedef mnncv::MNNRobustVideoMatting RobustVideoMatting;
		}
	}
}
#endif

