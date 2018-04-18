#pragma once
#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include "math.h"

using namespace std;
using namespace cv;

IplImage* ImagePreprocess_01(IplImage* src, float fscale = 1, int wnd_size = 1);
float cal_distance(IplImage *image, int p1, int p2);
