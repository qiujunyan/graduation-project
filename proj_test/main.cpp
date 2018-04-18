
// prj1.cpp : 定义控制台应用程序的入口点。
//
#include "stdafx.h"
//#include "ImagePreprocess.h"
#include "CameraCalibration.h"

int main()
{
	VideoCapture capture(0);
	Mat frame_mat;
	cvNamedWindow("camera");
	while (waitKey(10) != 27)
	{
		capture >> frame_mat;
		Mat frame_calibration_mat = camera_calibration(frame_mat);
		IplImage *frame = &IplImage(frame_calibration_mat);
		IplImage* image = ImagePreprocess_01(frame);
		cvShowImage("camera", image);

	}
	cvDestroyWindow("camera");
	return 0;
}
