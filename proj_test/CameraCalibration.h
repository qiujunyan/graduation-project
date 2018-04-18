#pragma once
#include "HeadFile.h"

Mat camera_calibration(Mat frame)
{
	/*Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
	cameraMatrix.at<double>(0, 0) = 4.525025334771931e+02;
	cameraMatrix.at<double>(0, 1) = 0.006541662194398;
	cameraMatrix.at<double>(0, 2) = 3.099939196147402e+02;
	cameraMatrix.at<double>(1, 1) = 4.522059133012765e+02;
	cameraMatrix.at<double>(1, 2) = 2.498314492717957e+02;

	Mat distCoeffs = Mat::zeros(5, 1, CV_64F);
	distCoeffs.at<double>(0, 0) = -0.400964211243742;
	distCoeffs.at<double>(1, 0) = 0.168310825866293;
	distCoeffs.at<double>(2, 0) = -8.573237654132746e-04;
	distCoeffs.at<double>(3, 0) = 2.373625440266020e-04;
	distCoeffs.at<double>(4, 0) = 0;*/
	
	//红外摄像头矫正参数
	float camera_fx = 611.980382778368;
	float camera_fy = 610.484061441236;
	float camera_cx = 609.3948339480835;
	float camera_cy = 563.8756319778688;

	float camera_k1 = 0.2710083724711451;
	float camera_k2 = -0.401755244621532;
	float camera_p1 = -0.008408804841841753;
	float camera_p2 = 0.0001460250206729388;
	float camera_p3 = 0.1273019108756047;

	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
	cameraMatrix.at<double>(0, 0) = camera_fx;   //fx         
	cameraMatrix.at<double>(0, 2) = camera_fy;   //cx  
	cameraMatrix.at<double>(1, 1) = camera_cx;   //fy  
	cameraMatrix.at<double>(1, 2) = camera_cy;   //cy  

	cameraMatrix.at<double>(0, 1) = 0;
	cameraMatrix.at<double>(1, 0) = 0;
	cameraMatrix.at<double>(2, 0) = 0;
	cameraMatrix.at<double>(2, 1) = 0;
	cameraMatrix.at<double>(2, 2) = 1;

	Mat distCoeffs = Mat::zeros(5, 1, CV_64F);
	distCoeffs.at<double>(0, 0) = camera_k1;   //k1  
	distCoeffs.at<double>(1, 0) = camera_k2;   //k2  
	distCoeffs.at<double>(2, 0) = camera_p1;   //p1  
	distCoeffs.at<double>(3, 0) = camera_p2;   //p2  
	distCoeffs.at<double>(4, 0) = camera_p3;   //p3

	Mat frame_calibrationMat;

	Mat view, rview, map1, map2;
	Size imageSize;
	imageSize = frame.size();
	initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
		getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
		imageSize, CV_16SC2, map1, map2);
	remap(frame, frame_calibrationMat, map1, map2, INTER_LINEAR);

	frame.release();
	cameraMatrix.release();
	distCoeffs.release();

	return frame_calibrationMat;
}
