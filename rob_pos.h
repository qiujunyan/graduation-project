#pragma once
#include "opencv2/opencv.hpp"
#include "CvvImage.h"
#include "math.h"

#define cameraID 1
#define RADIUS bestSideLength/2

using namespace std; 
using namespace cv;

/*****************句柄********************/
CRect RawVideoRect;
CDC *RawVideoPDC;
HDC RawVideoHDC;
CvvImage RawVideoCvvImage;

CRect BinaryFrameRect;
CDC *BinaryFramePDC;
HDC BinaryFrameHDC;
CvvImage BinaryFrameCvvImage;

CRect CalibrationVideoRect;
CDC *CalibrationVideoPDC;
HDC CalibrationVideoHDC;
CvvImage CalibrationVideoCvvImage;

CRect FrameContourRect;
CDC *FrameContourPDC;
HDC FrameContourHDC;
CvvImage FrameContourCvvImage;

/***************获取图像*************/
VideoCapture capture(cameraID);
Mat frame;
IplImage *rawFrame;
IplImage *frameCalibration;//矫正图像
IplImage *frameContour;//图像轮廓
Mat frameCalibrationMat;
Mat frameMat;

/***********第一判断模块***********/
Mat grayframeMat;//灰度图
Mat binaryframeMat;//二值化图
Mat frameContourMat;//轮廓图
IplImage* binaryFrame;
vector<vector<Point>> contours;
vector<Vec4i> hierarchy;
Scalar color(0, 0, 255);//轮廓颜色
int cPCount = 0;//矩形模块计数器
int cPCountSquare = 0;//编码模块计数器
Point2f centerPoint[100];//矩形模块中心点
Point2f centerPointSquare[100];//编码模块中心点
Point2f l, r, u, d;
Point2f vertex[4];//定义矩形的四个顶点
float Alpha[100];//矩形模块倾斜角度
float bestAlpha = 0;//最佳编码模块倾斜角度
Point2f bestCenterPoint;//最佳矩形模块中心点
float sideLength[100];//编码模块边长及网格距离长度
float bestSideLength;//最佳路标编码块边长

//其他
int ThresholdPosition;




/**************函数部分**************/
//距离计算函数
float calculation(Point vertex1, Point vertex2)
{
	float distance = 0;
	distance = sqrt(pow(float(vertex1.x - vertex2.x), 2) + pow(float(vertex1.y - vertex2.y), 2));
	return distance;
}

//矩阵乘法
vector<vector<float>> multiple(vector<vector<float>> v1, vector<vector<float>> v2)
{
	int i, j, k;
	int rows1 = v1.size();
	int cols1 = v1[0].size();
	int rows2 = v2.size();
	int cols2 = v2[0].size();
	vector<vector<float>> v(rows1, vector<float>(cols2, 0));//初始化输出数组
	if (cols1 != rows2)
	{
		//cout << "wrong input" << "\n";
		return v;
	}
	else
	{
		for (i = 0;i < rows1;i++)
		{
			for (j = 0;j < cols2;j++)
			{
				for (k = 0;k < cols1;k++)
				{
					v[i][j] += v1[i][k] * v2[k][j];
				}
				//cout << v[i][j] << "\t";
			}
			//cout << "\n";
		}
		return v;
	}
}

//矩阵减法
vector<vector<float>> vectorMinus(vector<vector<float>> v1, vector<vector<float>> v2)
{
	int i, j;
	int rows1 = v1.size();
	int cols1 = v1[0].size();
	int rows2 = v2.size();
	int cols2 = v2[0].size();
	vector<vector<float>> v(rows1, vector<float>(cols1, 0));//初始化输出数组
	if (rows1 != rows2 || cols1 != cols2)
	{
		cout << "wrong input" << "\n";
		return v;
	}
	else
	{
		for (i = 0;i < rows1;i++)
			for (j = 0;j < cols1;j++)
				v[i][j] = v1[i][j] - v2[i][j];
		return v;
	}
}

//坐标转换
Point2f Convertion(Point2f point)//[u,v]=[x-x0,y-y0]*([cosx,sinx],[-sinx,cosx])
{
	Point2f pointAfterConvertion;
	vector<vector<float>> T(1, vector<float>(2, 0)), coord(1, vector<float>(2, 0)), coordAfterConvertion(1, vector<float>(2, 0));
	vector<vector<float>> R(2, vector<float>(2, 0));
	T = { { bestCenterPoint.x,bestCenterPoint.y } };
	coord = { { point.x,point.y } };
	R = { { cos(bestAlpha),sin(bestAlpha) },{ -sin(bestAlpha),cos(bestAlpha) } };
	coordAfterConvertion = multiple(vectorMinus(coord, T), R);
	pointAfterConvertion.x = coordAfterConvertion[0][0];
	pointAfterConvertion.y = coordAfterConvertion[0][1];
	return pointAfterConvertion;
}

void drawContoursFunction()
{
	line(frameContourMat, vertex[0], vertex[1], Scalar(255, 255, 255), 1, 16);
	line(frameContourMat, vertex[1], vertex[2], Scalar(255, 255, 255), 1, 16);
	line(frameContourMat, vertex[2], vertex[3], Scalar(255, 255, 255), 1, 16);
	line(frameContourMat, vertex[3], vertex[0], Scalar(255, 255, 255), 1, 16);

	l.x = centerPoint[cPCount].x - 10;
	l.y = centerPoint[cPCount].y;

	r.x = centerPoint[cPCount].x + 10;
	r.y = centerPoint[cPCount].y;

	u.x = centerPoint[cPCount].x;
	u.y = centerPoint[cPCount].y - 10;

	d.x = centerPoint[cPCount].x;
	d.y = centerPoint[cPCount].y + 10;
	line(frameContourMat, l, r, Scalar(255, 255, 255), 1, 16);
	line(frameContourMat, u, d, Scalar(255, 255, 255), 1, 16);
}

void drawContoursFunctionSquare()
{
	line(frameContourMat, vertex[0], vertex[1], Scalar(255, 255, 255), 1, 16);
	line(frameContourMat, vertex[1], vertex[2], Scalar(255, 255, 255), 1, 16);
	line(frameContourMat, vertex[2], vertex[3], Scalar(255, 255, 255), 1, 16);
	line(frameContourMat, vertex[3], vertex[0], Scalar(255, 255, 255), 1, 16);

	centerPoint[cPCountSquare].x = (vertex[0].x + vertex[2].x) / 2.0;
	centerPoint[cPCountSquare].y = (vertex[0].y + vertex[2].y) / 2.0;
	l.x = centerPoint[cPCountSquare].x - 10;
	l.y = centerPoint[cPCountSquare].y;

	r.x = centerPoint[cPCountSquare].x + 10;
	r.y = centerPoint[cPCountSquare].y;

	u.x = centerPoint[cPCountSquare].x;
	u.y = centerPoint[cPCountSquare].y - 10;

	d.x = centerPoint[cPCountSquare].x;
	d.y = centerPoint[cPCountSquare].y + 10;
	line(frameContourMat, l, r, Scalar(255, 255, 255), 1, 16);
	line(frameContourMat, u, d, Scalar(255, 255, 255), 1, 16);
}

//画网格
void drawGrid()
{
	Point p1, p2;
	for (int i = 1;i < 10;i++)
	{
		p1.x = 0;
		p1.y = 48 * i;
		p2.x = 639;
		p2.y = 48 * i;
		if (i == 5)
			line(frameContourMat, p1, p2, Scalar(0, 255, 0), 1, 16);
		else
			line(frameContourMat, p1, p2, Scalar(255, 255, 255), 1, 16);
	}
	for (int i = 1;i < 10;i++)
	{
		p1.x = 64 * i;
		p1.y = 0;
		p2.x = 64 * i;
		p2.y = 479;
		if (i == 5)
			line(frameContourMat, p1, p2, Scalar(0, 255, 0), 1, 16);
		else
			line(frameContourMat, p1, p2, Scalar(255, 255, 255), 1, 16);
	}
}

//编码模块ID转换
int coordCalculation(Point2f p)
{
	//数据库，记录编码块坐标
	//(2,-2)、(4,-2)、(6,-2)
	//(2,0)、(4,0)、(6,0)
	//(2,2)、(4,2)、(6,2)
	if (pow((p.x - 2 * bestSideLength), 2) + pow((p.y + 2 * bestSideLength), 2) <= pow(RADIUS, 2))//(2,-2) 0000 0001
		return 1;
	else if (pow((p.x - 4 * bestSideLength), 2) + pow((p.y + 2 * bestSideLength), 2) <= pow(RADIUS, 2))//(4, -2)0000 0010
		return 2;
	else if (pow((p.x - 6 * bestSideLength), 2) + pow((p.y + 2 * bestSideLength), 2) <= pow(RADIUS, 2))//(6, -2)0000 0100
		return 4;
	else if (pow((p.x - 2 * bestSideLength), 2) + pow(p.y, 2) <= pow(RADIUS, 2))//(2,0)0000 1000
		return 8;
	else if (pow((p.x - 6 * bestSideLength), 2) + pow(p.y, 2) <= pow(RADIUS, 2))//(6,0)0001 0000
		return 16;
	else if (pow((p.x - 2 * bestSideLength), 2) + pow(p.y - 2 * bestSideLength, 2) <= pow(RADIUS, 2))//(2,2)0010 0000
		return 32;
	else if (pow((p.x - 4 * bestSideLength), 2) + pow(p.y - 2 * bestSideLength, 2) <= pow(RADIUS, 2))//(4,2)0100 0000
		return 64;
	else if (pow((p.x - 6 * bestSideLength), 2) + pow(p.y - 2 * bestSideLength, 2) <= pow(RADIUS, 2))//(6,2)1000 0000
		return 128;
	else if (pow((p.x - 4 * bestSideLength), 2) + pow(p.y, 2) <= pow(RADIUS, 2))
		return -1;
	else
		return 0;
}

//计算斜率
float slopeCalculation(Point p1, Point p2)
{
	float k;
	if (p2.x != p1.x)
	{
		k = float(float(p2.y) - float(p1.y)) / (float(p2.x) - float(p1.x));
		return k;
	}
	else
		return 9999.9999;
}
