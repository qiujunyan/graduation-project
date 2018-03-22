# graduation-project
// prj1.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include "iostream"
#include <fstream>  
#include "time.h"
#include "math.h"

using namespace cv;
using namespace std;

class ImagePreprocess
{
private:
	int m_nExtSize = 10;														//2*extSize=局部极大值判断窗口大小
	int m_nDistance = 80;														//cluster时类别之间最大距离
	int m_nClusterCount = 0;													//cluster分类数
	int const m_nThreshold = 100;
public:
	IplImage* ScaleImg(IplImage *src, float fScale);						//缩放图像
	IplImage* CS2(IplImage *CalIntegImage, int wndSize);						//显著性图计算2
	bool CalLocalMax(IplImage *CSImage, int rows, int columns);					//局部极大值判断
	int *MaxValCoord(IplImage *CSImage);										//选出前n个局部极大值像素点的坐标
	float CalDistance(IplImage *CSImage, int p1, int p2);						//距离求取函数
	vector<vector<Point>> cluster(IplImage *CSImage, int *coord);				//根据距离聚类
	vector<vector<Point>> drawGrid(IplImage *CSImage, vector<vector<Point>>point);
	IplImage* Otsu_fast(IplImage *clusterImage, vector<vector<Point>>point);		//大律法

};

IplImage* ImagePreprocess::ScaleImg(IplImage *src, float fScale)
{
	IplImage *dst = NULL;
	CvSize dstSize;																//Cvsize是矩形框的大小，单位是像素

	dstSize.width = int(src->width*fScale);
	dstSize.height = int(src->height*fScale);
	dst = cvCreateImage(dstSize, src->depth, src->nChannels);
	cvResize(src, dst, CV_INTER_NN);
	return dst;

	cvReleaseImage(&dst);
	cvReleaseImage(&src);
}


IplImage* ImagePreprocess::CS2(IplImage *ScaleImage, int wndSize)												//边缘检测
{
	int w = ScaleImage->width;
	int h = ScaleImage->height;
	int n = w*h;
	double pixel_LR;
	double pixel_UP;
	int num = wndSize*wndSize * 2;

	IplImage *srcGray = cvCreateImage(cvGetSize(ScaleImage), 8, 1);
	cvCvtColor(ScaleImage, srcGray, CV_BGR2GRAY);

	IplImage *dst = cvCreateImage(cvGetSize(ScaleImage), 8, 1);
	int *srcPixel = new int[n];
	CvScalar *dstPixel = new CvScalar[n];

	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
			srcPixel[i*w + j] = ((uchar *)(srcGray->imageData + i*srcGray->widthStep))[j];


	for (int i = wndSize; i<(h - wndSize); i++)
		for (int j = wndSize; j < (w - wndSize); j++)
		{
			int pos_i = i + wndSize;
			int pos_j = j + wndSize;
			int pos_ii = i - wndSize;
			int pos_jj = j - wndSize;
			pixel_LR = abs(srcPixel[pos_i*w + pos_j] - 2 * srcPixel[pos_i*w + j] + srcPixel[pos_ii*w + pos_j]
				- (srcPixel[pos_ii*w + pos_j] - srcPixel[pos_ii*w + pos_jj])) / num;
			pixel_UP = abs(srcPixel[pos_i*w + pos_j] - 2 * srcPixel[i*w + pos_j] + srcPixel[pos_ii*w + pos_j]
				- (srcPixel[pos_i*w + pos_jj] - srcPixel[pos_ii*w + pos_jj])) / num;

			dstPixel[i*w + j].val[0] = 10 * (((pixel_LR > pixel_UP) ? pixel_LR : pixel_UP));
			cvSet2D(dst, i, j, dstPixel[i*w + j]);
		}



	wndSize = wndSize;
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
			if (i <= wndSize || i >= h - wndSize || j <= wndSize || j >= w - wndSize)
				cvSet2D(dst, i, j, cvScalar(0, 0, 0));
	}
		

	return dst;
	delete[] srcPixel;
	delete[] dstPixel;
	cvReleaseImage(&dst);
}


bool ImagePreprocess::CalLocalMax(IplImage *CSImage, int x, int y)		//x,y表示要判断坐标点的横、纵坐标;窗口大小为(2*extSize+1)*(2*extSize+1)
{
	int w = CSImage->width;
	int h = CSImage->height;

	int x_begin = (x > m_nExtSize ? x - m_nExtSize : 0);
	int x_end = (x + m_nExtSize < h ? x + m_nExtSize : h - 1);
	int y_begin = (y > m_nExtSize ? y - m_nExtSize : 0);
	int y_end = (y + m_nExtSize < w ? m_nExtSize + y : w - 1);
	int wndSize = (x_end - x_begin + 1)*(y_end - y_begin + 1);					//窗口像素点数

	bool State;
	int Count = 0;
	int pixel_xy = ((uchar *)(CSImage->imageData + x*CSImage->widthStep))[y];
	int *pixel = new int[wndSize];

	if (pixel_xy < m_nThreshold)
		State = false;
	else
	{
		for (int i = x_begin; i <= x_end; i++)									//考虑到像素点在图像边缘
			for (int j = y_begin; j <= y_end; j++)
			{
				pixel[Count] = ((uchar *)(CSImage->imageData + i*CSImage->widthStep))[j];
				if (pixel_xy < pixel[Count])
					break;
				Count++;
			}
		if (Count >= wndSize)
			State = true;
		else
			State = false;
	}
	return State;
	delete[]pixel;
}

int *ImagePreprocess::MaxValCoord(IplImage *CSImage)
{
	int w = CSImage->width;
	int h = CSImage->height;
	int *coord = new int[w*h];													//存储局部极大值坐标

	coord[0] = 0;
	int num = 1;
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			if (CalLocalMax(CSImage, i, j))
			{
				coord[num] = i*w + j;
				num++;
				//cvSet2D(CSImage, i, j, cvScalar(255, 255, 255));
			}
			else
				cvSet2D(CSImage, i, j, cvScalar(0, 0, 0));
		}
	}
	coord[0] = num;
	return coord;
	delete[]coord;
}

float ImagePreprocess::CalDistance(IplImage *CSImage, int p1, int p2)
{
	int w = CSImage->width;
	int p1_y = p1%w;
	int p1_x = p1 / w;
	int p2_y = p2%w;
	int p2_x = p2 / w;
	float distance;
	distance = sqrt(float(pow(p1_x - p2_x, 2) + pow(p1_y - p2_y, 2)));

	return distance;
}

vector<vector<Point>> ImagePreprocess::cluster(IplImage *CSImage, int *coord)
{
	int sampleCount = coord[0];
	m_nClusterCount = sampleCount;

	int w = CSImage->width;
	int h = CSImage->height;

	vector<Point> pointTemp(sampleCount);
	vector<vector<Point>> points(20, vector<Point>(sampleCount));//数组所占空间越大，耗时越长
	vector<Point>centerPoint(sampleCount);

	int distance;
	int clusterNum = int(sampleCount / 10);

	int nDistance;
	int label = 0;
	for (int i = 1; i < sampleCount; i++)
	{
		int countTemp = 2;
		if (coord[i] != -1)
		{
			points[label][1].x = coord[i] / w;
			points[label][1].y = coord[i] % w;
			centerPoint[label].x = coord[i] / w;
			centerPoint[label].y = coord[i] % w;
			for (int j = i + 1; j < sampleCount; j++)
			{
				if (coord[j] != -1)
				{
					nDistance = CalDistance(CSImage, centerPoint[label].x*w + centerPoint[label].y, coord[j]);//j+1?
					if (nDistance < m_nDistance)
					{
						points[label][countTemp].x = coord[j] / w;
						points[label][countTemp].y = coord[j] % w;
						centerPoint[label].x = int((centerPoint[label].x* (countTemp - 1) + coord[j] / w) / countTemp);	//更新点集中心
						centerPoint[label].y = int((centerPoint[label].y* (countTemp - 1) + coord[j] % w) / countTemp);
						countTemp++;
						coord[j] = -1;
						m_nClusterCount--;
					}
				}
			}
			points[label][0].x = countTemp;
			if (points[label][0].x < ((clusterNum > 5) ? clusterNum : 5))//筛除点数较小的点集
			{
				centerPoint[label].x = -1;
				centerPoint[label].y = -1;
				m_nClusterCount--;
			}
			else
				label++;
		}
	}
	cout << "clusterCount:" << m_nClusterCount << endl;
	//for(int label=0;label<m_nClusterCount;label++)

	return points;
	centerPoint.clear();
	points.clear();
}

vector<vector<Point>> ImagePreprocess::drawGrid(IplImage *CSImage, vector<vector<Point>>point)
{
	Mat srcMat;
	srcMat = cvarrToMat(CSImage);
	vector<vector<Point>>fPoint(m_nClusterCount, vector<Point>(4));				//找出四个角点
  	for (int label = 0; label < m_nClusterCount; label++)
	{
		int x_min = point[label][1].x;
		int x_max = point[label][1].x;
		int y_min = point[label][1].y;
		int y_max = point[label][1].y;
		for (int i = 2; i < point[label][0].x; i++)								// point[label][0].x代表label类别的点集的数量
		{
			if (point[label][i].x < x_min&&point[label][i].x != 0)
				x_min = point[label][i].x;
			else if (point[label][i].x > x_max)
				x_max = point[label][i].x;

			if (point[label][i].y < y_min&&point[label][i].y != 0)
				y_min = point[label][i].y;
			else if (point[label][i].y > y_max)
				y_max = point[label][i].y;
		}
		fPoint[label][0].y = x_min;
		fPoint[label][0].x = y_min;

		fPoint[label][1].y = x_max;
		fPoint[label][1].x = y_max;

		fPoint[label][2].y = x_min;
		fPoint[label][2].x = y_max;

		fPoint[label][3].y = x_max;
		fPoint[label][3].x = y_min;
		//cout << label << ":[" << fPoint[label][0].x << "," << fPoint[label][0].y << "],[" << fPoint[label][1].x << "," << fPoint[label][1].y << "]" << endl;
		line(srcMat, fPoint[label][0], fPoint[label][2], Scalar(255, 255, 255), 1, 16);
		line(srcMat, fPoint[label][0], fPoint[label][3], Scalar(255, 255, 255), 1, 16);
		line(srcMat, fPoint[label][1], fPoint[label][2], Scalar(255, 255, 255), 1, 16);
		line(srcMat, fPoint[label][1], fPoint[label][3], Scalar(255, 255, 255), 1, 16);
	}

	imshow("ImagePreprocess", srcMat);
	return fPoint;
	fPoint.clear();
}




IplImage* ImagePreprocess::Otsu_fast(IplImage *ScaleImage, vector<vector<Point>>point)
{
	//图像转化成灰度图                                                                            
	IplImage *grayImg = cvCreateImage(cvSize(ScaleImage->width, ScaleImage->height), IPL_DEPTH_8U, 1);
	cvCvtColor(ScaleImage, grayImg, CV_BGR2GRAY);
	float fMax = 0.0;
	int pixelCount[256];
	float pixelPro[256];
	int label, i, j;

	//初始化
	for (i = 0; i < 256; i++)
	{
		pixelCount[i] = 0;
		pixelPro[i] = 0;
	}

	for (label = 0; label < m_nClusterCount; label++)
	{

		int nThreshold = 0;

		for (j = point[label][0].x; j < point[label][1].x; j++)
			for (i = point[label][0].y; i < point[label][1].y; i++)
			{
				float pixel = float(((uchar *)(grayImg->imageData + i*grayImg->widthStep))[j]);
				if (fMax < pixel)
					fMax = pixel;
			}


		for (j = point[label][0].x; j < point[label][1].x; j++)
			for (i = point[label][0].y; i < point[label][1].y; i++)
				((uchar *)(grayImg->imageData + i*grayImg->widthStep))[j]
				= 255 * ((uchar *)(grayImg->imageData + i*grayImg->widthStep))[j] / fMax;

		//统计灰度级中每个像素在ROI中的个数  
		for (j = point[label][0].x; j < point[label][1].x; j++)
			for (i = point[label][0].y; i < point[label][1].y; i++)
			{
				int pixel = ((uchar *)(grayImg->imageData + i*grayImg->widthStep))[j];
				pixelCount[pixel]++;
			}



		//计算每个灰度级在ROI中的比例 
		for (i = 0; i < 256; i++)
		{
			int pixelSum = (point[label][1].x - point[label][0].x)*(point[label][1].y - point[label][0].y);
			pixelPro[i] = (float)(pixelCount[i]) / (float)(pixelSum);
		}

		int n = 0;
		while (n < 2)
		{
			int begin = nThreshold;
			//大律法求阈值
			float w0, w1, u0tmp, u1tmp, u0, u1, u, deltaTmp, deltaMax = 0;
			for (i = begin; i < 256; i++)
			{
				w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = 0;
				for (j = begin; j < 256; j++)
				{
					if (j <= i) //背景部分  
					{
						w0 += pixelPro[j];
						u0tmp += j * pixelPro[j];
					}
					else       //前景部分  
					{
						w1 += pixelPro[j];
						u1tmp += j * pixelPro[j];
					}
					u0 = u0tmp / w0;        //第一类的平均灰度  
					u1 = u1tmp / w1;        //第二类的平均灰度  
					u = u0tmp + u1tmp;      //ROI的平均灰度  
											//计算类间方差  
					deltaTmp = w0 * (u0 - u)*(u0 - u) + w1 * (u1 - u)*(u1 - u);
					//找出最大类间方差以及对应的阈值  
					if (deltaTmp > deltaMax)
					{
						deltaMax = deltaTmp;
						nThreshold = i;
					}
				}
			}
			n++;
		}


		//二值化
		for (j = point[label][0].x; j < point[label][1].x; j++)
			for (i = point[label][0].y; i < point[label][1].y; i++)
			{
				if (((uchar *)(grayImg->imageData + i*grayImg->widthStep))[j] > nThreshold)
					cvSet2D(grayImg, i, j, cvScalar(255, 255, 255));
				else
					cvSet2D(grayImg, i, j, cvScalar(0, 0, 0));
			}

		bool Ctrl=false;
		if (Ctrl)
		{
			//设置ROI
			CvRect rect;
			rect.x = point[label][0].x;
			rect.y = point[label][0].y;
			rect.width = (point[label][1].x - point[label][0].x);
			rect.height = (point[label][1].y - point[label][0].y);
			cvSetImageROI(grayImg, rect);

			//提取目标图像轮廓
			int mode = CV_RETR_TREE;//设置提取轮廓模式
			CvMemStorage *storage = cvCreateMemStorage(0);
			CvSeq *contours = 0;//存储提取的轮廓图像
			cvFindContours(grayImg, storage, &contours, sizeof(CvContour), mode, CV_CHAIN_APPROX_NONE);
			for (; contours != 0; contours = contours->h_next)
			{
				cvDrawContours(grayImg, contours, cvScalar(255, 255, 255), cvScalar(255, 255, 255), -1, 1, 8);
			}

			//绘制最小外接矩形
		}
		cvResetImageROI(grayImg);
	}


	cvShowImage("binaryImage", grayImg);
	cvReleaseImage(&grayImg);
	return grayImg;
}


int main()
{
	time_t begin, end;
	time_t begin1, end1;
	time_t begin2, end2, begin6, end6;
	time_t begin3, end3;
	time_t begin4, end4;
	time_t begin5, end5;
	begin = clock();


	ImagePreprocess myClass;
	IplImage *src = cvLoadImage("F:\\毕业设计\\毕业设计\\图像\\quan\\c20.jpg");
	vector<vector<Point>>point(100, vector<Point>(100));
	IplImage *ScaleImage;
	IplImage *CS2Image;
	IplImage *CalIntegImage;
	IplImage *binaryImage;
	int *coord = new int[1000];

	begin1 = clock();
	ScaleImage = myClass.ScaleImg(src, 0.5);
	end1 = clock();
	cout << "clock1缩放图像:" << end1 - begin1 << endl;

	begin2 = clock();
	CS2Image = myClass.CS2(ScaleImage, 2);
	end2 = clock();
	cout << "clock2显著性图:" << end2 - begin2 << endl;
	cvShowImage("CS2Image", CS2Image);

	begin3 = clock();
	coord = myClass.MaxValCoord(CS2Image);
	end3 = clock();
	cout << "clock3局部极大值抑制:" << end3 - begin3 << endl;

	begin4 = clock();
	point = myClass.cluster(CS2Image, coord);
	end4 = clock();
	cout << "clock4聚类:" << end4 - begin4 << endl;

	begin5 = clock();
	vector<vector<Point>>angularPoint = myClass.drawGrid(CS2Image, point);
	end5 = clock();
	cout << "clock5画图:" << end5 - begin5 << endl;


	begin6 = clock();
	myClass.Otsu_fast(ScaleImage, angularPoint);
	end6 = clock();
	cout << "clock6:大律法" << end6 - begin6 << endl;

	end = clock();

	cout << "clock:" << end - begin << endl;

	cvWaitKey();


	return 0;
}


