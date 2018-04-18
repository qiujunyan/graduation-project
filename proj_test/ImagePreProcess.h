#pragma once

//#include "stdafx.h"
#include "HeadFile.h"

using namespace cv;
using namespace std;

class ImagePreprocess
{
private:
	int const m_nExtSize = 8;															//2*extSize=局部极大值判断窗口大小
	int const m_nDistance = 75;														//cluster时类别之间最大距离
	int m_nClusterCount = 0;													//cluster分类数
	int const m_nThreshold = 200;

private:
	IplImage * scale_img(IplImage *src, float fScale);							//缩放图像
	IplImage* cal_integ(IplImage *ScaleImage);									//图像积分
	IplImage* cs2(IplImage *CalIntegImage, int wndSize);						//显著性图计算2
	//bool cal_local_max(IplImage *CSImage, int x, int y);
	int *max_val_coord(IplImage *CSImage);										//选出前n个局部极大值像素点的坐标
	float cal_distance(IplImage *CSImage, int p1, int p2);						//距离求取函数
	vector<vector<Point>> cluster(IplImage *CSImage, int *coord);				//根据距离聚类
	vector<vector<Point>> draw_grid(IplImage *CSImage, vector<vector<Point>>point);
	IplImage* otsu_fast(IplImage *clusterImage, vector<vector<Point>>point);	//大律法

public:
	IplImage* image_preprocess(IplImage* src, float fscale=0.5, int wnd_size=1);
};

