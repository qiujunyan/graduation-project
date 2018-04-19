#pragma once
#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include "math.h"

IplImage* ImagePreprocess_01(IplImage* src, float fscale = 1, int wnd_size = 2);
float cal_distance(IplImage *image, int p1, int p2);

#define		LEN_RECT	30		// 编码板上长条的实际长度，厘米
#define		DIS_CC		24		// 编码板上长条中心距九宫格中心的距离，厘米
#define		DIS_SQUARE	12		// 编码板方块中心间距
#define		THRESH_BW	180
#define		RADIUS		4.0


float cal_distance(IplImage *image, int p1, int p2)
{
	int w = image->width;
	int p1_y = p1 % w;
	int p1_x = p1 / w;
	int p2_y = p2 % w;
	int p2_x = p2 / w;
	float distance;
	distance = sqrt(float(pow(p1_x - p2_x, 2) + pow(p1_y - p2_y, 2)));

	return distance;
}

int coord_calculation(Point2f p, float side_length)
{
	//数据库，记录编码块坐标
	//(2,-2)、(4,-2)、(6,-2)
	//(2,0)、(4,0)、(6,0)
	//(2,2)、(4,2)、(6,2)
	if (pow((p.x - 2 * side_length), 2) + pow((p.y + 2 * side_length), 2) <= pow(RADIUS, 2))//(2,-2) 0000 0001
		return 1;
	else if (pow((p.x - 4 * side_length), 2) + pow((p.y + 2 * side_length), 2) <= pow(1.5*RADIUS, 2))//(4, -2)0000 0010
		return 2;
	else if (pow((p.x - 6 * side_length), 2) + pow((p.y + 2 * side_length), 2) <= pow(2*RADIUS, 2))//(6, -2)0000 0100
		return 4;
	else if (pow((p.x - 2 * side_length), 2) + pow(p.y, 2) <= pow(RADIUS, 2))//(2,0)0000 1000
		return 8;
	else if (pow((p.x - 6 * side_length), 2) + pow(p.y, 2) <= pow(2*RADIUS, 2))//(6,0)0001 0000
		return 16;
	else if (pow((p.x - 2 * side_length), 2) + pow(p.y - 2 * side_length, 2) <= pow(RADIUS, 2))//(2,2)0010 0000
		return 32;
	else if (pow((p.x - 4 * side_length), 2) + pow(p.y - 2 * side_length, 2) <= pow(1.5*RADIUS, 2))//(4,2)0100 0000
		return 64;
	else if (pow((p.x - 6 * side_length), 2) + pow(p.y - 2 * side_length, 2) <= pow(2*RADIUS, 2))//(6,2)1000 0000
		return 128;
	else if (pow((p.x - 4 * side_length), 2) + pow(p.y, 2) <= pow(1.5*RADIUS, 2))
		return -1;
	else
		return 0;
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
		for (i = 0; i < rows1; i++)
		{
			for (j = 0; j < cols2; j++)
			{
				for (k = 0; k < cols1; k++)
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

// 矩阵减法
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
		for (i = 0; i < rows1; i++)
			for (j = 0; j < cols1; j++)
				v[i][j] = v1[i][j] - v2[i][j];
		return v;
	}
}

//坐标转换
Point2f convertion(Point2f point, Point2f original_point, float alpha)//[u,v]=[x-x0,y-y0]*([cosx,sinx],[-sinx,cosx])
{
	Point2f pointAfterConvertion;
	alpha = alpha * CV_PI / 180;
	vector<vector<float>> T(1, vector<float>(2, 0)), coord(1, vector<float>(2, 0)), coordAfterConvertion(1, vector<float>(2, 0));
	vector<vector<float>> R(2, vector<float>(2, 0));
	T = { { original_point.x,original_point.y } };
	coord = { { point.x,point.y } };
	R = { { cos(alpha),sin(alpha) },{ -sin(alpha),cos(alpha) } };
	coordAfterConvertion = multiple(vectorMinus(coord, T), R);
	pointAfterConvertion.x = coordAfterConvertion[0][0];
	pointAfterConvertion.y = coordAfterConvertion[0][1];
	return pointAfterConvertion;
}

IplImage* ImagePreprocess_01(IplImage* src, float fscale, int wnd_size)
{
	// 01:缩放图像
	IplImage *scale_image = NULL;
	CvSize dstSize;

	dstSize.width = int(src->width*fscale);
	dstSize.height = int(src->height*fscale);
	scale_image = cvCreateImage(dstSize, src->depth, src->nChannels);
	cvResize(src, scale_image, CV_INTER_NN);

	// 02:图像积分
	IplImage *gray_image = cvCreateImage(cvGetSize(scale_image), 8, 1);
	IplImage *integ_image = cvCreateImage(cvGetSize(scale_image), IPL_DEPTH_32F, 1);
	cvCvtColor(scale_image, gray_image, CV_BGR2GRAY);

	int nw = scale_image->width;
	int nh = scale_image->height;
	int *src_pixel = new int[nw*nh];
	int *dst_pixel = new int[nw*nh];

	for (int i = 0; i < nh; i++)
		for (int j = 0; j < nw; j++)
			src_pixel[i*nw + j] = ((uchar*)(gray_image->imageData + gray_image->widthStep*i))[j];

	int *columnSum = new int[nw];
	columnSum[0] = src_pixel[0];
	dst_pixel[0] = columnSum[0];
	((float*)(integ_image->imageData + 0 * integ_image->widthStep))[0] = float(dst_pixel[0]);

	//第一行的像素值积分
	for (int j = 1; j < nw; j++)
	{
		columnSum[j] = src_pixel[j];
		dst_pixel[j] = columnSum[j];
		dst_pixel[j] += dst_pixel[j - 1];
		((float*)(integ_image->imageData + 0 * integ_image->widthStep))[j] = float(dst_pixel[0 * nw + j]);
	}

	for (int i = 1; i < nh; i++)
	{
		//第一列像素积分
		columnSum[0] += src_pixel[i*nw];
		dst_pixel[i*nw] = columnSum[0];
		((float*)(integ_image->imageData + i * integ_image->widthStep))[0] = float(dst_pixel[i * nw + 0]);
		//其他像素值积分
		for (int j = 1; j < nw; j++)
		{
			columnSum[j] += src_pixel[i*nw + j];
			dst_pixel[i*nw + j] = dst_pixel[i*nw + j - 1] + columnSum[j];
			((float*)(integ_image->imageData + integ_image->widthStep * i))[j] = float(dst_pixel[i * nw + j]);
		}
	}

	// 03: 显著性图
	IplImage *cs_image = cvCreateImage(cvGetSize(integ_image), 8, 1);
	int num = wnd_size * wnd_size * 2;

	for (int i = 0; i < nh; i++)
		for (int j = 0; j < nw; j++)
			src_pixel[i*nw + j] = int(((float *)(integ_image->imageData + i * integ_image->widthStep))[j]);

	for (int i = wnd_size + 1; i < (nh - wnd_size); i++)
	{
		for (int j = wnd_size + 1; j < (nw - wnd_size); j++)
		{
			int pos_i = i + wnd_size;
			int pos_j = j + wnd_size;
			int pos_ii = i - wnd_size;
			int pos_jj = j - wnd_size;
			int pixel_LR = int(abs(src_pixel[pos_i*nw + pos_j] - src_pixel[pos_i*nw + j] - (src_pixel[pos_i*nw + j - 1] - src_pixel[pos_i*nw + pos_jj - 1])
				- (src_pixel[(pos_ii - 1)*nw + pos_j] - src_pixel[(pos_ii - 1)*nw + j] - (src_pixel[(pos_ii - 1)*nw + j - 1] - src_pixel[(pos_ii - 1)*nw + pos_jj - 1]))) / num);
			int pixel_UP = int(abs(src_pixel[pos_i*nw + pos_j] - src_pixel[i*nw + pos_j] - (src_pixel[(i - 1)*nw + pos_j] - src_pixel[(pos_ii - 1)*nw + pos_j])
				- (src_pixel[pos_i*nw + (pos_jj - 1)] - src_pixel[i*nw + (pos_jj - 1)] - (src_pixel[(i - 1)*nw + (pos_jj - 1)] - src_pixel[(pos_ii - 1)*nw + (pos_jj - 1)]))) / num);

			dst_pixel[i*nw + j] = 2 * (((pixel_LR > pixel_UP) ? pixel_LR : pixel_UP));
			if (dst_pixel[i*nw + j] >= 255)
				dst_pixel[i*nw + j] = 255;
			((uchar*)(cs_image->imageData + i * cs_image->widthStep))[j] = dst_pixel[i*nw + j];
		}
	}

	for (int i = 0; i < nh; i++)
	{
		for (int j = 0; j < nw; j++)
			if (i <= wnd_size || i >= nh - wnd_size || j <= wnd_size || j >= nw - wnd_size)
				((uchar*)(cs_image->imageData + i * cs_image->widthStep))[j] = 0;
	}

	// 04: 非局部极大值抑制
	int m_nExtSize = 8;
	int m_nThreshold = 200;
	int *coord = new int[nw*nh];

	for (int i = 0; i<nh; i++)
		for (int j = 0; j < nw; j++)
			src_pixel[i*nw + j] = ((uchar*)(cs_image->imageData + cs_image->widthStep*i))[j];

	for (int i = 0; i < nh; i++)
	{
		for (int j = 0; j < nw; j++)
		{
			if (i<m_nExtSize || i>nh - m_nExtSize || j<m_nExtSize || j>nw - m_nExtSize)
				continue;
			bool state = true;
			if (src_pixel[i*nw + j] < m_nThreshold)
			{
				state = false;
				continue;
			}
			for (int ii = i - m_nExtSize; ii < i + m_nExtSize; ii++)
			{
				for (int jj = j - m_nExtSize; jj < j + m_nExtSize; jj++)
				{
					if (ii == i && jj == j)
						continue;
					if (src_pixel[i*nw + j] < src_pixel[ii*nw + jj])
					{
						state = false;
						break;
					}
				}
				if (!state)
					break;
			}
			if (state)
			{
				coord[num] = i * nw + j;
				num++;
			}
		}
	}
	coord[0] = num;

	// 05: 聚类
	int sample_count = coord[0];
	int m_nClusterCount = sample_count;
	int const m_nDistance = 75;

	vector<Point> pointTemp(sample_count);
	vector<vector<Point>> points(20, vector<Point>(sample_count));//数组所占空间越大，耗时越长
	vector<Point>centerPoint(sample_count);

	int clusterNum = int(sample_count / 10);

	int nDistance;
	int label = 0;
	for (int i = 1; i < sample_count; i++)
	{
		int countTemp = 2;
		if (coord[i] != -1)
		{
			points[label][1].x = coord[i] / nw;
			points[label][1].y = coord[i] % nw;
			centerPoint[label].x = coord[i] / nw;
			centerPoint[label].y = coord[i] % nw;
			for (int j = i + 1; j < sample_count; j++)
			{
				if (coord[j] != -1)
				{
					nDistance = int(cal_distance(cs_image, centerPoint[label].x*nw + centerPoint[label].y, coord[j]));//j+1?
					if (nDistance < m_nDistance)
					{
						points[label][countTemp].x = coord[j] / nw;
						points[label][countTemp].y = coord[j] % nw;
						centerPoint[label].x = int((centerPoint[label].x* (countTemp - 1) + coord[j] / nw) / countTemp);	//更新点集中心
						centerPoint[label].y = int((centerPoint[label].y* (countTemp - 1) + coord[j] % nw) / countTemp);
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
	m_nClusterCount--;

	pointTemp.clear();
	centerPoint.clear();

	// 06: 找出点集最小外接矩形
	Mat srcMat;
	srcMat = cvarrToMat(cs_image);
	vector<vector<Point>>fPoint(m_nClusterCount, vector<Point>(4));				//找出四个角点
	int delta = 2;
	for (int label = 0; label < m_nClusterCount; label++)
	{
		int delta = 0;
		int x_min = points[label][1].x - delta;
		int x_max = points[label][1].x + delta;
		int y_min = points[label][1].y - delta;
		int y_max = points[label][1].y + delta;
		for (int i = 2; i < points[label][0].x; i++)								// point[label][0].x代表label类别的点集的数量
		{
			if (points[label][i].x < x_min&&points[label][i].x != 0)
				x_min = points[label][i].x;
			else if (points[label][i].x > x_max)
				x_max = points[label][i].x;

			if (points[label][i].y < y_min&&points[label][i].y != 0)
				y_min = points[label][i].y;
			else if (points[label][i].y > y_max)
				y_max = points[label][i].y;
		}
		fPoint[label][0].y = x_min;
		fPoint[label][0].x = y_min;

		fPoint[label][1].y = x_min;
		fPoint[label][1].x = y_max;

		fPoint[label][2].y = x_max;
		fPoint[label][2].x = y_max;

		fPoint[label][3].y = x_max;
		fPoint[label][3].x = y_min;
		//cout << label << ":[" << fPoint[label][0].x << "," << fPoint[label][0].y << "],[" << fPoint[label][1].x << "," << fPoint[label][1].y << "]" << endl;
		for (int i = 0; i<4; i++)
			line(srcMat, fPoint[label][i], fPoint[label][(i + 1) % 4], Scalar(255, 255, 255), 1, 16);
	}

	// 07: 大律法
	cvCvtColor(scale_image, gray_image, CV_BGR2GRAY);
	Mat grayImgMat = cvarrToMat(gray_image);
	int i, j;

	for (label = 0; label < m_nClusterCount; label++)
	{
		//07_01:设置ROI及图片预处理
		Rect rect(fPoint[label][0].x, fPoint[label][0].y, fPoint[label][2].x - fPoint[label][0].x, fPoint[label][2].y - fPoint[label][0].y);
		Mat grayImgROI(grayImgMat, rect);
		int size = grayImgROI.rows*grayImgROI.cols;
		int count[256] = { 0 }, pixel, maxPixel = 0;
		for (i = 0; i < grayImgROI.rows; i++)
		{
			uchar *p = grayImgROI.ptr<uchar>(i);
			for (j = 0; j < grayImgROI.cols; j++)
				if (p[j] > maxPixel)
					maxPixel = p[j];
		}

		for (i = 0; i < grayImgROI.rows; i++)
		{
			uchar *p = grayImgROI.ptr<uchar>(i);
			for (j = 0; j < grayImgROI.cols; j++)
			{
				p[j] = int(pow(p[j], 1) / float(pow(maxPixel, 0)));
				pixel = p[j];
				count[pixel]++;
			}
		}


		double mu = 0, scale = 1. / (size);//mu为图像灰度平均值
		for (i = 0; i < 256; i++)
			mu += i * (double)count[i];
		mu *= scale;
		double mu1 = 0, q1 = 0;
		double max_sigma = 0, max_val = 0;
		for (i = 0; i < 256; i++)
		{
			double p_i, q2, mu2, sigma;

			p_i = count[i] * scale;
			mu1 *= q1;
			q1 += p_i;
			q2 = 1. - q1;

			if (std::min(q1, q2) < FLT_EPSILON || std::max(q1, q2) > 1. - FLT_EPSILON)
				continue;

			mu1 = (mu1 + i * p_i) / q1;
			mu2 = (mu - q1 * mu1) / q2;
			sigma = q1 * q2*(mu1 - mu2)*(mu1 - mu2);
			if (sigma > max_sigma)
			{
				max_sigma = sigma;
				max_val = i;
			}
		}
		int nThreshold = int(max_val);

		//07_02:腐蚀膨胀
		int kernelSize = 3;
		Mat element = getStructuringElement(MORPH_RECT, Size(kernelSize, kernelSize));
		Mat grayImgROI_cp = grayImgROI.clone();
		erode(grayImgROI_cp, grayImgROI_cp, element);
		dilate(grayImgROI_cp, grayImgROI_cp, element);
		threshold(grayImgROI_cp, grayImgROI_cp, nThreshold, 255, 0);

		//07_03:提取目标图像轮廓
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(grayImgROI_cp, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

		//07_04:绘制最小外接矩形
		bool state = false;
		grayImgROI_cp = Mat::zeros(rect.size(), CV_8UC1);
		vector<vector<Point>> rectangle;//可能的长方形的轮廓
		vector<vector<Point>> code_block;//九宫格每个方块的轮廓

		for (i = 0; i < contours.size(); i++)
		{
			vector<RotatedRect> box(contours.size());
			Point2f rect2[4];
			box[i] = minAreaRect(Mat(contours[i]));  //计算每个轮廓最小外接矩形
			box[i].points(rect2);  //把最小外接矩形四个端点复制给rect数组
			
			int l = (int)((box[i].size.height > box[i].size.width) ? box[i].size.height : box[i].size.width);
			int s = (int)((box[i].size.height < box[i].size.width) ? box[i].size.height : box[i].size.width);
			int area = l * s;
			if (float(l) / s >= 2)
			{
				if (area < 200)
					continue;
				state = true;
				rectangle.push_back(contours[i]);
				for (int j = 0; j<4; j++)
					line(grayImgROI_cp, rect2[j], rect2[(j + 1) % 4], Scalar(255), 1, 4);  //绘制最小外接矩形每条边
			}

			else if (float(l / s) == 1 && area >= 40)
			{
				code_block.push_back(contours[i]);
				for (int j = 0; j<4; j++)
					line(grayImgROI_cp, rect2[j], rect2[(j + 1) % 4], Scalar(255), 1, 4);  //绘制最小外接矩形每条边
			}

		}

		if (!state|| int(code_block.size())<=3)
			continue;
		grayImgROI_cp.copyTo(grayImgROI);//clone会重新分配内存地址，而copyTo不会


		 //07_05:得到九宫格每个小块的中心位置
		vector<Point2f> code_centers;
		for (int q = 0; q < code_block.size(); q++)
		{
			RotatedRect box = minAreaRect(Mat(code_block[q]));
			float fx = box.center.x;
			float fy = box.center.y;
			Point2f p(fx, fy);
			code_centers.push_back(p);
		}

		//07_06:找出矩形定位模块以及九宫格的位置
		Point2f rectangle_center; //矩形中心点的坐标
		for (int q = 0; q<int(rectangle.size()); q++)
		{
			RotatedRect box;
			box = minAreaRect(Mat(rectangle[q]));
			float angle = abs(box.angle);
			float length = (int)((box.size.height > box.size.width) ? box.size.height : box.size.width);;
			int x = (int)(box.center.x);
			int y = (int)(box.center.y);

			int ratio = 12;				// 长条中心距九宫格中心的距离，厘米，数值
			float step1 = 0;			//按比例估计长度  长方形中心与九宫格中心之间距离
			float step2 = 0;			//两个九宫格小块之间的距离
			vector<Point> true_p4;
			vector<int> true_state;
			bool flag;

			int rows = grayImgROI_cp.rows;
			int cols = grayImgROI_cp.cols;

			for (ratio = DIS_CC - 2; ratio < DIS_CC + 2; ratio++)//DIS_CC：编码板上长条中心距九宫格中心的距离，厘米
			{
				//判断九宫格在矩形的哪个方向
				step1 = length * ratio / LEN_RECT;				// LEN_RECT:编码板上长条的实际长度，厘米。step1：长条中心距九宫格中心的像素距离
				step2 = DIS_SQUARE * length*ratio / (DIS_CC*LEN_RECT);//DIS_SQUARE:编码板方块中心间距。step2：编码板方块中心像素距离

				vector<Point> p4;
				Point pp;
				pp.x = (int)(x + step1 * cos(angle*CV_PI / 180));
				pp.y = (int)(y - step1 * sin(angle*CV_PI / 180));
				p4.push_back(pp);
				pp.x = (int)(x + step1 * sin(angle*CV_PI / 180));
				pp.y = (int)(y + step1 * cos(angle*CV_PI / 180));
				p4.push_back(pp);
				pp.x = (int)(x - step1 * cos(angle*CV_PI / 180));
				pp.y = (int)(y + step1 * sin(angle*CV_PI / 180));
				p4.push_back(pp);
				pp.x = (int)(x - step1 * sin(angle*CV_PI / 180));
				pp.y = (int)(y - step1 * cos(angle*CV_PI / 180));
				p4.push_back(pp);

				//分别对四个方向进行检测，看是否发光
				vector<int> state;
				for (int i = 0; i < 4; i++)
				{
					int x1 = p4[i].x;
					int y1 = p4[i].y;
					if (x1 < 3 || y1 < 3 || y1 >= rows - 3 || x1 >= cols - 3)
						continue;
					for (int j = 0; j < (int)code_centers.size(); j++)
					{
						int jx = (int)(code_centers[j].x);
						int jy = (int)(code_centers[j].y);
						float dis = (float)(sqrt((x1 - jx)*(x1 - jx) + (y1 - jy)*(y1 - jy)));
						if (dis < step2*0.5)
						{
							p4[i].x = jx;
							p4[i].y = jy;
							state.push_back(i);
							break;
						}
					}
				}

				if (state.size() == 1)
				{
					flag = true;
					true_state.push_back(state[0]);
					for (int k = 0; k < (int)p4.size(); k++)
						true_p4.push_back(p4[k]);
					rectangle_center = box.center;
					break;
				}
				p4.clear();
				state.clear();
			}
			if (!flag)
				continue;

			//旋转角度，将九宫格摆正，长方形在左侧
			float rot = 0;
			if (true_state[0] == 0)
				rot = angle;
			else if (true_state[0] == 1)
				rot = 90 - angle;
			else if (true_state[0] == 2)
				rot = 180 - angle;
			else
				rot = 270 - angle;

			float code_block_length = step2 / 2.0;//编码块的边长
			int id_temp, id = 0;
			Point2f coord_after_convertion;
			for (int q = 0; q < code_block.size(); q++)
			{
				coord_after_convertion = convertion(code_centers[q], rectangle_center, rot);
				id_temp = coord_calculation(coord_after_convertion, code_block_length);
				if (id_temp != -1)
					id += id_temp;
			}
			char ch[20] = {0};
			sprintf(ch,"ID:%d", id);
			string str = ch;
			putText(grayImgMat, str, Point(20, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 4, 1);
			break;
		}

		contours.clear();
		hierarchy.clear();
		rectangle.clear();
		code_block.clear();

	}

	//释放内存
	delete[]src_pixel;
	delete[]dst_pixel;
	delete[]columnSum;
	delete[]coord;
	cvReleaseImage(&scale_image);
	cvReleaseImage(&integ_image);
	cvReleaseImage(&cs_image);
	srcMat.release();
	grayImgMat.release();

	return gray_image;
}


