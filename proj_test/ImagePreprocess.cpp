#include "stdafx.h"
#include "ImagePreprocess.h"

IplImage* ImagePreprocess::scale_img(IplImage *src, float fscale)
{
	IplImage *dst = NULL;
	CvSize dstSize;																//Cvsize是矩形框的大小，单位是像素

	dstSize.width = int(src->width*fscale);
	dstSize.height = int(src->height*fscale);
	dst = cvCreateImage(dstSize, src->depth, src->nChannels);
	cvResize(src, dst, CV_INTER_NN);

	return dst;
}

IplImage* ImagePreprocess::cal_integ(IplImage *ScaleImage)
{
	IplImage *srcGray = cvCreateImage(cvGetSize(ScaleImage), 8, 1);
	IplImage *dst = cvCreateImage(cvGetSize(ScaleImage), IPL_DEPTH_32F, 1);

	cvCvtColor(ScaleImage, srcGray, CV_BGR2GRAY);

	int nw = dst->width;
	int nh = dst->height;
	int *srcPixel = new int[nw*nh];							//原始图像的像素值
	int *dstPixel = new int[nw*nh];							//目标图像的像素值                    

	for (int i = 0; i < nh; i++)
		for (int j = 0; j < nw; j++)
			srcPixel[i*nw + j] = ((uchar*)(srcGray->imageData + srcGray->widthStep*i))[j];

	int *columnSum = new int[nw];								//columnSum[j]表示第j列前n行的像素值积分
	columnSum[0] = srcPixel[0];
	dstPixel[0] = columnSum[0];
	((float*)(dst->imageData + 0 * dst->widthStep))[0] = dstPixel[0];

	//第一行的像素值积分
	for (int j = 1; j < nw; j++)
	{
		columnSum[j] = srcPixel[j];
		dstPixel[j] = columnSum[j];
		dstPixel[j] += dstPixel[j - 1];
		((float*)(dst->imageData + 0 * dst->widthStep))[j] = dstPixel[0 * nw + j];
	}


	for (int i = 1; i < nh; i++)
	{
		//第一列像素积分
		columnSum[0] += srcPixel[i*nw];
		dstPixel[i*nw] = columnSum[0];
		((float*)(dst->imageData + i * dst->widthStep))[0] = dstPixel[i * nw + 0];
		//其他像素值积分
		for (int j = 1; j < nw; j++)
		{
			columnSum[j] += srcPixel[i*nw + j];
			dstPixel[i*nw + j] = dstPixel[i*nw + j - 1] + columnSum[j];
			((float*)(dst->imageData + dst->widthStep * i))[j] = dstPixel[i * nw + j];
		}
	}

	delete[]srcPixel;
	delete[]dstPixel;
	delete[]columnSum;
	cvReleaseImage(&srcGray);
	return dst;
}



IplImage* ImagePreprocess::cs2(IplImage *CalIntgImage, int wndSize)												//边缘检测
{
	int w = CalIntgImage->width;
	int h = CalIntgImage->height;
	int n = w * h;
	int pixel_LR;
	int pixel_UP;
	int num = wndSize * wndSize * 2;

	IplImage *dst = cvCreateImage(cvGetSize(CalIntgImage), 8, 1);
	int *srcPixel = new int[n];
	int *dstPixel = new int[n];

	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
			srcPixel[i*w + j] = ((float *)(CalIntgImage->imageData + i * CalIntgImage->widthStep))[j];


	for (int i = wndSize + 1; i < (h - wndSize); i++)
	{
		for (int j = wndSize + 1; j < (w - wndSize); j++)
		{
			int pos_i = i + wndSize;
			int pos_j = j + wndSize;
			int pos_ii = i - wndSize;
			int pos_jj = j - wndSize;
			pixel_LR = int(abs(srcPixel[pos_i*w + pos_j] - srcPixel[pos_i*w + j] - (srcPixel[pos_i*w + j - 1] - srcPixel[pos_i*w + pos_jj - 1])
				- (srcPixel[(pos_ii - 1)*w + pos_j] - srcPixel[(pos_ii - 1)*w + j] - (srcPixel[(pos_ii - 1)*w + j - 1] - srcPixel[(pos_ii - 1)*w + pos_jj - 1]))) / num);
			pixel_UP = int(abs(srcPixel[pos_i*w + pos_j] - srcPixel[i*w + pos_j] - (srcPixel[(i - 1)*w + pos_j] - srcPixel[(pos_ii - 1)*w + pos_j])
				- (srcPixel[pos_i*w + (pos_jj - 1)] - srcPixel[i*w + (pos_jj - 1)] - (srcPixel[(i - 1)*w + (pos_jj - 1)] - srcPixel[(pos_ii - 1)*w + (pos_jj - 1)]))) / num);

			dstPixel[i*w + j] = 3 * (((pixel_LR > pixel_UP) ? pixel_LR : pixel_UP));
			if (dstPixel[i*w + j] >= 255)
				dstPixel[i*w + j] = 255;
			((uchar*)(dst->imageData + i * dst->widthStep))[j] = dstPixel[i*w + j];
		}
	}


	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
			if (i <= wndSize || i >= h - wndSize || j <= wndSize || j >= w - wndSize)
				((uchar*)(dst->imageData + i * dst->widthStep))[j] = 0;
	}

	delete[] srcPixel;
	delete[] dstPixel;
	return dst;
}

int *ImagePreprocess::max_val_coord(IplImage *CSImage)
{
	int w = CSImage->width;
	int h = CSImage->height;
	int *coord = new int[w*h];													
	int *p = new int[w*h];
	int num = 1;

	for (int i = 0; i<h; i++)
		for (int j = 0; j < w; j++)
			p[i*w + j] = ((uchar*)(CSImage->imageData + CSImage->widthStep*i))[j];

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			if (i<m_nExtSize || i>h - m_nExtSize || j<m_nExtSize || j>w - m_nExtSize)
				continue;
			bool state = true;
			if (p[i*w + j] < m_nThreshold)
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
					if (p[i*w + j] < p[ii*w + jj])
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
				coord[num] = i * w + j;
				num++;
			}
		}
	}
	coord[0] = num;
		
	return coord;
}

float ImagePreprocess::cal_distance(IplImage *CSImage, int p1, int p2)
{
	int w = CSImage->width;
	int p1_y = p1 % w;
	int p1_x = p1 / w;
	int p2_y = p2 % w;
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
					nDistance = cal_distance(CSImage, centerPoint[label].x*w + centerPoint[label].y, coord[j]);//j+1?
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
	m_nClusterCount--;
	return points;
}

vector<vector<Point>> ImagePreprocess::draw_grid(IplImage *CSImage, vector<vector<Point>>point)
{
	Mat srcMat;
	srcMat = cvarrToMat(CSImage);
	vector<vector<Point>>fPoint(m_nClusterCount, vector<Point>(4));				//找出四个角点
	int delta = 2;
	for (int label = 0; label < m_nClusterCount; label++)
	{
		int delta = 0;
		int x_min = point[label][1].x - delta;
		int x_max = point[label][1].x + delta;
		int y_min = point[label][1].y - delta;
		int y_max = point[label][1].y + delta;
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

	return fPoint;
}

IplImage* ImagePreprocess::otsu_fast(IplImage *ScaleImage, vector<vector<Point>>point)
{
	//图像转化成灰度图                                                                            
	IplImage *grayImg = cvCreateImage(cvSize(ScaleImage->width, ScaleImage->height), IPL_DEPTH_8U, 1);
	cvCvtColor(ScaleImage, grayImg, CV_BGR2GRAY);
	Mat grayImgMat = cvarrToMat(grayImg);
	int label, i, j;

	for (label = 0; label < m_nClusterCount; label++)
	{
		bool Ctrl = true;
		if (Ctrl)
		{
			//设置ROI及图片预处理
			Rect rect(point[label][0].x, point[label][0].y, point[label][2].x - point[label][0].x, point[label][2].y - point[label][0].y);
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
			int nThreshold = max_val;

			//腐蚀膨胀
			int kernelSize = 3;
			Mat element = getStructuringElement(MORPH_RECT, Size(kernelSize, kernelSize));
			erode(grayImgROI, grayImgROI, element);
			dilate(grayImgROI, grayImgROI, element);
			threshold(grayImgROI, grayImgROI, nThreshold, 255, 0);

			//提取目标图像轮廓
			vector<vector<Point>> contours;
			vector<Vec4i> hierarchy;
			findContours(grayImgROI, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
			grayImgROI = Mat::zeros(rect.size(), CV_8UC1);
			//drawContours(grayImgROI, contours, -1, Scalar(255, 255, 255));

			//绘制最小外接矩形
			vector<RotatedRect> box(contours.size());
			Point2f rect2[4];
			for (i = 0; i<contours.size(); i++)
			{
				box[i] = minAreaRect(Mat(contours[i]));  //计算每个轮廓最小外接矩形
				box[i].points(rect2);  //把最小外接矩形四个端点复制给rect数组
				for (int j = 0; j<4; j++)
					line(grayImgROI, rect2[j], rect2[(j + 1) % 4], Scalar(255), 1, 4);  //绘制最小外接矩形每条边
			}
		}
	}
	return grayImg;
}

IplImage* ImagePreprocess::image_preprocess(IplImage* src, float fscale, int wnd_size)
{
	IplImage *scale_image = scale_img(src, fscale);
	IplImage *cal_integ_image = cal_integ(scale_image);
	IplImage *cs2_image = cs2(cal_integ_image, wnd_size);

	int *coord;
	coord = max_val_coord(cs2_image);

	vector<vector<Point>>point(100, vector<Point>(100));
	point = cluster(cs2_image, coord);
	vector<vector<Point>>angular_point = draw_grid(cs2_image, point);
	IplImage* preprocess_image = otsu_fast(scale_image, angular_point);

	return preprocess_image;
}



