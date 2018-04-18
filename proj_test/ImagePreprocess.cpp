#include "stdafx.h"
#include "HeadFile.h"

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
	IplImage *integ_image= cvCreateImage(cvGetSize(scale_image), IPL_DEPTH_32F, 1);
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
	((float*)(integ_image->imageData + 0 * integ_image->widthStep))[0] = dst_pixel[0];

	//第一行的像素值积分
	for (int j = 1; j < nw; j++)
	{
		columnSum[j] = src_pixel[j];
		dst_pixel[j] = columnSum[j];
		dst_pixel[j] += dst_pixel[j - 1];
		((float*)(integ_image->imageData + 0 * integ_image->widthStep))[j] = dst_pixel[0 * nw + j];
	}

	for (int i = 1; i < nh; i++)
	{
		//第一列像素积分
		columnSum[0] += src_pixel[i*nw];
		dst_pixel[i*nw] = columnSum[0];
		((float*)(integ_image->imageData + i * integ_image->widthStep))[0] = dst_pixel[i * nw + 0];
		//其他像素值积分
		for (int j = 1; j < nw; j++)
		{
			columnSum[j] += src_pixel[i*nw + j];
			dst_pixel[i*nw + j] = dst_pixel[i*nw + j - 1] + columnSum[j];
			((float*)(integ_image->imageData + integ_image->widthStep * i))[j] = dst_pixel[i * nw + j];
		}
	}

	// 03: 显著性图
	wnd_size = 1;
	IplImage *cs_image = cvCreateImage(cvGetSize(integ_image), 8, 1);
	int num = wnd_size * wnd_size * 2;

	for (int i = 0; i < nh; i++)
		for (int j = 0; j < nw; j++)
			src_pixel[i*nw + j] = ((float *)(integ_image->imageData + i * integ_image->widthStep))[j];

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

			dst_pixel[i*nw + j] = 3 * (((pixel_LR > pixel_UP) ? pixel_LR : pixel_UP));
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
					nDistance = cal_distance(cs_image, centerPoint[label].x*nw + centerPoint[label].y, coord[j]);//j+1?
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
		bool ctrl = true;
		if (ctrl)
		{
			//设置ROI及图片预处理
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


