
// RobotPosDlg.cpp : 实现文件
//

#include "stdafx.h"
#include "RobotPos.h"
#include "RobotPosDlg.h"
#include "afxdialogex.h"
#include "MyHeaderFile.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CRobotPosDlg 对话框



CRobotPosDlg::CRobotPosDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_ROBOTPOS_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CRobotPosDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_SLIDER1, ThresholdSlider);
}

BEGIN_MESSAGE_MAP(CRobotPosDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &CRobotPosDlg::OpenCamera)
	ON_WM_TIMER()
	ON_NOTIFY(NM_CUSTOMDRAW, IDC_SLIDER1, &CRobotPosDlg::OnNMCustomdrawSlider1)
	ON_NOTIFY(NM_CUSTOMDRAW, IDC_SLIDER1, &CRobotPosDlg::OnNMCustomdrawSlider1)
END_MESSAGE_MAP()


// CRobotPosDlg 消息处理程序

BOOL CRobotPosDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CRobotPosDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CRobotPosDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CRobotPosDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CRobotPosDlg::OpenCamera()
{
	int exposure = -5;
	// TODO: 在此添加控件通知处理程序代码
	//capture = cvCreateCameraCapture(cameraID);
	//exposure = capture.get(CV_CAP_PROP_EXPOSURE); 
	capture.set(CV_CAP_PROP_EXPOSURE, exposure);
	if (false == capture.isOpened())
	{
		MessageBox(_T("无法连接摄像头！！！"));
		return;
	}
	SetTimer(1, 100, NULL); //定时器，定时时间和帧率一致 
}


void CRobotPosDlg::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: 在此添加消息处理程序代码和/或调用默认值
	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
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
	distCoeffs.at<double>(4, 0) = 0;
	Mat view, rview, map1, map2;
	Size imageSize;
	capture >> frame;
	imageSize = frame.size();
	initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
		getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
		imageSize, CV_16SC2, map1, map2);
	remap(frame, frameCalibrationMat, map1, map2, INTER_LINEAR);
	
	cvtColor(frameCalibrationMat, grayframeMat, CV_BGR2GRAY);
	threshold(grayframeMat, binaryframeMat, ThresholdPosition, 255, CV_THRESH_BINARY);
	frameContourMat = Mat::zeros(frameCalibrationMat.rows, frameCalibrationMat.cols, CV_8UC3);
	findContours(binaryframeMat, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

	/**********显示图像**********/
	rawFrame=&IplImage(frame);
	RawVideoPDC = GetDlgItem(RawVideo)->GetDC();//GetDlgItem(IDC_PIC_STATIC)意思为获取显示控件的句柄（句柄就是指针），获取显示控件的DC  
	GetDlgItem(RawVideo)->GetClientRect(&RawVideoRect);
	RawVideoHDC = RawVideoPDC->GetSafeHdc();//获取显示控件的句柄 
	RawVideoCvvImage.CopyOf(rawFrame, 1); //复制该帧图像  
	RawVideoCvvImage.DrawToHDC(RawVideoHDC, &RawVideoRect);

	binaryFrame = &IplImage(binaryframeMat);
	BinaryFramePDC = GetDlgItem(BinaryFrame)->GetDC();//GetDlgItem(IDC_PIC_STATIC)意思为获取显示控件的句柄（句柄就是指针），获取显示控件的DC  
	GetDlgItem(BinaryFrame)->GetClientRect(&BinaryFrameRect);
	BinaryFrameHDC = BinaryFramePDC->GetSafeHdc();//获取显示控件的句柄 
	BinaryFrameCvvImage.CopyOf(binaryFrame, 1); //复制该帧图像  
	BinaryFrameCvvImage.DrawToHDC(BinaryFrameHDC, &BinaryFrameRect);

	CalibrationVideoPDC = GetDlgItem(CalibrationVideo)->GetDC();//GetDlgItem(IDC_PIC_STATIC)意思为获取显示控件的句柄（句柄就是指针），获取显示控件的DC  
	GetDlgItem(CalibrationVideo)->GetClientRect(&CalibrationVideoRect);
	CalibrationVideoHDC = CalibrationVideoPDC->GetSafeHdc();//获取显示控件的句柄 
	frameCalibration = &IplImage(frameCalibrationMat);
	CalibrationVideoCvvImage.CopyOf(frameCalibration, 1); //复制该帧图像  
	CalibrationVideoCvvImage.DrawToHDC(CalibrationVideoHDC, &CalibrationVideoRect);
	/*********************************/

	cPCount = 0;
	cPCountSquare = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		vector<Point> points = contours[i];
		RotatedRect rect = minAreaRect(Mat(points));//返回最小外接矩形的四个顶点
		rect.points(vertex);//计算矩形的四个顶点
		float ratio = calculation(vertex[0], vertex[1]) / calculation(vertex[1], vertex[2]);
		if (((ratio >= 0.18&&ratio <= 0.22) || (ratio >= 4.5&&ratio <= 5.5)) && contourArea(contours[i]) >= 50)
		{
			centerPoint[cPCount].x = (vertex[0].x + vertex[2].x) / 2.0;
			centerPoint[cPCount].y = (vertex[0].y + vertex[2].y) / 2.0;
			drawContoursFunction();
			drawContours(frameContourMat, contours, i, color, NULL, 8, hierarchy);
			sideLength[cPCount] = (ratio >= 0.19&&ratio <= 0.22) ? calculation(vertex[0], vertex[1]) : calculation(vertex[1], vertex[2]);//编码块边长
			Alpha[cPCount] = (ratio >= 0.19&&ratio <= 0.22) ? -atan(slopeCalculation(vertex[0], vertex[1])) : -atan(slopeCalculation(vertex[1], vertex[2]));
			if (ratio >= 0.18&&ratio <= 0.22)
				Alpha[cPCount] = -atan(slopeCalculation(vertex[0], vertex[1]));
			else if ((ratio >= 4.5&&ratio <= 5.5))
				Alpha[cPCount] = -atan(slopeCalculation(vertex[1], vertex[2]));
			cPCount++;
		}
		else if (ratio <= 1.1&&ratio >= 0.9&&contourArea(contours[i]) >= 50)
		{
			centerPointSquare[cPCountSquare].x = (vertex[0].x + vertex[2].x) / 2.0;
			centerPointSquare[cPCountSquare].y = (vertex[0].y + vertex[2].y) / 2.0;
			drawContoursFunctionSquare();
			drawContours(frameContourMat, contours, i, color, NULL, 8, hierarchy);
			cPCountSquare++;
		}
	}

	Point frameCenter;
	frameCenter.y = frameContourMat.rows / 2;
	frameCenter.x = frameContourMat.cols / 2;
	int BestCPID = 0;//Best CenterPoint ID
	float minDistance = calculation(centerPoint[0], frameCenter);//计算矩形路标定位模块与图像中心的距离
	for (int i = 1;i < cPCount;i++)//选择与图像中心距离最小的矩形路标定位模块
	{
		if (calculation(centerPoint[i], frameCenter) <= minDistance)
		{
			minDistance = calculation(centerPoint[i], frameCenter);
			BestCPID = i;
		}
	}
	bestAlpha = Alpha[BestCPID];//最佳矩形模块倾斜角度
	bestCenterPoint = centerPoint[BestCPID];//最佳矩形模块中心点
	bestSideLength = sideLength[BestCPID];//最佳路标编码块长度  
	
	//画网格
	drawGrid();

	//筛选出最佳矩形模块倾斜角度
	int count = 0;
	for (;count<cPCountSquare;count++)
	{
		if (coordCalculation(Convertion(centerPointSquare[count])) == -1)
			break;
	}
	cout << count << "," << cPCountSquare << "\n";
	if (count >= cPCountSquare)
	{
		if (bestAlpha >= 0)
			bestAlpha = bestAlpha - CV_PI;
		else
			bestAlpha = bestAlpha + CV_PI;
	}

	/**********显示图像**********/
	frameContour = &IplImage(frameContourMat);
	FrameContourPDC = GetDlgItem(FrameContour)->GetDC();//GetDlgItem(IDC_PIC_STATIC)意思为获取显示控件的句柄（句柄就是指针），获取显示控件的DC  
	GetDlgItem(FrameContour)->GetClientRect(&FrameContourRect);
	FrameContourHDC = FrameContourPDC->GetSafeHdc();//获取显示控件的句柄 
	FrameContourCvvImage.CopyOf(frameContour, 1); //复制该帧图像  
	FrameContourCvvImage.DrawToHDC(FrameContourHDC, &FrameContourRect);

	CDialogEx::OnTimer(nIDEvent);
}





void CRobotPosDlg::OnNMCustomdrawSlider1(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMCUSTOMDRAW pNMCD = reinterpret_cast<LPNMCUSTOMDRAW>(pNMHDR);
	// TODO: 在此添加控件通知处理程序代码
	CString strText;
	ThresholdPosition = ThresholdSlider.GetPos();
	ThresholdSlider.SetRange(0,255);
	strText.Format(_T("Current Value :%d"), ThresholdPosition);
	GetDlgItem(Threshold)->SetWindowText(strText);
	*pResult = 0;
}
