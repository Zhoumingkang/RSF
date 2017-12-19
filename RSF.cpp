#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <fstream>
#include <ctime>
using namespace std;
using namespace cv;

//function gaussian_kernal 生成高斯核
Mat gaussian_kernal(int kernelSize, double sigma0)
{  
	int halfSize = (kernelSize-1)/ 2;
	Mat K(kernelSize, kernelSize, CV_64FC1);  

	//生成二维高斯核  
	double s2 = 2.0 * sigma0 * sigma0;  
	for(int i = (-halfSize); i <= halfSize; i++)  
	{  
		int m = i + halfSize;  
		for (int j = (-halfSize); j <= halfSize; j++)  
		{  
			int n = j + halfSize;  
			double v = exp(-(1.0*i*i + 1.0*j*j) / s2);  
			K.ptr<double>(m)[n] = v;  
		}  
	}  
	Scalar all = sum(K);  
	Mat gaussK;  
	K.convertTo(gaussK, CV_64FC1, (1/all[0]));  

	return gaussK;  
}



enum ConvolutionType
{   
	/* Return the full convolution, including border */  
	CONVOLUTION_FULL,   
   
	/* Return only the part that corresponds to the original image */  
	CONVOLUTION_SAME,  
	/* Return only the submatrix containing elements that were not influenced by the border */  
	 CONVOLUTION_VALID  
}; 
//function conv2 对应Matlab中conv2函数，计算二维卷积
Mat conv2(const Mat &img, const Mat& ikernel, ConvolutionType type)   
{  
	Mat dest(img.size(), CV_64FC1);  
	Mat kernel(img.size(), CV_64FC1); 
	flip(ikernel,kernel,-1);  
	Mat source(img.size(), CV_64FC1, img.data);

	if(CONVOLUTION_FULL == type)   
	{  
		 source = Mat();  
		 const int additionalRows = kernel.rows-1, additionalCols = kernel.cols-1;  
		 copyMakeBorder(img, source, (additionalRows+1)/2, additionalRows/2, (additionalCols+1)/2, additionalCols/2, BORDER_CONSTANT, Scalar(0));  
	}  
	Point anchor(kernel.cols - kernel.cols/2 - 1, kernel.rows - kernel.rows/2 - 1);  
	int borderMode = BORDER_CONSTANT;  
	filter2D(source, dest, CV_64FC1, kernel, anchor, 0, borderMode);  
   
	if(CONVOLUTION_VALID == type)   
	{  
		 dest = dest.colRange((kernel.cols-1)/2, dest.cols - kernel.cols/2).rowRange((kernel.rows-1)/2, dest.rows - kernel.rows/2);  
	}  
	 return dest;  
}  

//function del2 in Matlab
//l(i,j)=[u(i+1,j)+u(i-1,j)+u(i,j+1)+u(i,j-1)]/4 - u(i,j)
//数组左上角角点的公式应为：l(i,j)=[u(i+2,j)+u(i+2,j)+2u(i,j)]/4 - [u(i+1,j)+u(i,j+1)]/2
//数组左边边点公式应为：l(i,j)=[u(i+1,j)+u(i-1,j)+u(i,j+2)+u(i,j)]/4 - [u(i,j)+u(i,j+1)]/2
//依这两个公式类推，可以得到其他角点和其他边点的计算公式，从而得到计算结果。
void del2(Mat &u, Mat &L)
{
	for (int i = 1; i < u.rows - 1; i++)
		for (int j = 1; j < u.cols - 1; j++)
			L.at<double>(i,j) = (u.at<double>(i+1, j) + u.at<double>(i-1,j) + u.at<double>(i,j+1) + u.at<double>(i,j-1))/4  - u.at<double>(i,j);
	for (int row = 1; row < u.rows - 1; row++)
	{
		L.at<double>(row,0) = (u.at<double>(row+1,0) + u.at<double>(row-1,0) + u.at<double>(row,2) + u.at<double>(row,0))/4 - (u.at<double>(row,0) + u.at<double>(row,1))/2;
		L.at<double>(row,u.cols-1) = (u.at<double>(row+1,u.cols-1) + u.at<double>(row-1,u.cols-1) + u.at<double>(row,u.cols-3) + u.at<double>(row,u.cols-1))/4 - (u.at<double>(row,u.cols-1) + u.at<double>(row,u.cols-2))/2;
	}
	for (int col = 1; col < u.cols-1; col++)
	{
		L.at<double>(0,col) = (u.at<double>(0,col+1) + u.at<double>(0,col-1) + u.at<double>(2,col) + u.at<double>(0,col))/4 - (u.at<double>(0,col) + u.at<double>(1, col))/2;
		L.at<double>(u.rows-1,col) = (u.at<double>(u.rows-1, col+1) + u.at<double>(u.rows-1,col-1) + u.at<double>(u.rows-3,col) + u.at<double>(u.rows-1,col))/4 - (u.at<double>(u.rows-1,col) + u.at<double>(u.rows-2,col))/2;
	}
	L.at<double>(0,0) = (u.at<double>(2,0) + u.at<double>(0,2) + 2*u.at<double>(0,0))/4 - (u.at<double>(1,0) + u.at<double>(0,1))/2;
	L.at<double>(0,u.cols-1) = (u.at<double>(2,u.cols-1) + u.at<double>(0,u.cols-3) + 2*u.at<double>(0,u.cols-1))/4 - (u.at<double>(1,u.cols-1) + u.at<double>(0,u.cols-2))/2;
	L.at<double>(u.rows-1,0) = (u.at<double>(u.rows-1,2) + u.at<double>(u.rows-3,0) + 2*u.at<double>(u.rows-1,0))/4 - (u.at<double>(u.rows-1,1) + u.at<double>(u.rows-2,0))/2;
	L.at<double>(u.rows-1,u.cols-1) = (u.at<double>(u.rows-3,u.cols-1) + u.at<double>(u.rows-1,u.cols-3) + 2*u.at<double>(u.rows-1,u.cols-1))/4 - (u.at<double>(u.rows-2,u.cols-1) + u.at<double>(u.rows-1, u.cols-2))/2;
}

//function gradient in Matlab
//[Fx,Fy]=gradient(F)，其中Fx为其水平方向上的梯度，
//Fy为其垂直方向上的梯度，Fx的第一列元素为原矩阵第二列与第一列元素之差，
//Fx的第二列元素为原矩阵第三列与第一列元素之差除以2，
//以此类推：Fx(i,j)=(F(i,j+1)-F(i,j-1))/2。最后一列则为最后两列之差。同理，可以得到Fy
void gradient(Mat &Ix, Mat &Iy, Mat input){  
    //typedef BOOST_TYPEOF(input.data) ElementType;  
    for (int nrow = 0; nrow < input.rows; nrow++)  
    {  
        for (int ncol = 0; ncol < input.cols; ncol++)  
        {  
            if (ncol == 0){  
                Ix.at<double>(nrow, ncol) = input.at<double>(nrow, 1) - input.at<double>(nrow, 0);  
            }  
            else if (ncol == input.cols - 1){  
                Ix.at<double>(nrow, ncol) = input.at<double>(nrow, ncol) - input.at<double>(nrow, ncol - 1);  
  
            }  
            else  
                Ix.at<double>(nrow, ncol) = (input.at<double>(nrow, ncol + 1) - input.at<double>(nrow, ncol - 1)) / 2;  
        }  
    }   
    for (int nrow = 0; nrow < input.rows; nrow++)  
    {  
        for (int ncol = 0; ncol < input.cols; ncol++)  
        {  
            if (nrow == 0){  
                Iy.at<double>(nrow, ncol) = input.at<double>(1, ncol) - input.at<double>(0, ncol);  
            }  
            else if (nrow == input.rows - 1){  
                Iy.at<double>(nrow, ncol) = input.at<double>(nrow, ncol) - input.at<double>(nrow - 1, ncol);  
            }  
            else  
                Iy.at<double>(nrow, ncol) = (input.at<double>(nrow + 1, ncol) - input.at<double>(nrow - 1, ncol)) / 2;  
        }  
  
    }  
  
}  

//function NeumannBoundCond 纽曼边界条件
//第三行赋给第一行，倒数第三行赋给最后一行
//左右两边同样操作，四个角点用内角点赋值
void NeumannBoundCond(Mat &g)
{
	g.at<double>(0,0) = g.at<double>(2,2);
	//g.at<double>(74,79) = g.at<double>(74,79);
	g.at<double>(0,g.cols-1) = g.at<double>(2,g.cols-3);
	g.at<double>(g.rows-1,0) = g.at<double>((g.rows-3),2);
	g.at<double>(g.rows-1,g.cols-1) = g.at<double>(g.rows-3,g.cols-3);
	for (int row = 1; row < g.rows-1; row++)
	{
		g.at<double>(row,0) = g.at<double>(row,2);
		g.at<double>(row,g.cols-1) = g.at<double>(row,g.cols-3);
	}
}

//function curvature_central 计算曲率
void curvature_central(Mat &u, Mat &K)
{
	Mat ux(u.size(), CV_64FC1);
	Mat uy(u.size(), CV_64FC1);
	Mat normDu(u.size(), CV_64FC1);
	Mat Nx(u.size(), CV_64FC1);
	Mat Ny(u.size(), CV_64FC1);
	Mat junk(u.size(), CV_64FC1);
	Mat nxx(u.size(), CV_64FC1);
	Mat nyy(u.size(), CV_64FC1);
	gradient(ux, uy, u);
	for (int i = 0; i < ux.rows; i++)
		for (int j = 0; j < uy.cols; j++)
			normDu.at<double>(i,j) = sqrt(pow(ux.at<double>(i,j), 2) + pow(uy.at<double>(i,j), 2) + 1e-10);
	divide(ux, normDu, Nx);
	divide(uy, normDu, Ny);
	gradient(nxx, junk, Nx);
	gradient(junk, nyy, Ny);
	K = nxx + nyy;
}

//compute function Dric
void Dric(double epsilon, Mat &u, Mat &DrcU)
{
	for (int i = 0; i < u.rows; i++)
		for (int j = 0; j < u.cols; j++)
			DrcU.at<double>(i,j) = (epsilon/CV_PI) / (pow(epsilon,2) + pow((u.at<double>(i,j)),2));
}

//function localBinaryFit  局部能量拟合函数
void localBinaryFit(Mat &Img, Mat &u, Mat KI, Mat &KONE,
					Mat &Ksigma, double epsilon, Mat &f1, Mat &f2)
{
	Mat Hu(u.size(), CV_64FC1);
	Mat I(u.size(), CV_64FC1);
	Mat c1(u.size(), CV_64FC1);
	Mat c2(u.size(), CV_64FC1);
	for (int i = 0; i < u.rows; i++)
		for (int j = 0; j < u.cols; j++)
			Hu.at<double>(i,j) = 0.5*(1+(2/CV_PI)*atan((u.at<double>(i,j))/epsilon));
	I = Img.mul(Hu);
	c1 = conv2(Hu, Ksigma, CONVOLUTION_SAME);
	c2 = conv2(I, Ksigma, CONVOLUTION_SAME);
	divide(c2, c1, f1);
	c2 = KI-c2;
	c1 = KONE-c1;
	divide(c2, c1, f2);
}

//function s to compute s1 and s2
void s(double lambda1, double lambda2, Mat &f1, Mat &f2, Mat &s1, Mat &s2)
{
	for(int i = 0; i < f1.rows; i++)
		for (int j = 0; j < f1.cols; j++)
		{
			s1.at<double>(i,j) = lambda1*pow((f1.at<double>(i,j)), 2) - lambda2*pow((f2.at<double>(i,j)), 2);
			s2.at<double>(i,j) = lambda1*(f1.at<double>(i,j)) - lambda2*(f2.at<double>(i,j));
		}
}

//function DataForce to compute dataForce
void DataForce(double lambda1, double lambda2, Mat &KONE, Mat &Img,
			   Mat &Ksigma, Mat &s1, Mat &s2, Mat &dataForce)
{
	Mat D1, D2, D3, D4;
	D1 = (lambda1 - lambda2)*KONE;
	D1 = D1.mul(Img);
	D1 = D1.mul(Img);
	D2 = conv2(s1, Ksigma, CONVOLUTION_SAME);
	D3 = conv2(s2, Ksigma, CONVOLUTION_SAME);
	D4 = 2*Img;
	D3 = D4.mul(D3);

	dataForce = D1 + D2 - D3;
}

//function RSF
void RSF(Mat &u, Mat &Img, Mat &Ksigma, Mat &KI,Mat &KONE, double nu, double timestep, int mu, double lambda1,
				 double lambda2, double epsilon, int numIter=1)
{
	Mat s1(u.size(), CV_64FC1);
	Mat s2(u.size(), CV_64FC1);
	Mat dataForce(u.size(), CV_64FC1);
	Mat A(u.size(), CV_64FC1);
	Mat P(u.size(), CV_64FC1);
	Mat L(u.size(), CV_64FC1);
	Mat f1(u.size(), CV_64FC1);
	Mat f2(u.size(), CV_64FC1);
	Mat K(u.size(), CV_64FC1);
	Mat L1(u.size(), CV_64FC1);
	Mat DrcU(u.size(), CV_64FC1);
	//for (int k1 = 1; k1 < numIter; k1++)
	//{
		NeumannBoundCond(u);
		curvature_central(u, K);
		Dric(epsilon, u, DrcU);
		localBinaryFit(Img, u, KI, KONE, Ksigma,epsilon, f1, f2);
		s(lambda1, lambda2, f1, f2, s1, s2);
		DataForce(lambda1, lambda2, KONE, Img, Ksigma, s1, s2,dataForce);
		A = -DrcU.mul(dataForce);
		del2(u,L1);
		P = mu*(4*L1 - K);
		L = nu*(DrcU.mul(K));
		u = u + timestep*(L+P+A);
	//}
}


//function contour
void contour(Mat &u, Mat &img)
{
	double left, right, up, down, x;
	for (int i = 1; i < u.rows-1; i++)
		for (int j = 1; j < u.cols-1; j++)
		{
			left = u.at<double>(i,j-1);
			right = u.at<double>(i,j+1);
			up = u.at<double>(i-1,j);
			down = u.at<double>(i+1,j);
			x = u.at<double>(i,j);
			if (x*left<0 || x*right<0 || x*up<0 || x*down<0)
			{
				img.at<cv::Vec3b>(i,j)[0] = 0;
				img.at<cv::Vec3b>(i,j)[1] = 0;
				img.at<cv::Vec3b>(i,j)[2] = 255;
			}
		}
}

int main()
{
	//计算运行时间
	clock_t start, end;
	double totaltime;
	start = clock();

	int c0 = 2;
	Mat Img1 = imread("1.jpg");
	Mat Img;
	cvtColor(Img1, Img, CV_RGB2GRAY);
	int iterNum = 100;
	double lambda1 = 1.0;
	double lambda2 = 1.0;
	double nu = 0.002 * 255 * 255; //coefficient of the length term
	Mat initialLSF(Img.size(), CV_64FC1,c0);//初始曲线
	Mat u(initialLSF);
	for (int i = 100; i < 107; i++)
		for (int j = 30; j < 40; j++)
			u.at<double>(i,j) = -c0;

	double timestep = 0.1;
	int mu = 1;
	double epsilon = 1.0;
	double sigma = 10.0;
	Mat K(Img.size(), CV_64FC1);
	Mat KI(Img.size(), CV_64FC1);
	Mat KONE(Img.size(), CV_64FC1);
	K = gaussian_kernal(((2.0*sigma)*2.0+1.0), sigma);
	Mat I(Img.size(), CV_64FC1);
	for (int i = 0; i < Img.rows; i++)
		for (int j = 0; j < Img.cols; j++)
			I.at<double>(i,j) = (double)Img.at<uchar>(i,j);
	KI = conv2(I, K, CONVOLUTION_SAME);
	Mat Ones(Img.size(), CV_64FC1, 1);
	KONE = conv2(Ones, K, CONVOLUTION_SAME);
	for (int n = 1; n < iterNum+1; n++)
	{
		cout << "第 " << n << "迭代" << endl;
		RSF(u, I, K, KI, KONE, nu, timestep, mu, lambda1, lambda2, epsilon, 1);
	}

	contour(u, Img1);
	namedWindow("contour", CV_WINDOW_NORMAL);
	imshow("contour", Img1);

	end = clock();

	totaltime = (double)(end - start) / CLOCKS_PER_SEC;
	cout << "迭代了" << iterNum << "次，需要" << totaltime << " 秒!" << endl;
	waitKey();
	return 0;
}