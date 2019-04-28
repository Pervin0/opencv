#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
using namespace std;
using namespace cv;
int thresh = 50, N = 11;
const char* wndname = "Detection";

static double angle(Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

static double eqv(double a, double b, double eps = 0.03)
{
	return fabs(a-b)<eps;
}

static bool border(vector<Point>&b, vector<Point>&s)
{
	int k = 0;
	for (int i=0; i<s.size(); i++)
		for (int j=0; j<b.size(); j++)
			if (s[i].x == b[j].x && s[i].y == b[j].y)
				k++;
	return k>=2;
}

static void drawSquares(vector<vector<Point> > &squares, const Mat & image, const Scalar& color, const char * name)
{
	for (size_t i = 0; i < squares.size(); i++)
	{
		const Point* p = &squares[i][0];
		int n = (int)squares[i].size();
		polylines(image, &p, &n, 1, true, color, 3, LINE_AA);
	}

	imshow(name, image);
}




static void findSquares(const Mat& image, vector<vector<Point> >& squares)
{
	squares.clear();

	Mat pyr, timg, gray0(image.size(), CV_8U), gray;

	pyrDown(image, pyr, Size(image.cols / 2, image.rows / 2));
	pyrUp(pyr, timg, image.size());
	vector<vector<Point> > contours;

	vector<Point> img;
	int weight = image.rows;
	int height = image.cols;
	img.push_back(Point(0, 0));
	img.push_back(Point(height, 0));
	img.push_back(Point(height, weight));
	img.push_back(Point(0, weight));
	double img_area = fabs(contourArea(img));

	vector<vector<Point> > sq1, sq2;

	for (int c = 0; c < 3; c++)
	{
		int ch[] = { c, 0 };
		mixChannels(&timg, 1, &gray0, 1, ch, 1);

		for (int l = 0; l < N; l++)
		{
			if (l == 0)
			{
				Canny(gray0, gray, 0, thresh, 5);
				dilate(gray, gray, Mat(), Point(-1, -1));
			}
			else
			{
				gray = gray0 >= (l + 1) * 255 / N;
			}

			findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

			vector<Point> approx;
			for (size_t i = 0; i < contours.size(); i++)
			{
				approxPolyDP(contours[i], approx, arcLength(contours[i], true)*0.02, true);
				double ca = fabs(contourArea(approx));
				if (ca > 1000 && isContourConvex(approx)) sq1.push_back(approx);
				if (approx.size() == 4 &&
					ca > 1000 &&
					isContourConvex(approx) &&
					fabs(ca-img_area) > 10000 &&
					!border(img, approx))
				{
					sq2.push_back(approx);
					double maxCosine = 0;
					double cosine[4];
					cosine[0] = fabs(angle(approx[1], approx[3], approx[0]));

					for (int j = 2; j < 5; j++)
					{
						cosine[j-1] = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
					}

					if ( ( eqv(cosine[0],cosine[1]) && eqv(cosine[2], cosine[3]) ) ||
						 ( eqv(cosine[0],cosine[2]) && eqv(cosine[1], cosine[4]) ) )
						squares.push_back(approx);
				}
			}
		}
	}
	Mat tmp, tmp1;
	image.copyTo(tmp);
	image.copyTo(tmp1);
	drawSquares(sq1,tmp,Scalar(255, 0, 0), "All");
	drawSquares(sq2,tmp1,Scalar(0, 0, 255), "4");
}

int main()
{
	Mat image;
	image = imread("test3.jpg", IMREAD_COLOR);
	if (image.empty())
	{
		cout << "Couldn't load " << endl;
		cin.get();
		return 1;
	}
	namedWindow("Original image", 1);
	imshow("Original image", image);

	int weight = image.rows;
	int height = image.cols;

	vector<vector<Point> > squares;

	findSquares(image, squares);
	drawSquares(squares,image,Scalar(0, 255, 0), "Final");

	waitKey(0);
	return 0;
}
