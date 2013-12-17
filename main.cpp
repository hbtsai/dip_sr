////////////////////////////////////////////////////////////////////////
//
// hello-world.cpp
//
// This is a simple, introductory OpenCV program. The program reads an
// image from a file, creates RGB channel, converts to YIQ and display
//
////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <math.h>
#include <opencv/highgui.h>
#include "SubLayer.h"


#if defined(_DEBUG)
#define dprintf(M, ...) fprintf(stderr, "DEBUG %s:%d: " M "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define dprintf(M, ...)
#endif

using namespace cv;

int main(int argc, char *argv[])
{
	Mat img;
	
	if(argc<2){
	  printf("Usage: %s <image-file-name>\n\7", argv[0]);
	  exit(0);
	}
	
	// load an image  
	img=imread(argv[1]);
	if(!img.data)
	{
	  printf("Could not load image file: %s\n",argv[1]);
	  exit(0);
	}

	uchar r, g, b;
	double y, i, q;

	Mat img_y(img.rows, img.cols, CV_8UC1);
	Mat img_i(img.rows, img.cols, CV_8UC1);
	Mat img_q(img.rows, img.cols, CV_8UC1);
	Mat out(img.rows, img.cols, CV_8UC3);

	/* convert image from RGB to YIQ */

	int nc = img.channels();
	int m=0, n=0;
	for(m=0; m<img.rows; m++)
	{
		for(n=0; n<img.cols; n++)
		{
			r = img.data[m*img.step + n*nc + 2];
			g = img.data[m*img.step + n*nc + 1];
			b = img.data[m*img.step + n*nc ];
			y = 0.299*r + 0.587*g + 0.114*b;
			i = 0.595716*r - 0.274453*g - 0.321263*b;
			q = 0.211456*r - 0.522591*g + 0.311135*b;

			img_y.data[m*img_y.step+n] = y;
			img_i.data[m*img_i.step+n] = CV_CAST_8U((int)(i));
			img_q.data[m*img_q.step+n ] = CV_CAST_8U((int)(q));
			/*
			out.data[m*img.step+n*nc +2] = y;
			out.data[m*img.step+n*nc +1] = CV_CAST_8U((int)(i));
			out.data[m*img.step+n*nc ] = CV_CAST_8U((int)(q));
			*/

		}
	}

	Mat img_yf(img_y.rows, img_y.cols, CV_64FC1);
	img_y.convertTo(img_yf, CV_64F);

	for(m=0; m<img_y.rows; m++)
		for(n=0; n<img_y.cols; n++)
			img_yf.at<double>(m, n) /= 255.0;


	// img_yf is the y channel of original image, dynamic range [0...1]


	SubLayer sub_layers[6];



#if 0
	namedWindow("img_hr", CV_WINDOW_AUTOSIZE);
	imshow("img_hr", img);
	namedWindow("img_y", CV_WINDOW_AUTOSIZE);
	imshow("img_y", out_y);
	namedWindow("img_i", CV_WINDOW_AUTOSIZE);
	imshow("img_i", out_i);
	namedWindow("img_q", CV_WINDOW_AUTOSIZE);
	imshow("img_q", out_q);
	namedWindow("img_yiq", CV_WINDOW_AUTOSIZE);
	imshow("img_yiq", out);
#endif

	waitKey(0);
			
	return 0;
}
