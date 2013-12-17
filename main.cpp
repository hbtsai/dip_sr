////////////////////////////////////////////////////////////////////////
//
// hello-world.cpp
//
// This is a simple, introductory OpenCV program. The program reads an
// image from a file, inverts it, and displays the result. 
//
////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>


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
	  printf("Usage: main <image-file-name>\n\7");
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

	Mat out_y(img.rows, img.cols, CV_8UC1);
	Mat out_i(img.rows, img.cols, CV_8UC1);
	Mat out_q(img.rows, img.cols, CV_8UC1);
	Mat out(img.rows, img.cols, CV_8UC3);


	/* convert image from RGB to YIQ */

	int m=0, n=0;
	for(m=0; m<img.rows; m++)
	{
		for(n=0; n<img.cols; n++)
		{
			r = img.data[m*img.step + n*3 + 2];
			g = img.data[m*img.step + n*3 + 1];
			b = img.data[m*img.step + n*3 ];
			y = 0.299*r + 0.587*g + 0.114*b;
			i = 0.596*r - 0.275*g - 0.321*b;
			q = 0.212*r - 0.523*g + 0.311*b;

			out_y.data[m*img.rows+n] = y;
			out_i.data[m*img.rows+n] = CV_CAST_8U((int)(i+128));
			out_q.data[m*img.rows+n ] = CV_CAST_8U((int)(q+128));
			out.data[m*img.step+n*3 +2] = y;
			out.data[m*img.step+n*3 +1] = CV_CAST_8U((int)(i+128));
			out.data[m*img.step+n*3 ] = CV_CAST_8U((int)(q+128));

		}
	}
	
	/* upscale Y image */



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

	waitKey(0);
			
	return 0;
}
