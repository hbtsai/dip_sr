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
#include "Convolute.h"
#include "Backprojection.h"
#include "BuildINumber.h"

#if defined(_DEBUG)
#define dprintf(M, ...) fprintf(stderr, "DEBUG %s:%d: " M "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define dprintf(M, ...)
#endif

using namespace cv;

void BuildPatchRecordTable(StructSubLayer* HighLayer, StructSubLayer* LowLayer, double ScalePerLayer, int iter, StructPatchRecordTable *retPRTable)
{

	//StructPatchRecordTable ret;

	int HighHeight = HighLayer->ValidHeight;
	int HighWidth = HighLayer->ValidWidth;
	int PatchNum = (HighHeight-4)*(HighWidth-4);
	int INumber = HighLayer->INumber;
    int lowridx = 0, lowcidx=0;
    Mat LowPatch(4, 4, CV_64FC1);
    int r=0, c=0, r1=0, r2=0, r3=0, c1=0, c2=0, c3=0;
    double Top=0.0, Bottom=0.0, Left=0.0, Right=0.0;
    double Sum=0.0, Portion=0.0;
    double PortionTop=0.0, PortionBottom=0.0, PortionLeft=0.0, PortionRight=0.0;
    int idx=0;

        double Scale = pow(ScalePerLayer, (iter-1));      //the Scale mean the res of Con0 to Con-i
        Mat Conv = LowLayer->Conv;
        for(r=1; r<=HighHeight-4; r++)
        {
            for(c=1; c<=HighWidth-4; c++)
            {
                //Compute the grid of low
                LowPatch = Mat::zeros(LowPatch.rows, LowPatch.cols, CV_64F);
                for(lowridx = 1; lowridx<=4; lowridx++)
                {

                    Top = (r-1 + 1.25 * (lowridx-1)) * Scale;         
                    Bottom = Top + 1.25 * Scale;
                    r1 = floor(Top) + 1;
                    r2 = floor(Bottom-ExSmall)+1;
                    for(lowcidx = 1; lowcidx<=4; lowcidx++)
                    {
                        Left = (c-1 + 1.25 * (lowcidx-1)) * Scale;
                        Right = Left + 1.25 * Scale;
                        c1 = floor(Left) + 1;                   //c1 means cstart
                        c2 = floor(Right-ExSmall) + 1;          //c2 means cend

                        Portion = (ceil(Top+ExSmall) - Top) * (ceil(Left+ExSmall)-Left);
                        Sum = Conv.at<double>(r1-1,c1-1) * Portion;     //TopLeft
                        Portion = (ceil(Top+ExSmall) - Top) * (Right-floor(Right-ExSmall));
                        Sum = Sum + Conv.at<double>(r1-1,c2-1) * Portion;   //TopRight
                        Portion = (Bottom - floor(Bottom-ExSmall)) * (ceil(Left+ExSmall)-Left);
                        Sum = Sum + Conv.at<double>(r2-1,c1-1) * Portion;   //BottomLeft
                        Portion = (Bottom - floor(Bottom-ExSmall)) * (Right-floor(Right-ExSmall));
                        Sum = Sum + Conv.at<double>(r2-1,c2-1) * Portion;   //BottomRight

                        //for 4 edge
                        if(c1+1 != c2)
                        {
                            PortionTop = ceil(Top+ExSmall) - Top;
                            PortionBottom = Bottom - floor(Bottom-ExSmall);
                            for(c3 = c1+1; c3<=c2-1; c3++)
                            {
                                Sum = Sum + Conv.at<double>(r1-1,c3-1)*PortionTop;      
                                Sum = Sum + Conv.at<double>(r2-1,c3-1)*PortionBottom;      
                            }
                        }
                        if(r1+1 != r2)
                        {
                            PortionLeft = ceil(Left+ExSmall) - Left;
                            PortionRight = Right - floor(Right-ExSmall);
                            for(r3 = r1+1; r3<=r2-1; r3++)
                            {
                                Sum = Sum + Conv.at<double>(r3-1,c1-1)*PortionLeft;
                                Sum = Sum + Conv.at<double>(r3-1,c2-1)*PortionRight;
                            }
                        }

                        //for interior pixels
                        if(r1+1<r2 && c1+1<c2)
                        {
                            for(r3=r1+1; r3<=r2-1; r3++)
                                for(c3=c1+1; c3<=c2-1; c3++)
                                    Sum = Sum + Conv.at<double>(r3-1,c3-1);
                        }
                        LowPatch.at<double>(lowridx-1,lowcidx-1) = Sum / pow((1.25*Scale), 2);       
                        //becuase we map 1.25*Scale x 1.25*Scale pixels into a pixel
                    }
                }

                //save the vector and r,c,i
                retPRTable[idx].HighPatch5x5_r = r;
                retPRTable[idx].HighPatch5x5_c = c;
                retPRTable[idx].HighPatch5x5_INumber = INumber;
                retPRTable[idx].Vector = LowPatch.clone();
                retPRTable[idx].Vector = LowPatch.reshape(16,1);           
                idx++;
            }
        }

}




Mat GridSubSampling(Mat* Conv, int FormatHeight, int FormatWidth, int ValidHeight, int ValidWidth, 
		int TrueHeight, int TrueWidth, double Ratio, int HighHeight, int HighWidth)
{

        Mat Grid(FormatHeight , FormatWidth, CV_64FC1);

		double Ratio_r=0.0, Ratio_c=0.0, PixelValue=0.0;
		bool bBoundaryCase_r = false, bBoundaryCase_c = false;
		double HighTop=0.0, HighBottom=0.0, HighLeft=0.0, HighRight=0.0;
		int r=0, c=0, r1=0, r2=0, c1=0, c2=0;
		int low_r = 0, low_c=0;
		double Portion=0.0, PortionTop=0.0, PortionBottom=0.0, PortionLeft=0.0, PortionRight=0.0;
	   double Sum=0.0;
        //Do grid subsampling high-res to low-res. Current case: high is integer res, but low is non-integer res.
		//if the TureHeight is not integer, the last low_r will be floor(TrueHeight)=ValidHeight, 
		//but now I need it reaches FormatHeight
        for(low_r=1; low_r<=FormatHeight; low_r++) 
		{
            HighTop = (low_r-1) * Ratio;
            r1 = floor(HighTop) + 1;   
            if(low_r <= ValidHeight)
			{
                HighBottom = HighTop + Ratio;
                r2 = floor(HighBottom-ExSmall) + 1;
                Ratio_r = Ratio;
                bBoundaryCase_r = false;
			}
            else if(ValidHeight != FormatHeight)
			{
                HighBottom = HighHeight;
                r2 = HighHeight;                
                Ratio_r = HighBottom - HighTop;
                bBoundaryCase_r = true;                
			}

            for(low_c=1; low_c<=FormatWidth; low_c++)
			{
                HighLeft = (low_c-1) * Ratio;
                c1 = floor(HighLeft) + 1;
                if( low_c <= ValidWidth)
				{
                    HighRight = HighLeft + Ratio;
                    c2 = floor(HighRight-ExSmall) + 1;
                    Ratio_c = Ratio;
                    bBoundaryCase_c = false;
				}
                else if(ValidWidth != FormatWidth)
				{
                    HighRight = HighWidth;
                    c2 = HighWidth;
                    Ratio_c = HighRight - HighLeft;
                    bBoundaryCase_c = true;
				}

                //for 4 corners
                Portion = (ceil(HighTop+ExSmall) - HighTop) * (ceil(HighLeft+ExSmall)-HighLeft);
                Sum = Conv->at<double>(r1-1,c1-1) * Portion;     //TopLeft
                if(r1<r2)
				{
                    Portion = (HighBottom - floor(HighBottom-ExSmall)) * (ceil(HighLeft+ExSmall)-HighLeft);
                    Sum = Sum + Conv->at<double>(r2-1,c1-1) * Portion;   //BottomLeft
				}
                if(c1<c2)
				{
                    Portion = (ceil(HighTop+ExSmall) - HighTop) * (HighRight-floor(HighRight-ExSmall));
                    Sum = Sum + Conv->at<double>(r1-1,c2-1) * Portion;   //TopRight
				}
                if (r1<r2 && c1<c2)
				{
                    Portion = (HighBottom - floor(HighBottom-ExSmall)) * (HighRight-floor(HighRight-ExSmall));
                    Sum = Sum + Conv->at<double>(r2-1,c2-1) * Portion;   //BottomRight
				}

                //for 4 edge
                if(c1+1 < c2)   //for left edge exclusive top-left and bottom-left
				{
                    PortionTop = ceil(HighTop+ExSmall) - HighTop;

                    for(c = c1+1; c<=c2-1; c++)
                        Sum = Sum + Conv->at<double>(r1-1,c-1)*PortionTop;      

                    if (!bBoundaryCase_r)
					{
                        PortionBottom = HighBottom - floor(HighBottom-ExSmall);
                        for(c = c1+1; c<=c2-1; c++)
                            Sum = Sum + Conv->at<double>(r2-1,c-1)*PortionBottom;      
					}
				}

                if(r1+1 < r2)
				{
                    PortionLeft = ceil(HighLeft+ExSmall) - HighLeft;
                    for(r = r1+1; r<=r2-1; r++)
                        Sum = Sum + Conv->at<double>(r-1,c1-1)*PortionLeft;

                    if(!bBoundaryCase_c)
					{
                        PortionRight = HighRight - floor(HighRight-ExSmall);
                        for(r = r1+1; r<=r2-1; r++)
                            Sum = Sum + Conv->at<double>(r-1,c2-1)*PortionRight;
					}
				}

                //for interior pixels
                if(bBoundaryCase_c)
				{
                    if(r1+1<r2 && c1+1<=c2)
					{
                        for(r=r1+1; r<=r2-1; r++)
                            for(c=c1+1; c<=c2; c++)
                                Sum = Sum + Conv->at<double>(r-1,c-1);
					}
				}
                else if(bBoundaryCase_r)
				{
                    if(r1+1<=r2 && c1+1<c2)
					{
                        for(r=r1+1; r<=r2; r++)
                            for(c=c1+1; c<=c2-1; c++)
                                Sum = Sum + Conv->at<double>(r-1,c-1);
					}
				}
                else if(r1+1<r2 && c1+1<c2)
				{
                    for(r=r1+1; r<=r2-1; r++)
                        for(c=c1+1; c<=c2-1; c++)
                            Sum = Sum + Conv->at<double>(r-1,c-1);
				}

                PixelValue = Sum / (Ratio_r * Ratio_c);
                Grid.at<double>(low_r-1,low_c-1) = PixelValue;                
			}

		}
		return Grid;


}

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


	StructSubLayer L0;
    L0.INumber = 0;
    L0.TrueWidth = img_y.cols;
    L0.TrueHeight = img_y.rows;
    L0.FormatWidth = L0.TrueWidth;
    L0.FormatHeight = L0.TrueHeight;
    L0.ValidWidth = L0.TrueWidth;
    L0.ValidHeight = L0.TrueHeight;
    L0.Conv = img_yf;
    L0.GridAsHighLayer = img_yf;
//    L0.PatchRecordTable;

	const int NUM_SUBLAYERS=7;
	double GauVar_r=1.0, GauVar=0.0;
	double ScalePerLayer=1.25;
    int ReconPixelOverlap = 4;
	StructSubLayer SubLayers[NUM_SUBLAYERS]; // sub_layers[0]=L0
	int iter=1;

	while(iter<NUM_SUBLAYERS)
	{

		GauVar=GauVar_r*((double)iter/(double)NUM_SUBLAYERS);

		SubLayers[iter].INumber = -iter;
        SubLayers[iter].TrueWidth = L0.TrueWidth / pow(ScalePerLayer,iter);
        SubLayers[iter].TrueHeight = L0.TrueHeight / pow(ScalePerLayer, iter);
        SubLayers[iter].FormatWidth = ceil(  SubLayers[iter].TrueWidth );
        SubLayers[iter].FormatHeight = ceil( SubLayers[iter].TrueHeight );
        SubLayers[iter].ValidWidth = floor(  SubLayers[iter].TrueWidth );
        SubLayers[iter].ValidHeight = floor( SubLayers[iter].TrueHeight );
		SubLayers[iter].Conv = img_yf;
        Convolute(&img_yf, &SubLayers[iter].Conv, GauVar );

        Mat Conv = SubLayers[iter].Conv;
        double Ratio = pow(ScalePerLayer, iter);

		Mat Grid = GridSubSampling(&Conv, SubLayers[iter].FormatHeight, SubLayers[iter].FormatWidth, 
				SubLayers[iter].ValidHeight, SubLayers[iter].ValidWidth, 
				SubLayers[iter].TrueHeight, SubLayers[iter].TrueWidth, 
				Ratio, Conv.rows, Conv.cols);

		SubLayers[iter].GridAsHighLayer = Grid;
		iter++;
	}

	
#if 0
	char fname[1024]={};
	Mat testimg, output;
	for(iter=1; iter<NUM_SUBLAYERS; iter++)
	{
		sprintf(fname, "MyTest_%d.png", SubLayers[iter].INumber);
	    //SaveFileName = [Para.TempDataFolder 'Lower_INumber' num2str(INumber) '.png'];
		

		testimg = SubLayers[iter].GridAsHighLayer;
	
		for(m=0; m<testimg.rows; m++)
			for(n=0; n<testimg.cols; n++)
				testimg.at<double>(m, n) *= 255.0;

		testimg.convertTo(output, CV_8U);

	    imwrite( fname, output );
	}
#endif


    int PatchNum;
    for(iter=1; iter<=6; iter++)
    {
        PatchNum = (SubLayers[iter].ValidHeight-4)*(SubLayers[iter].ValidWidth-4);
        StructPatchRecordTable *SubLayersPRTable = new StructPatchRecordTable[PatchNum];
		BuildPatchRecordTable(&SubLayers[iter-1], &SubLayers[iter], ScalePerLayer, iter, SubLayersPRTable);
        SubLayers[iter].PatchRecordTable = SubLayersPRTable;
    }

	for(iter=1; iter<NUM_SUBLAYERS; iter++)
    {

//        BuildINumber_func(iter, SubLayers, ScalePerLayer, ReconPixelOverlap, NUM_SUBLAYERS);


//        BackProjection(GauVar_r, iter);


    }


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
