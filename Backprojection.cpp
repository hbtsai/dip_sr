#include <math.h>
#include "SubLayer.h"
#include "Convolute.h"
#include "Backprojection.h"

const double ExSmall=1e-12;

void BackProjection(double B_GauVar, int BuildINumber, Mat *AverageImage, int BackProjectionLoopNum, StructSubLayer* L0, StructSubLayer* BuildLayer)
{

		//----------------------------------------//
    /*
		Mat AverageImage;
		Mat ConvResult;
		Mat HighImage;
		AverageImage= ImageSum /ImageWeight;
		double Bk_GauVar = 0.0;
		double B_GauVar =1.0;
		int BackProjectionLoopNum = 3;
		int Top =0, Bottom =0;
		int r1 = 0, r2 = 0, c1 = 0, c2 = 0;
		int Left =0, Right =0;
		StructSubLayer BuildLayer;
        */

        double v, v11, v12, v21, v22, c1, c2, c3, x1, x2, x, r1, r2, r3, y1, y2, y;
        int r, c;
        double Portion=0.0, PortionLeft, PortionRight, PortionTop, PortionBottom;
		int LowFormatHeight=0,LowFormatWidth =0;
		int HighHeight =0,HighWidth =0; 
		int LowHeight =0, LowWidth=0;
        double Sum, Left, Right, Top, Bottom;
		double ScaleH =0.0,ScaleW =0.0;
		double Bk_GauVar = (B_GauVar * BuildINumber) / 6;
		Mat Img = *AverageImage;
        Mat ConvResult = Img;
		int bk1=0;
		for(bk1=1; bk1 <=BackProjectionLoopNum; bk1++)
        {
			Convolute(&ConvResult, &Img, Bk_GauVar );
			Mat HighImage = ConvResult;
			HighHeight = BuildLayer->TrueHeight;
		    HighWidth = BuildLayer->TrueWidth;
			LowHeight =L0->TrueHeight;
			LowWidth =L0->TrueWidth;
			LowFormatHeight = L0->FormatHeight;
            LowFormatWidth = L0->FormatWidth;
			Mat LowImage = Mat::zeros(LowFormatHeight, LowFormatWidth, CV_64F);
			ScaleH = HighHeight / LowHeight;
            ScaleW = HighWidth / LowWidth;
			for (r =1; r<=floor(LowHeight); r++)
            {
				Top =(r-1) * ScaleH;
				Bottom = Top + ScaleH;
				r1 = floor(Top) + 1;
                r2 = floor(Bottom-ExSmall) + 1;
				for (c=1; c<=floor(LowWidth); c++)
                {
					Left = (c-1) * ScaleW;
                    Right = Left + ScaleW;
                    c1 = floor(Left) + 1;
                    c2 = floor(Right-ExSmall) + 1;
					Portion = (ceil(Top+ExSmall) - Top) * (ceil(Left+ExSmall)-Left);
                    Sum = HighImage.at<double>(r1-1, c1-1) * Portion;    
                    Portion = (ceil(Top+ExSmall) - Top) * (Right-floor(Right-ExSmall));
                    Sum = Sum + HighImage.at<double>(r1-1,c2-1) * Portion;   
                    Portion = (Bottom - floor(Bottom-ExSmall)) * (ceil(Left+ExSmall)-Left);
                    Sum = Sum + HighImage.at<double>(r2-1, c1-1) * Portion;  
                    Portion = (Bottom - floor(Bottom-ExSmall)) * (Right-floor(Right-ExSmall));
                    Sum = Sum + HighImage.at<double>(r2-1, c2-1) * Portion;   
						if (c1+1 != c2)
                        {
							PortionTop = ceil(Top+ExSmall) - Top;
                            PortionBottom = Bottom - floor(Bottom-ExSmall);
							for (c3 = c1+1; c3<=c2-1; c3++)
                            {
								Sum = Sum + HighImage.at<double>(r1-1, c3-1)*PortionTop;      
								Sum = Sum + HighImage.at<double>(r2-1, c3-1)*PortionBottom; 
							}
						}
						else if (r1+1 != r2){
		                    PortionLeft = ceil(Left+ExSmall) - Left;
							PortionRight = Right - floor(Right-ExSmall);
							for (r3 = r1+1; c3<=r2-1; r3++){
								Sum = Sum + HighImage.at<double>(r3-1, c1-1)*PortionLeft;
								Sum = Sum + HighImage.at<double>(r3-1, c2-1)*PortionRight;
							}
						}
						else if ( r1 +1 < r2 && c1 +1 < c2){
							for (r3 = r1+1; r3 <=r2-1; r3++){
								for (c3 = c1+1; r3 <=c2-1; c3++){
									Sum = Sum + HighImage.at<double>(r3-1, c3-1);
								}
							}
						}
						int PixelValue = Sum / (ScaleH*ScaleW);
						LowImage.at<double>(r-1,c-1) = PixelValue;
				}
			}

            Mat DownSampling = LowImage;
            /* fixme: Mat cannot use arthmetic */
            Mat Diff = L0->GridAsHighLayer;// - DownSampling;

            //I need a hand-made upward imresize
            //[low_h, low_w] = size(Diff);
            int low_h = Diff.rows;
            int low_w = Diff.cols;
            int high_h = BuildLayer->TrueHeight;
            int high_w = BuildLayer->TrueWidth;
            Mat Up = Mat::zeros(BuildLayer->FormatHeight, BuildLayer->FormatWidth, CV_64F);
            double ScaleH = high_h / low_h;
            double ScaleW = high_w / low_w;
            //extend Diff 1 pixel for boundary case.
            Mat Diff_Ext = Mat::zeros(low_h+1 , low_w+1, CV_64F);


            int m, n;
//          Diff_Ext(1:low_h,1:low_w) = Diff;
            for(m=1; m<=low_h; m++)
                for(n=1; n<=low_w; n++)
                    Diff_Ext.at<double>(m-1, n-1) = Diff.at<double>(m-1, n-1);
                        

//          Diff_Ext(low_h+1,1:low_w) = Diff(low_h,1:low_w);
            for(n=1; n<=low_w; n++)
                Diff_Ext.at<double>(low_h, n-1) = Diff.at<double>(low_h-1, n-1);


//          Diff_Ext(1:low_h,low_w+1) = Diff(1:low_h,low_w);
            for(m=1; m<=low_h; m++)
                Diff_Ext.at<double>(m-1, low_w) = Diff.at<double>(m-1, low_w-1);

//          Diff_Ext(low_h+1,low_w+1) = Diff_Ext(low_h,low_w);
            Diff_Ext.at<double>(low_h, low_w)=Diff_Ext.at<double>(low_h-1, low_w-1);

            for(r=1;r<=BuildLayer->FormatHeight; r++)         
            {
                y = (r-1+0.5)/ScaleH;
                y1 = floor(y+0.5)-0.5;
                y2 = y1+1;
                r1 = y1+0.5;
                r2 = r1+1;
                if(r1 == 0)
                    r1 = 1;
                for(c=1; c<=BuildLayer->FormatWidth; c++)
                {

                    x = (c-1+0.5)/ScaleW;
                    //use the closest 4 points to compute the interpolated value, if it is boundary case, extend the boundary
                    x1 = floor(x+0.5)-0.5;
                    x2 = x1+1;
                    c1 = x1+0.5;
                    c2 = c1+1;
                    //boundary case
                    if(c1 == 0)
                        c1 = 1;

                    v11 = Diff_Ext.at<double>(r1-1,c1-1);      //%top left pixel
                    v12 = Diff_Ext.at<double>(r1-1,c2-1);      //%top right pixel
                    v21 = Diff_Ext.at<double>(r2-1,c1-1);      //%bottom left pixel
                    v22 = Diff_Ext.at<double>(r2-1,c2-1);

                    //interpolate the value
                    v = v11 *(x2-x)*(y2-y) + v12*(x-x1)*(y2-y) + v21*(x2-x)*(y-y1) + v22*(x-x1)*(y-y1);
                    Up.at<double>(r-1,c-1) = v;
                }
            }
            //final = Convolute(Up, Bk_GauVar);
            Mat finalImg;
            Convolute(&finalImg, &Up, Bk_GauVar);
//            Img = Img + final;



		}

        
}
