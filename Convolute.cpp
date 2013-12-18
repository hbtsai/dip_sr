#include <math.h>
#include <opencv/cv.h>

using namespace cv;

struct MyPoint
{
	int X; 
	int Y; 
	int ColIdx; 
	int RowIdx; 
};

void Convolute(Mat* inputArray, Mat* outputArray, double GaussianVariant)
{

	double Sigma = sqrt(GaussianVariant);
	int height=inputArray->rows;
	int width=inputArray->cols;
	double YCoor, XCoor; 
	double EffectiveRadius;
	MyPoint MostLeftTopSourcePoint, MostRightBottomSourcePoint;

	int m=0, n=0;
	for(m=0; m<height; m++)
	{
		for(n=0; n<width; n++)
		{
            YCoor = (m-0.5);
            XCoor = (n-0.5);
            EffectiveRadius = Sigma * 4;
            MostLeftTopSourcePoint.X = XCoor - EffectiveRadius;
            MostLeftTopSourcePoint.Y = YCoor - EffectiveRadius;
            MostRightBottomSourcePoint.X = XCoor + EffectiveRadius;
            MostRightBottomSourcePoint.Y = YCoor + EffectiveRadius;

            //trim
            if(MostLeftTopSourcePoint.X < 0.5)
                MostLeftTopSourcePoint.X = 0.5;

            if(MostLeftTopSourcePoint.Y < 0.5)
                MostLeftTopSourcePoint.Y = 0.5;

            if(MostRightBottomSourcePoint.X > width - 0.5)
                MostRightBottomSourcePoint.X = width - 0.5;

            if(MostRightBottomSourcePoint.Y > height - 0.5)
                MostRightBottomSourcePoint.Y = height - 0.5;

            //Convert Coordinate to grid index
            MostLeftTopSourcePoint.ColIdx = floor(MostLeftTopSourcePoint.X + 0.5);
            MostLeftTopSourcePoint.RowIdx = floor(MostLeftTopSourcePoint.Y + 0.5);
            MostRightBottomSourcePoint.ColIdx = floor(MostRightBottomSourcePoint.X + 0.5);
            MostRightBottomSourcePoint.RowIdx = floor(MostRightBottomSourcePoint.Y + 0.5);
            
            double WeightSum = 0.0;
            double IntensitySum = 0.0;
			double weight = 0.0;
			double distSqr=0.0;
			int r=0, c=0;
			double SrcY=0.0, SrcX=0.0;
            for(r = MostLeftTopSourcePoint.RowIdx; r<MostRightBottomSourcePoint.RowIdx; r++)
			{
                for(c = MostLeftTopSourcePoint.ColIdx; c<MostRightBottomSourcePoint.ColIdx; c++)
				{
                    SrcY = r - 0.5;
                    SrcX = c - 0.5;
                    distSqr = (SrcY - YCoor)*(SrcY - YCoor) + (SrcX - XCoor)*(SrcX - XCoor);
                    if( distSqr < GaussianVariant * 16)
					{
                        weight = exp(-distSqr/(2*GaussianVariant))/(2*M_PI*GaussianVariant);
                        WeightSum = WeightSum + weight;
                        IntensitySum = IntensitySum + weight * inputArray->at<double>(r , c );
					}
				}
			}
            outputArray->at<double>(m,n) = IntensitySum / WeightSum ;
		}
	}

	/* matlab code */
	/*
    Sigma = sqrt( Bk_GauVar );
    [height width] = size( x );
    HBk = zeros( height, width);
    for i = 1:height
        for j = 1:width
            YCoor = (i-0.5);
            XCoor = (j-0.5);
            EffectiveRadius = Sigma * 4;
            MostLeftTopSourcePoint.X = XCoor - EffectiveRadius;
            MostLeftTopSourcePoint.Y = YCoor - EffectiveRadius;
            MostRightBottomSourcePoint.X = XCoor + EffectiveRadius;
            MostRightBottomSourcePoint.Y = YCoor + EffectiveRadius;
            %trim
            if MostLeftTopSourcePoint.X < 0.5
                MostLeftTopSourcePoint.X = 0.5;
            end
            if MostLeftTopSourcePoint.Y < 0.5
                MostLeftTopSourcePoint.Y = 0.5;
            end
            if MostRightBottomSourcePoint.X > width - 0.5
                MostRightBottomSourcePoint.X = width - 0.5;
            end
            if MostRightBottomSourcePoint.Y > height - 0.5
                MostRightBottomSourcePoint.Y = height - 0.5;
            end
            
            %Convert Coordinate to grid index
            MostLeftTopSourcePoint.ColIdx = floor(MostLeftTopSourcePoint.X + 0.5);
            MostLeftTopSourcePoint.RowIdx = floor(MostLeftTopSourcePoint.Y + 0.5);
            MostRightBottomSourcePoint.ColIdx = floor(MostRightBottomSourcePoint.X + 0.5);
            MostRightBottomSourcePoint.RowIdx = floor(MostRightBottomSourcePoint.Y + 0.5);
            
            WeightSum = 0;
            IntensitySum = 0;
            for r = MostLeftTopSourcePoint.RowIdx:MostRightBottomSourcePoint.RowIdx
                for c = MostLeftTopSourcePoint.ColIdx:MostRightBottomSourcePoint.ColIdx
                    SrcY = r - 0.5;
                    SrcX = c - 0.5;
                    distSqr = (SrcY - YCoor)^2 + (SrcX - XCoor)^2;
                    if( distSqr < Bk_GauVar * 16);
                        weight = exp(-distSqr/(2*Bk_GauVar))/(2*pi*Bk_GauVar);
                        WeightSum = WeightSum + weight;
                        IntensitySum = IntensitySum + weight * x(r , c );
                    end
                end
            end
            HBk(i,j) = IntensitySum / WeightSum ;
        end
    end
	*/
}
