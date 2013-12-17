#include <opencv/cv.h>

using namespace cv;

class SubLayer
{
	public:
		int iNumber;
		int TrueWidth;
		int TrueHeight;
		int FormatWidth;
		int FormatHeight;
		int ValidWidth;
		int ValidHeight;
		Mat* Conf;
};
