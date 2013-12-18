#include <opencv/cv.h>

using namespace cv;


class StructPatchRecordTable
{
	double Vector[16];
	int HighPatch5x5_r;
	int HighPatch5x5_c;
	int HighPatch5x5_INumber;
};

class StructSubLayer
{
	public:
		int INumber;
		int TrueWidth;
		int TrueHeight;
		int FormatWidth;
		int FormatHeight;
		int ValidWidth;
		int ValidHeight;
		Mat Conv;
		Mat GridAsHighLayer;
		StructPatchRecordTable PatchRecordTable;
};
