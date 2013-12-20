/*
Filename:   BuildINumber.cpp
Author:     Mave Yeap
Date:       Dec,2013
[DESCRIPTION]: BuildINumber file.
*/

using namespace cv;

StructPatchRecordTable BuildINumber(StructSubLayer* L0, StructSubLayer* AllLayers, double ScalePerLayer,int ReconPixelOverlap, int NUM_SUBLAYERS) {
	int HighHeight, HighWidth, Overlap, INumber, ridx, r, c, h, w;
	int TopPortion, LeftPortion, BottomPortion, RightPortion;
	int BuildLayerPatchNumber;
	double HighTop, HighLeft, LowTop, LowLeft;
	double TopContribution, BottomContribution, LeftContribution, RightContribution;
	double TopLeftPortion, TopRightPortion, BottomLeftPortion, BottomRightPortion;
	double TopLeftContribution, TopRightContribution, BottomLeftContribution, BottomRightContribution;
	StructSubLayer BuildLayer, LowLayer;
	Mat LowPatch(4, 4, CV_64FC1);
	/*Add an imagine layer to be reconstructed*/
	BuildLayer.INumber = BuildINumber;
	BuildLayer.TrueWidth = L0.TrueWidth * pow(ScalePerLayer, BuildINumber);
	BuildLayer.TrueHeight = L0.TrueHeight * pow(ScalePerLayer, BuildINumber);
	BuildLayer.FormatWidth = ceil(BuildLayer.TrueWidth);
	BuildLayer.FormatHeight = ceil(BuildLayer.TrueHeight);
	BuildLayer.ValidWidth = floor(BuildLayer.TrueWidth);
	BuildLayer.ValidHeight = floor(BuildLayer.TrueHeight);
	BuildLayer.Conv = [];
	BuildLayer.GridAsHighLayer = [];
	BuildLayer.PatchRecordTable = [];
	/*Build PatchRecordTable for ANN Query*/
	HighHeight = BuildLayer.FormatHeight;
	HighWidth = BuildLayer.FormatWidth;
	Overlap = ReconPixelOverlap;
	/*Matlab array*/
	/*rArray = 1:5-Overlap:HighHeight-4;
    if rArray(end) ~= HighHeight-4
        rArray = [1:5-Overlap:HighHeight-4 HighHeight-4];       %the last patch has to be done
    end
    cArray = 1:5-Overlap:HighWidth-4;
    if cArray(end) ~= HighWidth-4
        cArray = [1:5-Overlap:HighWidth-4 HighWidth-4];
    end*/
    LowLayer = AllLayers[1];
    h = LowLayer.FormatHeight;
    w = LowLayer.FormatWidth;
    Mat LowerLayer_Grid = Mat::zeros(h + 1, w + 1, CV_64F);
    /*Matlab Code*/
    /*LowerLayer_Grid(1:h, 1:w) = LowLayer.Conv;
    LowerLayer_Grid(1:h, w+1) = LowerLayer_Grid(1:h, w); %copy the last column to the extended
    LowerLayer_Grid(h+1, 1:w) = LowerLayer_Grid(h,1:w);  %copy the last row to the extended
    LowerLayer_Grid(h+1, w+1) = LowerLayer_Grid(h,w);    %copy the right-bottom corner
    idx = 0;*/

    /*run whole image for database*/
    PatchNum = (HighHeight - 4) * (HighWidth - 4);
    INumber = BuildINumber;
    StructPatchRecordTable ret;
    for(ridx = 1; ridx <= HighWidth - 4; ridx++) {
    	r = ridx;
    	for(cidx = 1; cidx <= HighWidth - 4; cidx++) {
    		c = cidx;
    		fprintf("Build whole 4x4 patches for database INumber:%d r:%d c:%d\n", INumber, r, c);
    		/*Compute the grid of low*/
    		HighTop = r - 1;
    		HighLeft = c - 1;
    		LowTop = HighTop / ScalePerLayer;
    		LowLeft = HighLeft / ScalePerLayer;
    		r1 = floor(LowTop) + 1;
    		c1 = floor(LowLeft) + 1;
    		/*compute the corresponding 4*4 grid in low-res patch*/
    		if(floor(LowTop) == LowTop) {
    			TopPortion = 1;
    		} else {
    			TopPortion = ceil(LowTop) - LowTop;
    		}
    		if(floor(LowLeft) == LowLeft) {
    			LeftPortion = 1;
    		} else {
    			LeftPortion = ceil(LowLeft) - LowLeft;
    		}
    		BottomPortion = 1 - TopPortion;
    		RightPortion = 1 - LeftPortion;
    		if(TopPortion != 1 && LeftPortion != 1) {
    			TopLeftPortion = TopPortion * LeftPortion;
    			TopRightPortion = TopPortion * RightPortion;
    			BottomLeftPortion = BottomPortion * LeftPortion;
    			BottomRightPortion = BottomPortion * RightPortion;
    			/*Matlab code*/
    			/*TopLeftContribution     = LowerLayer_Grid(r1  :r1+3,c1  :c1+3) * TopLeftPortion;
                TopRightContribution    = LowerLayer_Grid(r1  :r1+3,c1+1:c1+4) * TopRightPortion;
                BottomLeftContribution  = LowerLayer_Grid(r1+1:r1+4,c1  :c1+3) * BottomLeftPortion;
                BottomRightContribution = LowerLayer_Grid(r1+1:r1+4,c1+1:c1+4) * BottomRightPortion;*/
                LowPatch = TopLeftContribution + TopRightContribution + BottomLeftContribution + BottomRightContribution;
    		} else if(TopPortion != 1 && LeftPortion == 1) {
    			/*TopContribution = LowerLayer_Grid(r1  :r1+3,c1:c1+3) * TopPortion;
    			BottomContribution = LowerLayer_Grid(r1+1:r1+4,c1:c1+3) * BottomPortion;*/
    			LowPatch = TopContribution + BottomContribution;
    		} else if(TopPortion == 1 && LeftPortion != 1) {
    			/*LeftContribution = LowerLayer_Grid(r1:r1+3,c1  :c1+3) * LeftPortion;
    			RightContribution = LowerLayer_Grid(r1:r1+3,c1+1:c1+4) * RightPortion;*/
    			LowPatch = LeftContribution + RightContribution;
    		} else {
    			/*LowPatch = LowerLayer_Grid(r1:r1+3,c1:c1+3);*/
    		}
    		/*save the vector and r, c, i*/
    		ret.HighPatch5x5_r = r;
    		ret.HighPatch5x5_c = c;
    		ret.HighPatch5x5_INumber = INumber;
    		ret.Vector = LowPatch.reshape(16,1);
    	}
    }
    delete ret;
    /*run part of the patch for reconstrunction*/
    StructPatchRecordTable ret;
    PatchNum = length(rArray) * length(cArray);
    for(ridx = 1; ridx <= length(rArray); ridx++) {
    	r = rArray(ridx);
    	for(cidx = 1; cidx <= length(cArray); cidx++) {
    		c = cArray(cidx);
    		fprintf("Build ANN source data BuildINumber:%d r:%d c:%d\n", INumber, r, c);
    		/*Compute the grid of low*/
    		HighTop = r - 1;
    		HighLeft = c - 1;
    		LowTop = HighTop / ScalePerLayer;
    		LowLeft = HighLeft / ScalePerLayer;
    		r1 = floor(LowTop) + 1;
    		c1 = floor(LowLeft) + 1;
    		/*compute the corresponding 4x4 grid in low-res patch*/
    		if(floor(LowTop) == LowTop) {
    			TopPortion = 1;
    		} else {
    			TopPortion = ceil(LowTop) - LowTop;
    		}
    		if(floor(LowLeft) == LowLeft) {
    			LeftPortion = 1;
    		} else {
    			LeftPortion = ceil(LowLeft) - LowLeft;
    		}
    		BottomPortion = 1 - TopPortion;
    		RightPortion = 1- LeftPortion;
    		if(TopPortion != 1 && LeftPortion != 1) {
    			TopLeftPortion = TopPortion * LeftPortion;
    			TopRightPortion = TopPortion * RightPortion;
    			BottomLeftPortion = BottomPortion * LeftPortion;
    			BottomRightPortion = BottomPortion * RightPortion;
    			/*TopLeftContribution = LowerLayer_Grid(r1  :r1+3,c1  :c1+3) * TopLeftPortion;
    			TopRightContribution = LowerLayer_Grid(r1  :r1+3,c1+1:c1+4) * TopRightPortion;
    			BottomLeftContribution = LowerLayer_Grid(r1+1:r1+4,c1  :c1+3) * BottomLeftPortion;
    			BottomRightContribution = LowerLayer_Grid(r1+1:r1+4,c1+1:c1+4) * BottomRightPortion;*/
    			LowPatch = TopLeftContribution + TopRightContribution + BottomLeftContribution + BottomRightContribution;
    		} else if(TopPortion != 1 && LeftPortion == 1) {
    			/*TopContribution = LowerLayer_Grid(r1  :r1+3,c1:c1+3) * TopPortion;
    			BottomContribution = LowerLayer_Grid(r1+1:r1+4,c1:c1+3) * BottomPortion;*/
    			LowPatch = TopContribution + BottomContribution;
    		} else if(TopPortion == 1 && LeftPortion != 1) {
    			/*LeftContribution = LowerLayer_Grid(r1:r1+3,c1  :c1+3) * LeftPortion;
    			RightContribution = LowerLayer_Grid(r1:r1+3,c1+1:c1+4) * RightPortion;*/
    			LowPatch = LeftContribution + RightContribution;
    		} else {
    			LowPatch = LowerLayer_Grid(r1:r1+3,c1:c1+3);
    		}
    		/*save the vector and r, c, i*/
    		ret.HighPatch5x5_r = r;
    		ret.HighPatch5x5_c = c;
    		ret.HighPatch5x5_INumber = INumber;
    		ret.Vector = LowPatch.reshape(16,1);
    	}
    }
    return ret;
}


