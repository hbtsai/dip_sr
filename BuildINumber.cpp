/*
Filename:   BuildINumber.cpp
Author:     Mave Yeap
Date:       Dec,2013
[DESCRIPTION]: BuildINumber file.
*/
#include "SubLayer.h"
#include "BuildINumber.h"

using namespace cv;


StructPatchRecordTable BuildINumber_func(int BuildINumber, StructSubLayer* AllLayers, double ScalePerLayer,int ReconPixelOverlap, int NUM_SUBLAYERS) 
{
	int HighHeight, HighWidth, Overlap, INumber, PatchNum, nn, ridx, cidx, r, c, h, w;
	int TopPortion, LeftPortion, BottomPortion, RightPortion;
	int BuildLayerPatchNumber, TotalPatchNum, AvailableLayerNum, AnnSearchPoolNumber, Dim, sigma;
    double c1, r1;
	double HighTop, HighLeft, LowTop, LowLeft, WeightSum;
//	double TopContribution, BottomContribution, LeftContribution, RightContribution;
	double TopLeftPortion, TopRightPortion, BottomLeftPortion, BottomRightPortion;
//	double TopLeftContribution, TopRightContribution, BottomLeftContribution, BottomRightContribution;
    Mat TopLeftContribution = Mat::zeros(4, 4, CV_64F);
    Mat TopRightContribution = Mat::zeros(4, 4, CV_64F);
    Mat BottomLeftContribution = Mat::zeros(4, 4, CV_64F);
    Mat BottomRightContribution = Mat::zeros(4, 4, CV_64F);
    Mat TopContribution = Mat::zeros(4, 4, CV_64F);
    Mat BottomContribution = Mat::zeros(4, 4, CV_64F);
    Mat LeftContribution = Mat::zeros(4, 4, CV_64F);
    Mat RightContribution = Mat::zeros(4, 4, CV_64F);
    bool ispc;
	StructSubLayer BuildLayer, LowLayer;
    StructPatchRecordTable PatchMappingTable;
	Mat LowPatch(4, 4, CV_64FC1);
    StructSubLayer L0 = AllLayers[0];

	/*Add an imagine layer to be reconstructed*/
    BuildLayer.INumber = BuildINumber;
    BuildLayer.TrueWidth = L0.TrueWidth * pow(ScalePerLayer, BuildINumber);
    BuildLayer.TrueHeight = L0.TrueHeight * pow(ScalePerLayer, BuildINumber);
    BuildLayer.FormatWidth = ceil(BuildLayer.TrueWidth);
    BuildLayer.FormatHeight = ceil(BuildLayer.TrueHeight);
    BuildLayer.ValidWidth = floor(BuildLayer.TrueWidth);
    BuildLayer.ValidHeight = floor(BuildLayer.TrueHeight);
    /* zeroing following struct
    BuildLayer.Conv = 
    BuildLayer.GridAsHighLayer = 
    BuildLayer.PatchRecordTable = 
    */

	/*Build PatchRecordTable for ANN Query*/
	HighHeight = BuildLayer.FormatHeight;
	HighWidth = BuildLayer.FormatWidth;
	Overlap = ReconPixelOverlap;
	/*Matlab array*/
    int rArray[HighHeight];
    int cArray[HighWidth];
    int m=0, n=0;
    for(m=0; m<=(HighHeight-4); m++)
        rArray[m]=m;
    for(n=0; n<=(HighWidth-4); n++)
        cArray[n]=n;
    if(rArray[HighHeight-4]!=HighHeight-4)
        rArray[HighHeight-3]=HighHeight-4;
    if(cArray[HighWidth-4]!=HighWidth-4)
        cArray[HighWidth-3]=HighWidth-4;

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
    for(m=1; m<=h; m++)
        for(n=1; n<=w; n++)
            LowerLayer_Grid.at<double>(m-1, n-1) = LowLayer.Conv.at<double>(m-1, n-1);

    for(m=1; m<=h; m++)
        LowerLayer_Grid.at<double>(m-1, w) = LowerLayer_Grid.at<double>(m-1, w-1);

    for(n=1; n<=w; n++)
        LowerLayer_Grid.at<double>(h, n-1) = LowerLayer_Grid.at<double>(h-1, n-1);

    LowerLayer_Grid.at<double>(h, w) = LowerLayer_Grid.at<double>(h-1, w-1);

    /*
    LowerLayer_Grid(1:h, 1:w) = LowLayer.Conv;
    LowerLayer_Grid(1:h, w+1) = LowerLayer_Grid(1:h, w); %copy the last column to the extended
    LowerLayer_Grid(h+1, 1:w) = LowerLayer_Grid(h,1:w);  %copy the last row to the extended
    LowerLayer_Grid(h+1, w+1) = LowerLayer_Grid(h,w);    %copy the right-bottom corner
    idx = 0;*/

    int idx=0;
    /*run whole image for database*/
    PatchNum = (HighHeight - 4) * (HighWidth - 4);
    INumber = BuildINumber;
    StructPatchRecordTable PRTable[PatchNum];
    for(m=0; m<PatchNum; m++)
        PRTable[m].Vector = Mat::zeros(16, 1, CV_64F);

    for(ridx = 1; ridx <= HighWidth - 4; ridx++) 
    {
    	r = ridx;
    	for(cidx = 1; cidx <= HighWidth - 4; cidx++) 
        {
    		c = cidx;
    		fprintf(stderr, "Build whole 4x4 patches for database INumber:%d r:%d c:%d\n", INumber, r, c);
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
    		if(TopPortion != 1 && LeftPortion != 1) 
            {
    			TopLeftPortion = TopPortion * LeftPortion;
    			TopRightPortion = TopPortion * RightPortion;
    			BottomLeftPortion = BottomPortion * LeftPortion;
    			BottomRightPortion = BottomPortion * RightPortion;
    			/*Matlab code*/

                for(m=r1; m<=r1+3; m++)
                    for(n=c1; n<=c1+3; n++)
                        TopLeftContribution.at<double>(m-r1, n-c1)=LowerLayer_Grid.at<double>(m-1, n-1)*TopLeftPortion;

                for(m=r1; m<=r1+3; m++)
                    for(n=c1+1; n<=c1+4; n++)
                        TopRightContribution.at<double>(m-r1, n-(c1+1))=LowerLayer_Grid.at<double>(m-1, n-1)*TopRightPortion;

                for(m=r1+1; m<=r1+4; m++)
                    for(n=c1; n<=c1+3; n++)
                        BottomLeftContribution.at<double>(m-(r1+1), n-c1)=LowerLayer_Grid.at<double>(m-1, n-1)*BottomLeftPortion;

                for(m=r1+1; m<=r1+4; m++)
                    for(n=c1+1; n<=c1+4; n++)
                        BottomLeftContribution.at<double>(m-(r1+1), n-(c1+1))=LowerLayer_Grid.at<double>(m-1, n-1)*BottomRightPortion;

    			/*TopLeftContribution     = LowerLayer_Grid(r1  :r1+3,c1  :c1+3) * TopLeftPortion;
                TopRightContribution    = LowerLayer_Grid(r1  :r1+3,c1+1:c1+4) * TopRightPortion;
                BottomLeftContribution  = LowerLayer_Grid(r1+1:r1+4,c1  :c1+3) * BottomLeftPortion;
                BottomRightContribution = LowerLayer_Grid(r1+1:r1+4,c1+1:c1+4) * BottomRightPortion;*/
                for(m=0; m<4; m++)
                    for(n=0; n<4; n++)
                        LowPatch.at<double>(m, n) = TopLeftContribution.at<double>(m, n) + TopRightContribution.at<double>(m, n) + BottomLeftContribution.at<double>(m, n) + BottomRightContribution.at<double>(m, n);
    		} 
            else if(TopPortion != 1 && LeftPortion == 1) 
            {
    			/*TopContribution = LowerLayer_Grid(r1  :r1+3,c1:c1+3) * TopPortion;
    			BottomContribution = LowerLayer_Grid(r1+1:r1+4,c1:c1+3) * BottomPortion;*/
                for(m=r1; m<=r1+3; m++)
                    for(n=c1; n<=c1+3; n++)
                        TopContribution.at<double>(m-r1, n-c1)=LowerLayer_Grid.at<double>(m-1, n-1)*TopPortion;

                for(m=r1+1; m<=r1+4; m++)
                    for(n=c1; n<=c1+3; n++)
                        TopContribution.at<double>(m-(r1+1), n-c1)=LowerLayer_Grid.at<double>(m-1, n-1)*BottomPortion;

                for(m=0; m<4; m++)
                    for(n=0; n<4; n++)
    			        LowPatch.at<double>(m, n) = TopContribution.at<double>(m,n) + BottomContribution.at<double>(m, n);
    		} 
            else if(TopPortion == 1 && LeftPortion != 1) 
            {
    			/*LeftContribution = LowerLayer_Grid(r1:r1+3,c1  :c1+3) * LeftPortion;
    			RightContribution = LowerLayer_Grid(r1:r1+3,c1+1:c1+4) * RightPortion;
    			LowPatch = LeftContribution + RightContribution;
                */

                for(m=r1; m<=r1+3; m++)
                    for(n=c1; n<=c1+3; n++)
                        LeftContribution.at<double>(m-r1, n-c1)=LowerLayer_Grid.at<double>(m-1, n-1)*LeftPortion;

                for(m=r1; m<=r1+3; m++)
                    for(n=c1+1; n<=c1+4; n++)
                        LeftContribution.at<double>(m-r1, n-(c1+1))=LowerLayer_Grid.at<double>(m-1, n-1)*RightPortion;

                for(m=0; m<4; m++)
                    for(n=0; n<4; n++)
                        LowPatch.at<double>(m, n)=LeftContribution.at<double>(m, n)+RightContribution.at<double>(m, n);
    		} 
            else 
            {
    			/*LowPatch = LowerLayer_Grid(r1:r1+3,c1:c1+3);*/
                for(m=r1; m<=r1+3; m++)
                    for(n=c1; n<c1+3; n++)
                        LowPatch.at<double>(m-r1, n-c1)=LowerLayer_Grid.at<double>(m-1, n-1);
    		}
    		/*save the vector and r, c, i*/
    		PRTable[idx].HighPatch5x5_r = r;
    		PRTable[idx].HighPatch5x5_c = c;
    		PRTable[idx].HighPatch5x5_INumber = INumber;
    		PRTable[idx].Vector = LowPatch.reshape(16,1);
            idx++;
    	}
    }

    AllLayers[1].PatchRecordTable = PRTable;

#if 0
    /*run part of the patch for reconstrunction*/
    StructPatchRecordTable ret;
    PatchNum = length(rArray) * length(cArray);
    for(ridx = 1; ridx <= length(rArray); ridx++) {
    	r = rArray(ridx);
    	for(cidx = 1; cidx <= length(cArray); cidx++) {
    		c = cArray(cidx);
    		fprintf(stderr, "Build ANN source data BuildINumber:%d r:%d c:%d\n", INumber, r, c);
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
    /*return ret;*/
#endif


#if 0
    BuildLayerPatchNumber = sizeof(ret);
    Mat AnnSource(16 , BuildLayerPatchNumber, CV_64FC1);
    for(int j = 1; j <= BuildLayerPatchNumber; j++) {
        for(int i = 1; i <= 16; i++) {
            AnnSource.at<double>(i, j) = ret[i].Vector;
        }
    }

    delete ret;
    /*Matlab code need to convert
    SaveFileName = fullfile(TempDataFolder, 'AnnSource.txt');
    dlmwrite( SaveFileName , AnnSource' , ' ');
    clear AnnSource SaveFileName*/

    /*Get ANN search pool from stored data*/
    TotalPatchNum = 0;
    AvailableLayerNum = sizeof(AllLayers) - 1;
    for(int i = 1; i <= AvailableLayerNum; i++) {
        PatchNumForALayer = sizeof(AllLayers(i + 1).Vector);
        TotalPatchNum += PatchNumForALayer;
    }

#endif

    /*Matlab code*/
    /*PatchRecordTable(TotalPatchNum,1) = struct( 'Vector' , zeros(16,1) , 'HighPatch5x5_r' , 0 , 'HighPatch5x5_c' , 0 , 'HighPatch5x5_INumber' , 0);       %16 is the low patchsize, 3 for the corresponding high-res location (INumber,top,left)
    idx = 1;
    %Collect all available PatchRecords
    for i=1:AvailableLayerNum
        PatchNumForALayer = length(AllLayers(i+1).PatchRecordTable);
        PatchRecordTable(idx:idx+PatchNumForALayer-1) = AllLayers(i+1).PatchRecordTable;
        idx = idx + PatchNumForALayer;
    end*/

#if 0

    StructPatchRecordTable ret;
    AnnSearchPoolNumber = sizeof(ret);
    Mat AnnSearchPool(16, AnnSearchPoolNumber);
    for(int j = 1; j <= AnnSearchPoolNumber; j++) {
        for(int i = 1; i <= 16; i++) {
            AnnSearchPool.at<double>(i, j) = ret[i].Vector;
        }
    }
    /*Matlab code*/
    /*SaveFileName = fullfile(TempDataFolder, 'AnnSearchPool.txt');
    dlmwrite( SaveFileName , AnnSearchPool' , ' ');

    DataFileName = fullfile(TempDataFolder, 'AnnSearchPool.txt');
    Data = dlmread(DataFileName);
    MaxInstance = size(Data,1);
    clear Data
    currentfolder = pwd;*/

    if(ispc) {
        /*Matlab code*/
        /*AnnPath = fullfile('Lib','Ann_Windows');*/
    } else if(isunix) {
        /*Matlab code*/
        /*AnnPath = fullfile('Lib','Ann_Linux');*/
    }
    /*cd(AnnPath)*/
    Dim = 16;
    /*QueryFileName = fullfile(currentfolder, TempDataFolder, 'AnnSource.txt');
    DataFileName = fullfile(currentfolder, TempDataFolder, 'AnnSearchPool.txt');
    OutputFileName = fullfile(currentfolder, TempDataFolder, 'AnnResult.txt');*/

    fprintf(stderr, "Doing ANN\n");

    /*if ispc
        ExecString = ['ann_sample.exe -d ' num2str(Dim) ' -max ' num2str(MaxInstance) ' -nn ' num2str(nn) ' -df ' DataFileName ' -qf ' QueryFileName ' -sa ' OutputFileName ];
        [s, w] = dos( ExecString );
    elseif isunix
        ExecString = ['./ann_sample -d ' num2str(Dim) ' -max ' num2str(MaxInstance) ' -nn ' num2str(nn) ' -df ' DataFileName ' -qf ' QueryFileName ' -sa ' OutputFileName ];
        [s, w] = system( ExecString );
    end
    cd( currentfolder);

    AnnResultFileName = fullfile(TempDataFolder, 'AnnResult.txt');
    AnnResult = dlmread( AnnResultFileName , ',' );
    */
    PatchNum = sizeof(AnnResult);
    /*col = size( AnnResult ,2);*/
    nn = (col - 1) / 2;
    if(PatchMappingTable) {
        delete PatchMappingTable;
    }
    fprintf(stderr, "Buidling PatchMappingTable\n");
    StructPatchRecordTable PatchMappingTable;
    for(int idx = 1; idx <= PatchNum; idx++) {
        for (int i = 1; i <= nn; i++) {
            int ColPosition = i * 2;
            IndexInPatchRecordTable = AnnResult( idx , ColPosition ) + 1;
            AnnDiff = AnnResult( idx , ColPosition+1 );
            INumber = ret(IndexInPatchRecordTable).HighPatch5x5_INumber;
            r = ret(IndexInPatchRecordTable).HighPatch5x5_r;
            c = ret(IndexInPatchRecordTable).HighPatch5x5_c;
            PatchMappingTable.HighPatch5x5_r = r;
            PatchMappingTable.HighPatch5x5_c = c;
            PatchMappingTable.HighPatch5x5_INumber = INumber;
            PatchMappingTable.Diff = AnnDiff;
        }
    }

    /*%delete Ann files, they are too fat for BSD300
    delete(fullfile(TempDataFolder, 'AnnResult.txt'));
    delete(fullfile(TempDataFolder, 'AnnSearchPool.txt'));
    delete(fullfile(TempDataFolder, 'AnnSource.txt'));*/

    /*Compute the corresponding high 5x5 patch by exp(-SSD/sigma)*/
    Mat ImageSum(BuildLayer.FormatHeight, BuildLayer.FormatWidth);
    Mat ImageWeight(BuildLayer.FormatHeight, BuildLayer.FormatWidth);
    sigma = SSD_Sigma;
    /*for ridx=1:length(rArray)
        r = rArray(ridx);
        for cidx=1:length(cArray)
            c = cArray(cidx);*/
            fprintf(stderr, "Computing the built high-res patch r:%d c:%d\n", r, c);
            Mat PatchSum = zeros(5, CV_64F);
            WeightSum = 0;
            for(int n = 1; n <= nn; n++) {
                PatchMappingTable
            }

            
                for n=1:nn
                    INumber = PatchMappingTable(idx,n).HighPatch5x5_INumber;
                    Found_r = PatchMappingTable(idx,n).HighPatch5x5_r;
                    Found_c = PatchMappingTable(idx,n).HighPatch5x5_c;
                    Diff = PatchMappingTable(idx,n).Diff;
                    Weight = exp(-Diff/sigma);
                    Patch = AllLayers(-INumber+BuildINumber).GridAsHighLayer(Found_r:Found_r+4,Found_c:Found_c+4);
                    %insert 2 lines
                    ImageSum(r:r+4,c:c+4) = ImageSum(r:r+4,c:c+4) + Weight*Patch;%BuiltPatch{r,c};
                    ImageWeight(r:r+4,c:c+4) = ImageWeight(r:r+4,c:c+4) + Weight;%ones(5);
                end
            end
        end
#endif
}

