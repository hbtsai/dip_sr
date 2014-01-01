/*
Filename:   BuildINumber.h
Author:     Mave Yeap
Date:       Dec,2013
[DESCRIPTION]: BuildINumber header file.
*/

#ifndef	__BuildINumber_H__
#define	__BuildINumber_H__

/**********************
*	Standard headers
**********************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

/**********************
*	Program headers
**********************/




/**********************
*	Program structure
**********************/
StructPatchRecordTable BuildINumber(StructSubLayer* L0, StructSubLayer* AllLayers, double ScalePerLayer,int ReconPixelOverlap, int NUM_SUBLAYERS);


/**********************
*	Program Define
**********************/


#endif
