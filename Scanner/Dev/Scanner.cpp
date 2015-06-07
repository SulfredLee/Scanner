// Scanner.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <math.h>
#include <algorithm>
#include <fstream>

std::string fileName;
cv::Mat image, orig, gray, edged, warped, warped2;
std::vector<std::vector<cv::Point> > cnts;
double ratio;
double invRatio;
double newHeight;
std::vector<cv::Vec4i> hierarchy;
std::vector<std::vector<cv::Point> > screenCnt;
cv::RNG rng(12345);

#ifndef UNICODE  
typedef std::string String;
#else
typedef std::wstring String;
#endif

double MatMean(const cv::Mat& frame){
	return cv::mean(frame).val[0];
}
std::string Argv2String(const _TCHAR* temp){
	String Arg = temp;
	std::string out(Arg.begin(), Arg.end());
	return out;
}
void GetRatio(){
	ratio = image.size().height / newHeight;
	invRatio = 1 / ratio;
}
bool SortVecPoint(const std::vector<cv::Point>& first, const std::vector<cv::Point>& sencond){	
	return (cv::contourArea(first) > cv::contourArea(sencond)); // from large area to small area
}
void DrawCont(const std::vector<std::vector<cv::Point> >& inCont){
	cv::Mat drawing = cv::Mat::zeros(edged.size(), CV_8UC3);
	for (int i = 0; i< inCont.size(); i++)
	{
		cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, inCont, i, color, 2, 8, hierarchy, 0, cv::Point());
		imshow("Contours", drawing);
		if (cv::waitKey(30) >= 0) break;
	}
}
void FindTopLeft(std::vector<cv::Point>& pts){
	double sum = pts[0].x + pts[0].y;
	size_t minIndex = 0;
	for (size_t i = 1; i < pts.size(); i++){
		if (sum > pts[i].x + pts[i].y){
			sum = pts[i].x + pts[i].y;
			minIndex = i;
		}
	}
	cv::Point temp = pts[minIndex];
	pts[minIndex] = pts[0];
	pts[0] = temp;
}
void FindBottomRight(std::vector<cv::Point>& pts){
	double sum = pts[0].x + pts[0].y;
	size_t maxIndex = 0;
	for (size_t i = 1; i < pts.size(); i++){
		if (sum < pts[i].x + pts[i].y){
			sum = pts[i].x + pts[i].y;
			maxIndex = i;
		}
	}
	cv::Point temp = pts[maxIndex];
	pts[maxIndex] = pts[2];
	pts[2] = temp;
}
void FindTopRight(std::vector<cv::Point>& pts){
	if (pts[1].y > pts[3].y){
		cv::Point temp = pts[1];
		pts[1] = pts[3];
		pts[3] = temp;
	}
}
void order_points(std::vector<cv::Point> pts, cv::Point& TL, cv::Point& TR, cv::Point& BR, cv::Point& BL){
	// The order of these functions cannot be changed
	// Four points order is
	// pts[0] = TopLeft, pts[1] = TopRight, pts[2] = BottomRight, pts[3] = BottomLeft
	FindTopLeft(pts);
	FindBottomRight(pts);
	FindTopRight(pts);

	TL = pts[0];
	TR = pts[1];
	BR = pts[2];
	BL = pts[3];
}
cv::Mat four_point_transform(const cv::Mat& image, const std::vector<cv::Point>& pts, const double& ratio){
	cv::Mat outMat;
	cv::Point TL, TR, BR, BL;

	order_points(pts, TL, TR, BR, BL);
	TL *= ratio;
	TR *= ratio;
	BR *= ratio;
	BL *= ratio;

	double widthA = std::sqrt((BR.x - BL.x)*(BR.x - BL.x) + (BR.y - BL.y)*(BR.y - BL.y));
	double widthB = std::sqrt((TR.x - TL.x)*(TR.x - TL.x) + (TR.y - TL.y)*(TR.y - TL.y));
	int maxWidth = std::max(static_cast<int>(widthA), static_cast<int>(widthB));

	double heightA = std::sqrt((TR.x - BR.x)*(TR.x - BR.x) + (TR.y - BR.y)*(TR.y - BR.y));
	double heightB = std::sqrt((TL.x - BL.x)*(TL.x - BL.x) + (TL.y - BL.y)*(TL.y - BL.y));
	int maxHeight = std::max(static_cast<int>(heightA), static_cast<int>(heightB));

	cv::Point2f orgAry[4];
	orgAry[0] = TL; 
	orgAry[1] = TR;
	orgAry[2] = BR; 
	orgAry[3] = BL;

	cv::Point2f dstAry[4];
	dstAry[0] = cv::Point(0, 0);
	dstAry[1] = cv::Point(maxWidth - 1, 0);
	dstAry[2] = cv::Point(maxWidth - 1, maxHeight - 1);
	dstAry[3] = cv::Point(0, maxHeight - 1);

	cv::Mat M = cv::getPerspectiveTransform(orgAry, dstAry);
	cv::warpPerspective(image, outMat, M, cv::Size(maxWidth, maxHeight));
	return outMat;
}
int GetRelativBlockSize(const cv::Mat& img){
	int imgArea = img.size().height * img.size().width;
	imgArea /= (512 + 128);

	if (imgArea % 2 == 0){
		imgArea++;
	}
	return imgArea;
}
int GetFileSize(const std::string& fileName){
	std::ifstream FH(fileName, std::ifstream::ate | std::ifstream::binary);
	return FH.tellg();
}
std::vector<cv::Mat> SplitImage(cv::Mat tempWarped, const int& parts){
	int width = tempWarped.size().width;
	int height = tempWarped.size().height;

	width /= parts;
	height /= parts;

	std::vector<cv::Mat> subImages;
	int x = 0;
	int y = 0;
	for (int i = 0; i < parts; i++){
		x = 0;
		for (int j = 0; j < parts; j++){
			cv::Mat subImage(tempWarped, cv::Rect(x, y, width, height));
			subImages.push_back(subImage);
			x += width;
		}

		y += height;
	}

	return subImages;
}
void BackGroundOff2(cv::Mat& subImage, const double& threshold){
	//std::vector<std::vector<unsigned char> > tempMat;
	//for (size_t i = 0; i < subImage.rows; i++){
	//	std::vector<unsigned char> tempV;
	//	subImage.row(i).copyTo(tempV);
	//	tempMat.push_back(tempV);
	//}
	for (int i = 0; i < subImage.rows; i++){
		for (int j = 0; j < subImage.cols; j++){
			if (subImage.data[subImage.step[0] * i + subImage.step[1]*j] > threshold){
				subImage.data[subImage.step[0] * i + subImage.step[1] * j] = 250;
			}
		}
	}
}
void BackGroundOff(std::vector<cv::Mat>& subImages, const int& shift){
	for (size_t i = 0; i < subImages.size(); i++){
		double mean = MatMean(subImages[i]);

		BackGroundOff2(subImages[i], mean + shift);
	}

}
int _tmain(int argc, _TCHAR* argv[])
{
	if (argc < 2){
		std::cerr << "Usage: *.exe [image]\n";
		exit(0);
	}

	newHeight = 500.0;

	fileName = Argv2String(argv[1]);
	image = cv::imread(fileName, CV_LOAD_IMAGE_COLOR);
	GetRatio();
	image.copyTo(orig);
	cv::resize(image, image, cv::Size(), invRatio, invRatio);


	cv::cvtColor(image, gray, CV_BGR2GRAY);
	cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
	//cv::Canny(gray, edged, 75, 200);
	int temptemp = 30;
	cv::Canny(gray, edged, temptemp, temptemp*2);

	std::cout << "STEP 1: Edge Detection\n";
	//cv::imshow("Image", image);
	//cv::imshow("Gray", gray);
	//cv::imshow("Edged", edged);
	//cv::waitKey(0);
	//cv::destroyAllWindows();

	//cv::findContours(edged, cnts, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	cv::findContours(edged, cnts, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	std::sort(cnts.begin(), cnts.end(), SortVecPoint);
	//DrawCont(cnts);

	for (size_t i = 0; i < cnts.size(); i++){
		double peri = cv::arcLength(cnts[i], true);
		std::vector<cv::Point> approx;
		cv::approxPolyDP(cv::Mat(cnts[i]), approx, 0.02 * peri, true);

		if (approx.size() == 4){
			screenCnt.push_back(approx);
			break;
		}
	}

	std::cout << "STEP 2: Find contours of paper\n";
	cv::drawContours(image, screenCnt, 0, (0, 255, 0), 2);
	//cv::imshow("Outline", image);
	//cv::waitKey(0);
	//cv::destroyAllWindows();

	//std::cout << screenCnt[0] << "\n";
	//std::cout << cv::Mat(screenCnt[0]).reshape(4, 2) << "\n";
	//std::cout << cv::Mat(screenCnt[0]).reshape(4, 2) << "\n";
	//cv::waitKey(0);
	//cv::destroyAllWindows();
	warped = four_point_transform(orig, screenCnt[0], ratio);

	cv::cvtColor(warped, warped, CV_BGR2GRAY);
	cv::adaptiveThreshold(warped, warped, 250, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 11, 5);
	cv::imwrite("Final.png", warped);

	std::cout << "STEP 3: Apply perspective transform\n";
	
	//cv::imshow("Original", orig);
	//cv::imshow("Scanned", warped);
	//if (cv::waitKey(30) >= 0) exit(0);
	//cv::imshow("Before", image);

	

	//cv::imshow("After", image);
	//cv::waitKey(0);
	return 0;
}

