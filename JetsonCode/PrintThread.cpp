#include "PrintThread.h"
#include "cuda.h"
#include "Definitions.h"
#include "Settings.h"

void print_thread(){
	printf("print_thread: started\n");

	float* tempArray;
	float* tempArray2;
	while(true){
		while(Settings::sleeping && Settings::connected && !Settings::force_exit){}
		if (Settings::force_exit) break;
			
		cudaMalloc(&tempArray, sizeof(float)*Settings::values[STG_WIDTH]*Settings::values[STG_HEIGHT]);
		cudaMalloc(&tempArray2, sizeof(float)*Settings::values[STG_WIDTH]*Settings::values[STG_HEIGHT]);
		float* output = (float*)malloc(sizeof(float)*Settings::values[STG_WIDTH]*Settings::values[STG_HEIGHT]);
		float* output2 = (float*)malloc(sizeof(float)*Settings::values[STG_WIDTH]*Settings::values[STG_HEIGHT]);
		cv::namedWindow("Basic Visualization", CV_WINDOW_NORMAL);
		cv::setWindowProperty("Basic Visualization", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
		//set the callback function for any mouse event
     	cv::setMouseCallback("Basic Visualization", CallBackFunc, NULL);

		while(!Settings::initialized && Settings::connected && !Settings::force_exit){}
		if (Settings::force_exit) break;

		while(!Settings::sleeping && Settings::connected){
			if(cycles >= 3){
				cycles = 0;
				mtx.lock();
				cudaMemcpy(tempArray, maximaGreen, sizeof(float)*Settings::values[STG_WIDTH]*Settings::values[STG_HEIGHT], cudaMemcpyDeviceToDevice);
				cudaMemcpy(tempArray2, outputArray, sizeof(float)*Settings::values[STG_WIDTH]*Settings::values[STG_HEIGHT], cudaMemcpyDeviceToDevice);
				mtx.unlock();
				cudaMemcpy(output, tempArray, sizeof(float)*Settings::values[STG_WIDTH]*Settings::values[STG_HEIGHT], cudaMemcpyDeviceToHost);
				cudaMemcpy(output2, tempArray2, sizeof(float)*Settings::values[STG_WIDTH]*Settings::values[STG_HEIGHT], cudaMemcpyDeviceToHost);
				const cv::Mat img(cv::Size(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT]), CV_32F, output);
				const cv::Mat img2(cv::Size(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT]), CV_32F, output2);

				const cv::Mat img2_trans(cv::Size(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT]), CV_32F);
				const cv::Mat img_trans(cv::Size(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT]), CV_32F);

				cv::flip(img2, img2_trans, -1);
				cv::transpose(img2_trans, img2);
				cv::flip(img, img_trans, -1);
				cv::transpose(img_trans, img);


				const cv::Mat result = img2+img;
				cv::imshow("Basic Visualization", result);
				cv::waitKey(1);
			}
			else{
				usleep(5000);
			}

			if (Settings::force_exit) break;
		}
		cudaFree(tempArray);
		cudaFree(tempArray2);
		free(output);
		free(output2);
	}

	printf("print_thread: ended\n");
}
