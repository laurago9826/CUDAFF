#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <malloc.h>
#include <stdio.h>
#include "Timing.h"
#include <math.h>

// Image data
#define IMG_INPUT "input.bmp"
#define IMG_OUTPUT "output.bmp"
#define IMG_HEADER 1080
//#define IMG_HEADER 1500
#define IMG_WIDTH 4000
#define IMG_HEIGHT 4000

#define BLOCKDIM 1000


#define THREAD_X 32
#define THREAD_Y 32


__device__ const char SOBELX[] = { 1, 2, 1,  0, 0, 0,  -1, -2, -1};
__device__ const char SOBELY[] = { -1, 0, 1,  -2, 0, 2,  -1, 0, 1};

#pragma endregion

__global__ void DoCalculating(unsigned char *img, unsigned char *img2)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ unsigned char *shr_img;

	int pos2 = pos - IMG_WIDTH;
	int pos3 = pos + IMG_WIDTH;

	short x = 0;
	short y = 0;
	short newColor = 0;

	x =
		img[pos2 - 1] * SOBELX[0] +
		img[pos2] * SOBELX[1] +
		img[pos2 + 1] * SOBELX[2] +
		img[pos - 1] * SOBELX[3] +
		img[pos] * SOBELX[4] +
		img[pos + 1] * SOBELX[5] +
		img[pos3 - 1] * SOBELX[6] +
		img[pos3] * SOBELX[7] +
		img[pos3 + 1] * SOBELX[8];

	y =
		img[pos2 - 1] * SOBELY[0] +
		img[pos2] * SOBELY[1] +
		img[pos2 + 1] * SOBELY[2] +
		img[pos - 1] * SOBELY[3] +
		img[pos] * SOBELY[4] +
		img[pos + 1] * SOBELY[5] +
		img[pos3 - 1] * SOBELY[6] +
		img[pos3] * SOBELY[7] +
		img[pos3 + 1] * SOBELY[8];

	newColor = sqrt((float)(x * x + y * y));

	if (newColor > 255)
		newColor = 255;
	else if (newColor < 0)
		newColor = 0;
	//newColor = 128;
	img2[pos] = newColor;
}


__global__ void DoCalculatingShared(unsigned char *img, unsigned char *img2)
{
	//int pos = blockIdx.y * blockDim.y + blockDim.y.x;

	__shared__ char shr_sobelx[9];
	__shared__ char shr_sobely[9];




	//int pos = blockDim.y * blockIdx.y + threadIdx.x ; // elsõ sor
	//int pos = blockDim.x * (blockIdx.x + threadIdx.y) + threadIdx.x ; // általános
	//int pos = blockIdx.y * IMG_WIDTH + threadIdx.x;
	int pos = (blockIdx.x) * blockDim.x + threadIdx.x;


	if (threadIdx.x < 9 && threadIdx.y == 0) {
		shr_sobelx[threadIdx.x] = SOBELX[threadIdx.x];
		shr_sobely[threadIdx.x] = SOBELY[threadIdx.x];
	}
	__syncthreads();


	/*shr_img[pos] = img[pos];
	//int shr_pos = pos - blockDim.x * blockIdx.x;
*/

	short newColor = 125;
	//if (threadIdx.y == 1) {
		int pos2 = pos - IMG_WIDTH;
		int pos3 = pos + IMG_WIDTH;

		short x = 0;
		short y = 0;

		//x =
		//	img[pos2 - 1] * SOBELX[0] +
		//	img[pos2] * SOBELX[1] +
		//	img[pos2 + 1] * SOBELX[2] +
		//	img[pos - 1] * SOBELX[3] +
		//	img[pos] * SOBELX[4] +
		//	img[pos + 1] * SOBELX[5] +
		//	img[pos3 - 1] * SOBELX[6] +
		//	img[pos3] * SOBELX[7] +
		//	img[pos3 + 1] * SOBELX[8];

		//y =
		//	img[pos2 - 1] * SOBELY[0] +
		//	img[pos2] * SOBELY[1] +
		//	img[pos2 + 1] * SOBELY[2] +
		//	img[pos - 1] * SOBELY[3] +
		//	img[pos] * SOBELY[4] +
		//	img[pos + 1] * SOBELY[5] +
		//	img[pos3 - 1] * SOBELY[6] +
		//	img[pos3] * SOBELY[7] +
		//	img[pos3 + 1] * SOBELY[8];


		x =
			img[pos2 - 1] * shr_sobelx[0] +
			img[pos2] * shr_sobelx[1] +
			img[pos2 + 1] * shr_sobelx[2] +
			img[pos - 1] * shr_sobelx[3] +
			img[pos] * shr_sobelx[4] +
			img[pos + 1] * shr_sobelx[5] +
			img[pos3 - 1] * shr_sobelx[6] +
			img[pos3] * shr_sobelx[7] +
			img[pos3 + 1] * shr_sobelx[8];
		y =
			img[pos2 - 1] * shr_sobely[0] +
			img[pos2] * shr_sobely[1] +
			img[pos2 + 1] * shr_sobely[2] +
			img[pos - 1] * shr_sobely[3] +
			img[pos] * shr_sobely[4] +
			img[pos + 1] * shr_sobely[5] +
			img[pos3 - 1] * shr_sobely[6] +
			img[pos3] * shr_sobely[7] +
			img[pos3 + 1] * shr_sobely[8];


		newColor = sqrt((float)(x * x + y * y));

		//if (newColor > 255)
		//	newColor = 255;
		//else if (newColor < 0)
		//	newColor = 0;
			
	//}
	img2[pos] = newColor;
}

__global__ void DoCalculatingShared2(unsigned char *img, unsigned char *img2)
{
	__shared__ unsigned char shr_img[THREAD_X + 2][THREAD_Y + 2];
	__shared__ char shr_sobelx[9];
	__shared__ char shr_sobely[9];

	//int pos = (blockIdx.x + threadIdx.y) * blockDim.x + threadIdx.x;
	//int pos = blockIdx.x * blockDim.x + threadIdx.y * IMG_WIDTH + threadIdx.x;
	int pos = blockIdx.x * THREAD_X * THREAD_Y + threadIdx.x + threadIdx.y * THREAD_Y; 

	char newTx = threadIdx.x + 1;
	char newTy = threadIdx.y + 1;

	shr_img[newTx][newTy] = img[pos];


	if (threadIdx.x < 9 && threadIdx.y == 0) {
		shr_sobelx[threadIdx.x] = SOBELX[threadIdx.x];
		shr_sobely[threadIdx.x] = SOBELY[threadIdx.x];
	}
	__syncthreads();

	if (newTx == 1)
		//shr_img[0][newTy - 1] = img[pos - 1];
		shr_img[0][newTy - 1] = img[pos + 1];
	else if (newTx == THREAD_X - 2)
		shr_img[THREAD_Y + 1][newTy + 1] = img[pos + 1];

	if (newTy == 1)
		shr_img[newTx - 1][0] = img[pos - IMG_WIDTH];
	else if (newTy == THREAD_Y - 2)
		shr_img[newTx + 1][THREAD_X + 1] = img[pos + IMG_WIDTH];

	if (newTx == 1 && newTy == 1)
		shr_img[newTx - 1][newTy - 1] = img[pos - IMG_WIDTH - 1];
	else if (newTx == THREAD_X - 1 && newTy == THREAD_Y - 1)
		shr_img[newTx + 1][newTy + 1] = img[pos + IMG_WIDTH + 1];
	else if (newTx == 1 && newTy == THREAD_Y - 1)
		shr_img[newTx - 1][newTy + 1] = img[pos + IMG_WIDTH - 1];
	else if (newTx == THREAD_X - 1 && newTy == 1)
		shr_img[newTx + 1][newTy - 1] = img[pos - IMG_WIDTH + 1];
	__syncthreads();

	short newColor = 0;
	short x = 0;
	short y = 0;

	//newTx++;
	//newTy++;

	x =
		shr_img[newTx - 1][newTy - 1] * shr_sobelx[0] +
		shr_img[newTx][newTy - 1] * shr_sobelx[1] +
		shr_img[newTx + 1][newTy - 1] * shr_sobelx[2] +
		shr_img[newTx - 1][newTy] * shr_sobelx[3] +
		shr_img[newTx][newTy] * shr_sobelx[4] +
		shr_img[newTx + 1][newTy] * shr_sobelx[5] +
		shr_img[newTx - 1][newTy + 1] * shr_sobelx[6] +
		shr_img[newTx][newTy + 1] * shr_sobelx[7] +
		shr_img[newTx + 1][newTy + 1] * shr_sobelx[8];

	y =
		shr_img[newTx - 1][newTy - 1] * shr_sobely[0] +
		shr_img[newTx][newTy - 1] * shr_sobely[1] +
		shr_img[newTx + 1][newTy - 1] * shr_sobely[2] +
		shr_img[newTx - 1][newTy] * shr_sobely[3] +
		shr_img[newTx][newTy] * shr_sobely[4] +
		shr_img[newTx + 1][newTy] * shr_sobely[5] +
		shr_img[newTx - 1][newTy + 1] * shr_sobely[6] +
		shr_img[newTx][newTy + 1] * shr_sobely[7] +
		shr_img[newTx + 1][newTy + 1] * shr_sobely[8];
	newColor = sqrt((float)(x * x + y * y));

	if (newColor > 255)
		newColor = 255;
	else if (newColor < 0)
		newColor = 0;

	//newColor = shr_img[newTx][newTy];
	img2[pos] = newColor;
}


__global__ void DoCalculatingShared3(unsigned char *img, unsigned char *img2)
{
	__shared__ unsigned char shr_img[BLOCKDIM * 3];
	__shared__ char shr_sobelx[9];
	__shared__ char shr_sobely[9];

	//int pos = (blockIdx.x + threadIdx.y) * blockDim.x + threadIdx.x;
	int pos = blockIdx.x * blockDim.x + (threadIdx.x + threadIdx.y * blockDim.x);

	int shr_pos = threadIdx.x + BLOCKDIM;
	int pos1 = shr_pos;

	shr_img[shr_pos - BLOCKDIM] = img[pos - IMG_WIDTH];
	shr_img[shr_pos] = img[pos];
	shr_img[shr_pos + BLOCKDIM] = img[pos + IMG_WIDTH];

	if (threadIdx.x < 9) {
		shr_sobelx[threadIdx.x] = SOBELX[threadIdx.x];
		shr_sobely[threadIdx.x] = SOBELY[threadIdx.x];
	}
	__syncthreads();


	short newColor = 0;
	int pos2 = shr_pos - BLOCKDIM;
	int pos3 = shr_pos + BLOCKDIM;

	short x = 0;
	short y = 0;

	if (threadIdx.x != 0 && threadIdx.x != blockDim.x - 1) {
		x =
			shr_img[pos2 - 1] * shr_sobelx[0] +
			shr_img[pos2] * shr_sobelx[1] +
			shr_img[pos2 + 1] * shr_sobelx[2] +
			shr_img[pos1 - 1] * shr_sobelx[3] +
			shr_img[pos1] * shr_sobelx[4] +
			shr_img[pos1 + 1] * shr_sobelx[5] +
			shr_img[pos3 - 1] * shr_sobelx[6] +
			shr_img[pos3] * shr_sobelx[7] +
			shr_img[pos3 + 1] * shr_sobelx[8];

		y =
			shr_img[pos2 - 1] * shr_sobely[0] +
			shr_img[pos2] * shr_sobely[1] +
			shr_img[pos2 + 1] * shr_sobely[2] +
			shr_img[pos1 - 1] * shr_sobely[3] +
			shr_img[pos1] * shr_sobely[4] +
			shr_img[pos1 + 1] * shr_sobely[5] +
			shr_img[pos3 - 1] * shr_sobely[6] +
			shr_img[pos3] * shr_sobely[7] +
			shr_img[pos3 + 1] * shr_sobely[8];
		newColor = sqrt((float)(x * x + y * y));

		if (newColor > 255)
			newColor = 255;
		else if (newColor < 0)
			newColor = 0;
		//newColor = 125;
	}
	img2[pos] = newColor;
}


void DetectEdges(unsigned char *img)
{
	unsigned char *d_img; //Allocate device memory
	unsigned char *d_num2;
	cudaMalloc((void**)&d_img, sizeof(unsigned char)*IMG_WIDTH*IMG_HEIGHT); 
	cudaMalloc((void**)&d_num2, sizeof(unsigned char)*IMG_WIDTH*IMG_HEIGHT); 
	cudaMemcpy(d_img, img + IMG_HEADER, sizeof(unsigned char)*IMG_WIDTH*IMG_HEIGHT, cudaMemcpyHostToDevice); //Memory copy H->D
	//DoCalculating << < 40000, dim3(4000, 4000) >> > (d_img, d_num2); //Launch kernel
	//DoCalculatingShared << < dim3(IMG_WIDTH / 32 + 1, IMG_HEIGHT / 32 + 1), dim3(32, 32) >> > (d_img, d_num2); //Launch kernel
	//DoCalculatingShared << <IMG_HEIGHT, dim3(IMG_WIDTH, 3) >> > (d_img, d_num2); //Launch kernel
	DoCalculatingShared2 << <15625, dim3(THREAD_X,THREAD_Y) >> > (d_img, d_num2); //Launch kernel
	//DoCalculatingShared << <15625, 1024 >> > (d_img, d_num2); //Launch kernel
	//DoCalculatingShared3 << <16000, 1000 >> > (d_img, d_num2); //Launch kernel
	cudaMemcpy(img + IMG_HEADER, d_num2, sizeof(unsigned char)*IMG_WIDTH*IMG_HEIGHT, cudaMemcpyDeviceToHost); //Memory copy D->H
	cudaFree(d_img); //Free device memory
}


int main()
{
	unsigned char *img;
	FILE *f_input_img, *f_output_img;

	// Load image
	img = (unsigned char*)malloc(IMG_HEADER + sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT);


	fopen_s(&f_input_img, IMG_INPUT, "rb");
	fread(img, 1, IMG_HEADER + IMG_WIDTH * IMG_HEIGHT, f_input_img);
	fclose(f_input_img);

	// Run single stream kernel
	MEASURE_TIME(1, "Edges", DetectEdges(img));

	// Save file
	fopen_s(&f_output_img, IMG_OUTPUT, "wb");
	fwrite(img, 1, IMG_HEADER + IMG_WIDTH * IMG_HEIGHT, f_output_img);
	fclose(f_output_img);
	free(img);

}
