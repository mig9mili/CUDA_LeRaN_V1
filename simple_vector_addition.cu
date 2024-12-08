#include<stdio.h>


__global__ void vectoradd(float* a , float *b , float *res , int n ){

 // get the  gloable thread ID 
 int id = blockIdx.x * blockDim.x + threadIdx.x;

 if(id < n){
    res[id] = a[id] + b[id] ;
 }
}

int main(){
     
     int n = 1000;
     size_t size = n* sizeof(float);

     //host vector
     float *h_a , *h_b , *h_result;

     //devic vetor 
     float  *d_a , *d_b , *d_result;

     //allocate memory for host vector
     
     h_a = (float*)malloc(size);
     h_b = (float*)malloc(size);
     h_result = (float*)malloc(size);

     //intializ the host vector

     for(int i =0; i < n ; i++){
        h_a[i] = rand() /(float)RAND_MAX ;
        h_b[i] = rand() /(float)RAND_MAX ;
     }

     //allocate memory for device vectors
     cudaMalloc(&d_a , size);
     cudaMalloc(&d_b, size);
     cudaMalloc(&d_result , size);

     //copy input to device 

     cudaMemcpy(d_a , h_a ,size , cudaMemcpyHostToDevice);
     cudaMemcpy(d_b, h_b ,size , cudaMemcpyHostToDevice);

     int threadsPerblock = 256 ;

    int blocksPerGrid = (n + threadsPerblock -1) / threadsPerblock ;

    vectoradd<<<blocksPerGrid , threadsPerblock >>>(d_a , d_b , d_result , n);
    

    cudaMemcpy(h_result,d_result, size, cudaMemcpyDeviceToHost);

    for(int i =0 ; i <5 ; i++){
        printf("%.2f + %.2f = %.2f \n " , h_a[i],h_b[i],h_result[i] );
    }
  


cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_result);

free(h_a);
free(h_b);
free(h_result);

return 0;
}