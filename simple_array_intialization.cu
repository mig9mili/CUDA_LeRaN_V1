
__global__ void initarray(int* array , int n){
    // calculate the uqique thread index for threads
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
 // check if this thread should work on this thread
    if (idx < n){
        array[idx] =idx ;
    }
}

int main(){

    int n =1000;
    int *h_array; //pointer to cpu array
    int *d_array; //pointer to gpu array
  

    ///allocate memory on cpu 
    h_array = (int *)malloc(n*sizeof(int));

    //allocate memory on gpu 
    //cudaMalloc take pointer to pointer 
    cudaMalloc(&d_array , n* sizeof(int));

    // calculate how to split our data ascross GPU Threads
    int threadsPerBlock = 256;


    //calculate how many block we need 

    int blockPerGrid = (n+threadsPerBlock-1)/threadsPerBlock;

    //launch the kerenel
   ///    <<<blocks , threads>>> is cuda special synatax to kerenal launch
   initarray<<<blockPerGrid , threadsPerBlock >>>(d_array , n);
  
   //copy result back to cpu (if needed)
   cudaMemcpy(h_array, d_array , n*sizeof(int) , cudaMemcpyDeviceToHost);

   //clean up
   free(h_array);
   cudaFree(d_array);
   return 0;
}