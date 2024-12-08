// code implements a parallel reduction algorithm in CUDA to sum all elements in an array
// We want to sum all elements in a large array efficiently using GPU parallelization.

__global__ void reducesum(float *input, float *output, int n)
{
    // shared memory for intermedite sums

    __shared__ float sharedData[256];

    int threadId = threadIdx.x; // thread ID withint the block
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;

    // step 1 load data in shared memory ;
    sharedData[threadId] = (globalID < n) ? input[globalID] : 0;
    __syncthreads(); // wait for all threads to load their data

    // perform parallel reduction

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadId < stride)
        {
            sharedData[threadId] += sharedData[threadId + stride];
        }

        __syncthreads(); // wait for all the addition
    }

    if (threadId == 0)
    {

        output[blockIdx.x] = sharedData[0];
    }
}

int main()
{

    // setup
    int arraysize = 1000;
    float *deviceinput, *deviceoutput;

    // calucate grid dimension
    int blocksize = 256;
    int numBlocks = (arraysize + blocksize - 1) / blocksize;

    // allocate the memory
    cudaMalloc(&deviceinput, arraysize * sizeof(float));
    cudaMalloc(&deviceoutput, numBlocks * sizeof(float));

    // lauch kerenal
    reducesum<<<numBlocks , blocksize>>>(deviceinput , deviceoutput , arraysize);

    //cleanup 
    cudaFree(deviceinput);
    cudaFree(deviceoutput);
    return 0;
    
}