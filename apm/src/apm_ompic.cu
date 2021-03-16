/**
 * APPROXIMATE PATTERN MATCHING
 *
 * INF560
 */
 #include <stdio.h>
 #include <string.h>
 #include <stdlib.h>
 #include <fcntl.h>
 #include <unistd.h>
 #include <sys/time.h>
 #include <cuda_runtime.h>
 #include <cuda.h>
 
 #define APM_DEBUG 0
 #define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))
 #define MIN(a, b) ((a) < (b) ? (a) : (b) )
 
 // MACRO for checking cuda functions
 #define CHECK(x) \
 do { \
   if (x != cudaSuccess) { \
     fprintf(stderr, "%s:%d: ", __func__, __LINE__); \
     printf("%s\n",cudaGetErrorString(x)); \
     exit(EXIT_FAILURE); \
   } \
 } while (0)
  
  // Kernel function for calculation the levenshtein function
 __global__ void compMatches(char * pattern, char * buf, int cuda_end, int n_bytes, int size_pattern, int approx_factor, int * resultsth, int * column_th, int max_pat) {
 
   int distance = 0;
   int size;
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   int n_th = gridDim.x * blockDim.x;
   resultsth[i] = 0;
   //  if (i == 0) printf("cuda started \n");
   if (i < cuda_end) {
     //int * column = (int * ) malloc((size_pattern + 1) * sizeof(int));
     int * column = column_th + i*(max_pat+1);
     for (int j = i; j < cuda_end; j += n_th) {
       size = size_pattern;
       if (n_bytes - j < size_pattern) 
         size = n_bytes - j;
 
       /////////////////// Levenshtein ///////////////////////////        
       int len = size;
       unsigned int x, y, lastdiag, olddiag;
 
       for (y = 1; y <= len; y++) 
         column[y] = y;
 
       for (x = 1; x <= len; x++) {
         column[0] = x;
         lastdiag = x - 1;
         for (y = 1; y <= len; y++) {
           olddiag = column[y];
           column[y] = MIN3(
             column[y] + 1,
             column[y - 1] + 1,
             lastdiag + (pattern[y - 1] == buf[j + x - 1] ? 0 : 1)
           );
           lastdiag = olddiag;
 
         }// End for y
       }// End for x
       distance = column[len];
       /////////////// End of Levenshtein ///////////////////////
 
       if(distance <= approx_factor)
         resultsth[i]++;
       
     }// END for j
   }// END if i 
 }// END compMatches func.
 
 // Function for calling cuda kernel function
 extern "C"
 void kernelCall(char * cpattern, char * cbuf, int cuda_end, int n_bytes, int size_pattern, int approx_factor, int * results_th,int * column_th, int nth_b, int nblock, int max_pat) {
   compMatches << < nblock, nth_b >>> (cpattern, cbuf, cuda_end, n_bytes, size_pattern, approx_factor, results_th, column_th, max_pat);
   CHECK(cudaGetLastError());
 }
 
 // Function for Synchronizing cuda kernel function and freeing the buffers
 extern "C"
 int finalcudaCall(char * cpattern, char * cbuf, int cuda_end, int * results_th,int * results, int nth_b, int nblock) {
  // int * results;
   int nth = nth_b * nblock;
   //results = (int * ) malloc(nth * sizeof(int));
   CHECK(cudaDeviceSynchronize());
   //  printf("cuda done\n");
   CHECK(cudaMemcpy(results, results_th, nth * sizeof(int), cudaMemcpyDeviceToHost));
 
   //int res = 0;
   //for (int j = 0; j < nth && j < cuda_end; j++) {
     //res += results[j];
   //}
  // free(results);
  // CHECK(cudaFree(cpattern));
 
   return 0;
 }
 
 // Function for making cudaMalloc and cudaMemcpy from host
 extern "C"
 char * cuda_malloc_cp(char * buf, int size) {
   char * dBuf;
   CHECK(cudaMalloc((void ** ) & dBuf, size));
   CHECK(cudaMemcpy(dBuf, buf, size, cudaMemcpyHostToDevice));
   return dBuf;
 }
 
 // Function for making cudaMalloc from host
 extern "C"
 int * cuda_malloc(int size) {
   int * dBuf;
   CHECK(cudaMalloc((void ** ) & dBuf, size));
   return dBuf;
 }
 
 extern "C"
 void cuda_free(void * buf) {
   CHECK(cudaFree(buf));
 }
 
 // Function for assigning the proccess to Gpu
 extern "C"
 void AssignDevices(int rank) {
   int nbGPU = 0;
   CHECK(cudaGetDeviceCount( & nbGPU));
   CHECK(cudaSetDevice(rank % nbGPU));
 }

//Prints info about GPU memory
extern "C"
void checkGpuMem(int rank)
{
  float free_m,total_m,used_m;
  size_t free_t,total_t;
  cudaMemGetInfo(&free_t,&total_t);
  free_m =(uint)free_t/1048576.0 ;
  total_m=(uint)total_t/1048576.0;
  used_m=total_m-free_m;

printf ( "  meem free %d .... %f MB mem total %d....%f MB mem used %f MB, rank %d\n",free_t,free_m,total_t,total_m,used_m,rank);
}
extern "C"
size_t freeMem()
{
  size_t free_t,total_t;
  cudaMemGetInfo(&free_t,&total_t);
  return (free_t);
}
 