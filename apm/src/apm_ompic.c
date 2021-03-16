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
 #include <mpi.h>
 #include <omp.h>
 
  #define MIN(a, b) ((a) < (b) ? (a) : (b) )


 // MACRO for calculating the time
 #define DIFFTEMPS(a,b) (((b).tv_sec - (a).tv_sec) + ((b).tv_usec - (a).tv_usec)/1000000.)
 #define APM_DEBUG 0
 
 char * 
 read_input_file( char * filename, int * size )
 {
     char * buf ;
     off_t fsize;
     int fd = 0 ;
     int n_bytes = 1 ;
 
     /* Open the text file */
     fd = open( filename, O_RDONLY ) ;
     if ( fd == -1 ) 
     {
         fprintf( stderr, "Unable to open the text file <%s>\n", filename ) ;
         return NULL ;
     }
 
 
     /* Get the number of characters in the textfile */
     fsize = lseek(fd, 0, SEEK_END);
     if ( fsize == -1 )
     {
         fprintf( stderr, "Unable to lseek to the end\n" ) ;
         return NULL ;
     }
 
 #if APM_DEBUG
     printf( "File length: %lld\n", fsize ) ;
 #endif
 
     /* Go back to the beginning of the input file */
     if ( lseek(fd, 0, SEEK_SET) == -1 ) 
     {
         fprintf( stderr, "Unable to lseek to start\n" ) ;
         return NULL ;
     }
 
     /* Allocate data to copy the target text */
     buf = (char *)malloc( fsize * sizeof ( char ) ) ;
     if ( buf == NULL ) 
     {
         fprintf( stderr, "Unable to allocate %ld byte(s) for main array\n",
                 fsize ) ;
         return NULL ;
     }
 
     n_bytes = read( fd, buf, fsize ) ;
     if ( n_bytes != fsize ) 
     {
         fprintf( stderr, 
                 "Unable to copy %ld byte(s) from text file (%d byte(s) copied)\n",
                 fsize, n_bytes) ;
         return NULL ;
     }
 
 #if APM_DEBUG
     printf( "Number of read bytes: %d\n", n_bytes ) ;
 #endif
 
     *size = n_bytes ;
 
 
     close( fd ) ;
 
 
     return buf ;
 }
 
 
 #define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))
 #define MIN(a, b) ((a) < (b) ? (a) : (b) )
 
 
 int levenshtein(char *s1, char *s2, int len, int * column) {
     unsigned int x, y, lastdiag, olddiag;
 
     for (y = 1; y <= len; y++)
     {
         column[y] = y;
     }
     for (x = 1; x <= len; x++) {
         column[0] = x;
         lastdiag = x-1 ;
         for (y = 1; y <= len; y++) {
             olddiag = column[y];
             column[y] = MIN3(
                     column[y] + 1, 
                     column[y-1] + 1, 
                     lastdiag + (s1[y-1] == s2[x-1] ? 0 : 1)
                     );
             lastdiag = olddiag;
 
         }
     }
     return(column[len]);
 }

char* cuda_malloc_cp(char *buf, int size);
int* cuda_malloc(int size);
int  finalcudaCall(char* cpattern,char * cbuf, int cuda_end, int * results_th ,int * results, int  nth_b,int nblock);
 void kernelCall(char * cpattern, char * cbuf, int cuda_end, int n_bytes, int size_pattern, int approx_factor, int * results_th,int * column_th, int nth_b, int nblock, int max_pat);
void AssignDevices(int rank);
void cuda_free( void *buf );
size_t freeMem();

int main( int argc, char ** argv ){


  char ** pattern ;
  char * filename ;
  char * local_buf;
  int local_buf_size;
  int buf_size; //buf size for each process without shadow cells (a part from rank 0)
  int max_pat = 0; // size of the largest Pattern
  int approx_factor = 0 ;
  int nb_patterns = 0 ;
  int i,j ;
  struct timeval t0, t1, t2;
  int n_bytes ;
  int * n_matches, *glob_matches ;
  int rank, size;
  double ratio = 0.5;
  

  MPI_Init (&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);	/* who am i */  
  MPI_Comm_size(MPI_COMM_WORLD, &size); /* number of processes */ 
  
   /* Check number of arguments */
  if ( argc < 4 ){
    if(rank == 0 )
      printf( "Usage: %s approximation_factor "
              "dna_database pattern1 pattern2 ...\n", 
              argv[0] ) ;
    MPI_Finalize();
    return 1 ;
   }

  /* Get the distance factor */
  approx_factor = atoi( argv[1] ) ;

   /* Grab the filename containing the target text */
  filename = argv[2] ;  

   /* Get the number of patterns that the user wants to search for */
  nb_patterns = argc - 3 ;
 

   /* Fill the pattern array */
  pattern = (char **)malloc( nb_patterns * sizeof( char * ) ) ;
  if ( pattern == NULL ){
    fprintf(stderr, 
            "Unable to allocate array of pattern of size %d\n", 
            nb_patterns ) ;
    return 1 ;
  }

   
   /* Grab the patterns */
  for ( i = 0 ; i < nb_patterns ; i++ ){
      int l = strlen(argv[i+3]) ;
      
      if( l > max_pat )
        max_pat = l;
       
      if ( l <= 0 ){
        fprintf( stderr, "Error while parsing argument %d\n", i+3 ) ;
        return 1 ;
      }
 
      pattern[i] = (char *)malloc( (l+1) * sizeof( char ) ) ;
      if ( pattern[i] == NULL ){
        fprintf( stderr, "Unable to allocate string of size %d\n", l ) ;
        return 1 ;
      }

      strncpy( pattern[i], argv[i+3], (l+1) ) ;
  }
 
  /* Allocate the array of matches */
  n_matches = (int *)malloc( nb_patterns * sizeof( int ) ) ;
  glob_matches = (int *)malloc( nb_patterns * sizeof( int ) ) ;

  if ( n_matches == NULL || glob_matches == NULL){
    fprintf( stderr, "Error: unable to allocate memory for %ldB\n",
            nb_patterns * sizeof( int ) ) ;
   return 1 ;
  }
 
  // Reading and distributing the file by Root
  if( rank == 0 ){
    char * buf ;
    printf( "Approximate Pattern Mathing: "
            "looking for %d pattern(s) in file %s w/ distance of %d\n", 
            nb_patterns, filename, approx_factor ) ;
 
    buf = read_input_file( filename, &n_bytes ) ;
    if ( buf == NULL ) return 1 ;
    
    // Start of the transfer of the data
    gettimeofday(&t0, NULL);
    buf_size = n_bytes / size;
    // Sending size of the local_buf to each Proc
    for (int to = 1; to < size; to++)
      MPI_Send(&n_bytes,1, MPI_INT, to, 0, MPI_COMM_WORLD);
    
    local_buf_size = n_bytes - buf_size*( size - 1);
    local_buf = (char *)malloc(sizeof(char)*(local_buf_size));
    strncpy(local_buf, &( buf[ buf_size*( size - 1) ] ), local_buf_size );
    
    // Sending the part of the data to each Proc
    for (int to = 1; to < size; to++){
      int start = (to-1)*buf_size;
      int end = MIN(to*buf_size + max_pat - 1, n_bytes);
      MPI_Send(&buf[start], end - start, MPI_CHAR, to, 1, MPI_COMM_WORLD);
    }
    
    buf_size = local_buf_size; //only change after data is sent
    free(buf);
  }
   // The rest should receive his part of data
  else{
    MPI_Recv(&n_bytes, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, NULL);
    buf_size = n_bytes / size;
    int start = (rank-1)*buf_size;
    int end = MIN(rank*buf_size + max_pat - 1, n_bytes);
    local_buf_size = end - start;
    local_buf = (char *)malloc(sizeof(char)*local_buf_size);
    MPI_Recv(local_buf, local_buf_size, MPI_CHAR, 0, 1, MPI_COMM_WORLD, NULL);
  }

  // Send data to gpu
  AssignDevices(rank);
  int cuda_end = ratio*buf_size;
  int nth_b = 1024;
  size_t fm = freeMem();
  if(fm < cuda_end*sizeof(char)){
    cuda_end = ratio*fm/sizeof(char) - 1;
  }
  double nth_max = ((fm/(double)size) - cuda_end*sizeof(char))/((max_pat + 10.0) * sizeof(int));
  int nblock_max = nth_max/nth_b;
  MPI_Barrier(MPI_COMM_WORLD);

  char * cbuf;
  cbuf = cuda_malloc_cp(local_buf, (cuda_end)* sizeof(char));

  int nblock = nblock_max;
  int nth = nblock*nth_b;

  int * results_th;
  results_th = cuda_malloc(nth* sizeof(int));

  int * column_th = cuda_malloc(nth * (max_pat + 1) * sizeof(int));
  /*************************** BEGIN MAIN LOOP ***************************************/
 
  /* Calculation is starting */
  if (rank == 0 ) gettimeofday(&t1, NULL);
 
   /* Check each pattern one by one */
  for ( i = 0 ; i < nb_patterns ; i++ ){

    /* Initialize the number of matches to 0 */
    n_matches[i] = 0 ;
  
    int size_pattern = strlen(pattern[i]) ;
    int num_matches_cuda = 0;
    char * cpattern;
    int num_matches = 0;
   
    
    //Send pattern to GPU
    cpattern =  cuda_malloc_cp(pattern[i],size_pattern* sizeof(char));
    kernelCall(cpattern,cbuf,cuda_end,local_buf_size,size_pattern,approx_factor,results_th,column_th ,nth_b,nblock, max_pat);
    int * results; //to hold cuda results
    results = (int * ) malloc(nth * sizeof(int));
    #pragma omp parallel
    {
      int * column = (int *)malloc( (size_pattern+1) * sizeof( int ) ) ;

      if ( column == NULL ) {
        fprintf( stderr, "Error: unable to allocate memory for column (%ldB)\n",
              (size_pattern+1) * sizeof( int ) ) ;
        exit( EXIT_FAILURE ) ;
      } // End if

      #pragma omp for reduction(+: num_matches) schedule(guided) nowait
      for ( j = cuda_end ; j < buf_size; j++ ){
        int distance = 0 ;
        int size_pat ;
 
        size_pat = size_pattern ;
        if ( local_buf_size < j + size_pattern )
          size_pat = local_buf_size - j ;
 
        distance = levenshtein( pattern[i], &local_buf[j], size_pat, column ) ;
 
        if ( distance <= approx_factor ) 
          num_matches++;

        #if APM_DEBUG
          if ( j % 100 == 0 ){
            printf( "Procesing byte %d (out of %d)\n", j, n_bytes ) ;
          }
        #endif

      }// END for j
      #pragma omp single
       finalcudaCall(cpattern,cbuf,cuda_end,results_th, results,nth_b,nblock);
      #pragma omp for reduction(+: num_matches_cuda) schedule(guided) nowait
      for (j = 0; j <MIN(cuda_end,nth); j++) {
            num_matches_cuda += results[j];
      }
      free( column );
    }//END pragma omp
    cuda_free(cpattern);
    free(results);
    n_matches[i] = num_matches + num_matches_cuda;

  } // END for i 
  MPI_Reduce(n_matches, glob_matches, nb_patterns, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

 /***************************** END MAIN LOOP ***************************************/

  if( rank == 0 ){
  
    gettimeofday(&t2, NULL);  // Timer Stop  
    printf( "\nAPM transfer done in %lf s\nAPM calculation done in %lf s\nTotal time : %lf s\n\n", DIFFTEMPS(t0,t1), DIFFTEMPS(t1,t2), DIFFTEMPS(t0,t2)) ;

    for ( i = 0 ; i < nb_patterns ; i++ )
      printf( "Number of matches for pattern <%s>: %d\n", pattern[i], glob_matches[i] ) ;
  } // END if rank
  
  // Cleaning the GPU
  cuda_free( (void *)cbuf );
  cuda_free( (void *)results_th);

   MPI_Finalize();

    
 
  return 0 ;
}// END main
