/* Get three args as noted above.
Determine how many pages must be accessed based on MB to access and page size.
Allocate memory for an array of pointers of type char (bytes).
For each element of the array allocate a number of bytes equal to the page size.*/
#include <malloc.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char *argv[]){
    long B_TO_MB = 1000000;
    long PAGE_SIZE = 4096;

    clock_t start_t, end_t;
    srand(time(NULL)); 

    if(argc != 4) {
        printf("Usage: \n thrash  MEGABYTES  SECONDS MODIFY \n");
        return 0;
    }

    long mb = atoi(argv[1]);
    long secs = atoi(argv[2]);
    long modify = atoi(argv[3]);

    long num_pages = B_TO_MB*mb / PAGE_SIZE;

    char **t = (char**) malloc(num_pages*PAGE_SIZE);
    printf("Running \n");
    // Initial memory allocation
    long i;
    for (i = 0 ; i< num_pages ; i++){
        t[i] = (char *) malloc(PAGE_SIZE*sizeof(char));
        *t[i]='\0';
    }

    long count = 0;
    start_t = clock();
    while ((double)(clock() - start_t) / CLOCKS_PER_SEC < secs) {
        // Create a random page number
        // access the random buffer (page), changing at least one byte if MODIFY is > 0 
        for (i = 0 ; i< num_pages ; i++){
            char *page_access = (t[i]);
            if (modify > 0) (*page_access)+=1; 
            count+=1;
        }
        i=0;
    }
    end_t = clock();

    // free all memory
    for (i = 0 ; i< num_pages-1 ; i++){
        // printf("- %ld \n",i);
        free( t[i] );
    }
    free( t );
    float pages_per_second =  ((float) count) / ((float)(end_t - start_t) / CLOCKS_PER_SEC);
    printf("Done\nTotal pages per second: %f \n",pages_per_second);

    return 0;
}


