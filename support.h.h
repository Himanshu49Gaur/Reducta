#ifndef __FILEH__
#define __FILEH__

// Include necessary headers for different operating systems
#ifdef _WIN32
    #include <Windows.h>  // Windows-specific time functions
#else
    #include <sys/time.h>  // Linux/Mac time functions
#endif

// Timer structure to measure execution time
typedef struct {
#ifdef _WIN32
    LARGE_INTEGER startTime;  // Start time for Windows
    LARGE_INTEGER endTime;    // End time for Windows
#else
    struct timeval startTime; // Start time for Linux/Mac
    struct timeval endTime;   // End time for Linux/Mac
#endif
} Timer;

// Ensure C++ compatibility by using extern "C" block
#ifdef __cplusplus
extern "C" {
#endif

// Function prototypes for vector initialization and verification
void initVector(float **vec_h, unsigned size);  // Initialize a vector with random values
void verify(float* input, unsigned num_elements, float result);  // Verify reduction results

// Timer functions for measuring execution time
void startTime(Timer* timer);  // Start timing
void stopTime(Timer* timer);   // Stop timing
float elapsedTime(Timer timer); // Calculate elapsed time

#ifdef __cplusplus
}
#endif

// Macro for handling fatal errors, prints error message and exits the program
#define FATAL(msg, ...) \
    do { \
        fprintf(stderr, "%s:%d: " msg "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
        exit(-1); \
    } while(0)

// Ensure the system has little-endian byte order for compatibility
#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

#endif
