#ifndef PROFILE_HPP
#define PROFILE_HPP

#include <stdio.h>
#include <time.h>
#include <sys/time.h>


#define TIME_IT(ROUTINE_NAME__, LOOPS__, ACTION__)                                                                              \
    {                                                                                                                           \
        printf("\nTiming '%s' started\n", ROUTINE_NAME__);                                                                      \
        const clock_t startTime = clock();                                                                                      \
        for (int loops = 0; loops < (LOOPS__); ++loops) ACTION__;                                                               \
        const clock_t endTime = clock();                                                                                        \
        const clock_t elapsedTime = endTime - startTime;                                                                        \
        const double timeInSeconds = (elapsedTime / (double)CLOCKS_PER_SEC);                                                    \
        printf("Clock Time (for %d iterations) = %g\n", LOOPS__, timeInSeconds * 1000000 / LOOPS__);                               \
        printf("Timing '%s' ended\n", ROUTINE_NAME__);                                                                          \
    }


#endif /* PROFILE_HPP */
