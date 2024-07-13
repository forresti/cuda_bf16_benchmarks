#include <iostream>
#include <iomanip>
#include <fstream>
#include <list>
#include <set>
#include <map>
#include <utility>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda.h"

#include "kernels.h"
#include "helpers.h"

using namespace std;

int main (int argc,char **argv)
{
    int repeat = 10;
    // unrolled_sanity_check(repeat);


    for(int i=1; i<4; i++){

        run_add(
            /* N = */ i*16777216,  /* 2^n is divisible by 256 */
            /* repeat = */ repeat
        );

        memcpy_ubench(
            /* N = */ i*16777216,  /* 2^n is divisible by 256 */
            /* repeat = */ repeat
        );

        std::cout << std::endl;
    }

    return 0;
}