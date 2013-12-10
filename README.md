******************************************
* Mobile CUDA (mocu for short)           *
*                                        *
* Author : Taichirou Suzuki @ Titech     *
*                                        *
* https://github.com/tbrand/mobile_cuda  *
*                                        *
******************************************

Simple Memory Scheduler for GPU programs.
Migrate processes betweeen GPUs when process failed to allocate memory region caused by CUDA_ERROR_OUT_OF_MEMORY.

To utilize this project:

1. @src

>make

2. @src

>make link

3. @somewhere

>export LD_LIBRARY_PATH=/path_to_mobile_cuda/lib64:${LD_LIBRARY_PATH}

Thanks.
