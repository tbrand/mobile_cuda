Mobile CUDA (mocu for short)

Author : Taichirou Suzuki @ Titech

https://github.com/tbrand/mobile_cuda.git

* Simple GPGPU Program Scheduler.
* Migrate processes betweeen GPUs.
* Mobile CUDA has 2 modes as triggers of migration.
* See top of the libcuda.c to check the description of each 2 modes.

* To utilize this project:

1. @src
>make

2. @src
>make link

3. @somewhere
>export LD_LIBRARY_PATH=/path_to_mobile_cuda/lib64:${LD_LIBRARY_PATH}

or you can use script like this

>source env.sh

#WARNING#

Do NOT call cudaSetDevice() in your program, there is some possibility of failing to migrate.	

This project depends on CUDA driver version, see Makefile @src.
This project using original libcuda.so (usually locate at /usr/lib64/libcuda.so) in libcuda.c, if you locate it another path, fix it before make it.
The script 'env.sh' depends on your environmental.

Mobile CUDA is NOT complete project, it contains many bugs.
And some program (using CUDA) will be failed to migrate.

If you want to get more information abount Mobile CUDA, please contact with me. t.suzuki.accept@gmail.com

Thanks.
