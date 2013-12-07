Mobile CUDA

Author : Taichirou Suzuki @ Titech

This Project aims to be a Simple Memory Scheduler.

This Project migrates CUDA programs between GPUs.
The migration trigger is when your program failed to allocate memory caused by CUDA_OUT_OF_MEMORY_ERROR.
Before the migration, Mobile CUDA try to find an enough memory region on GPUs.
If Mobile CUDA find it, migrates program to the GPU and continue to execute it.

If you want to utilize this project, move to src folder and make it like this.

> make

To make a symbolic link for it, type the following command.

> make link

After you make it successfully, open a path to lib64 like this.

> export LD_LIBRARY_PATH=/path_to_this_project/lib64/:${LD_LIBRARY_PATH}

Then try to run your CUDA project or you can use CUDA sdk sample program at app/bin/

If you want more information about Mobile CUDA, please contact me; t.susuzki.accept@gmail.com

Thanks.
