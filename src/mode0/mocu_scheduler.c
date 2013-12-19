#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <nvml.h>

#define PROC_NUM 20
#define DEV_NUM 4
#define PATH_TO_PROG   "../../app/0_Simple/matrixMul/matrixMul"
#define PATH_TO_PROG2  "../../app/orig/sample"
#define PATH_TO_PROG3  "../../app/0_Simple/matrixMulSmall/matrixMulSmall"

#define MATRIX_MEMORY    2123//[MB]

int status;

nvmlDevice_t dev[DEV_NUM];
nvmlMemory_t mem;

void fork_child_proc();
int can_sub_proc();

static float elapsed(struct timeval tv0,struct timeval tv1){
	return (float)(tv1.tv_sec - tv0.tv_sec)
		+ (float)(tv1.tv_usec - tv0.tv_usec)
		* 0.000001f;
}

int main(){

  int launched_proc = 0;
  int received_proc = 0;

  nvmlReturn_t res;

  res = nvmlInit();

  int i;
  for(i = 0 ; i < DEV_NUM ; i ++){
    res = nvmlDeviceGetHandleByIndex(i,&dev[i]);
  }

  struct timeval tv0,tv1;

  gettimeofday(&tv0);

  while(can_sub_proc() && launched_proc < DEV_NUM*2){
    fork_child_proc();
    launched_proc++;
    printf("Process %d launch.\n",launched_proc);
    sleep(6);
  }

  printf("First Step End...\n");

  while(launched_proc++ < PROC_NUM){
    pid_t res;
    int status;

    res = wait(&status);

    received_proc++;

    printf("Process %d finished.\n",received_proc);

    if(can_sub_proc()){
      fork_child_proc();
      printf("Process %d launch.\n",launched_proc);
    }
  }

  printf("Second Step End...\n");

  while(received_proc++ < PROC_NUM){
    pid_t res;
    int status;

    res = wait(&status);
    printf("Process %d finished.\n",received_proc);
  }

  printf("Every processes completed!\n");

  gettimeofday(&tv1);

  printf("Result time : %f[sec]\n",elapsed(tv0,tv1));

}

int can_sub_proc(){

  nvmlReturn_t res;

  int i;
  for(i = 0 ; i < DEV_NUM ; i ++){
    long memFree;

    res = nvmlDeviceGetMemoryInfo(dev[i],&mem);
    memFree = mem.free >> 20;

    if(memFree - MATRIX_MEMORY > 0){
      return 1;
    }
  }
  return 0;
}

int switch_counter = 0;

void fork_child_proc(){

  switch_counter++;

  pid_t child = fork();
  
  if(child == 0){
    printf("This process is child.\n");
    if(switch_counter%2 == 0)
      execl(PATH_TO_PROG,NULL);
    else
      execl(PATH_TO_PROG2,NULL);
    exit(0);
  }
}
