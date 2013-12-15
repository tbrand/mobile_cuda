#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <nvml.h>

#define PROC_NUM 20
#define DEV_NUM 4
#define PATH_TO_PROG "../../app/0_Simple/matrixMul/matrixMul"

#define MATRIX_MEMORY 2123//[MB]

int status;

nvmlDevice_t dev[DEV_NUM];
nvmlMemory_t mem;

void fork_child_proc();
int can_sub_proc();

int main(){

  int counter = 0;

  nvmlReturn_t res;

  res = nvmlInit();

  int i;
  for(i = 0 ; i < DEV_NUM ; i ++){
    res = nvmlDeviceGetHandleByIndex(i,&dev[i]);
  }

  while(can_sub_proc()){
    fork_child_proc();
    counter ++;
    printf("Process %d launch.\n",counter);
    sleep(5);
  }

  printf("First Step End...\n");

  while(counter++ < PROC_NUM){
    pid_t res;
    int status;
    res = wait(&status);
    printf("Process %d finished.\n",counter);
    if(can_sub_proc()){
      fork_child_proc();
    }
  }

  printf("Every processes completed!\n");

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

void fork_child_proc(){

  pid_t child = fork();
  
  if(child == 0){
    printf("This process is child.\n");
    execl(PATH_TO_PROG,NULL);
    exit(0);
  }
}
