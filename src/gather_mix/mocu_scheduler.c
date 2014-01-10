#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <nvml.h>

#define PROC_NUM 10
#define DEV_NUM 4
#define PATH_TO_PROG   "../../app/0_Simple/matrixMul/matrixMul"
//#define PATH_TO_PROG2  "../../app/orig/sample"
#define PATH_TO_PROG2 "../../app/1_Utilities/bandwidthTest/bandwidthTest"

#define MATRIX_MEMORY    2123//[MB]

int status;

nvmlDevice_t dev[DEV_NUM];
nvmlMemory_t mem;

typedef struct my_pid_time{
  pid_t my_pid;
  struct timeval start;
  struct timeval end;
} __pid;

int launched_proc = 0;
int received_proc = 0;

void fork_child_proc();
int can_sub_proc();

__pid DATA[PROC_NUM];

static float elapsed(struct timeval tv0,struct timeval tv1){
  return (float)(tv1.tv_sec - tv0.tv_sec);
    //    + (float)(tv1.tv_usec - tv0.tv_usec)
    //    * 0.000001f;
}

int main(){

  srand(110);

  nvmlReturn_t res;

  res = nvmlInit();

  int i;
  for(i = 0 ; i < DEV_NUM ; i ++){
    res = nvmlDeviceGetHandleByIndex(i,&dev[i]);
  }

  struct timeval tv0,tv1;

  gettimeofday(&tv0,NULL);

  while(can_sub_proc() && launched_proc < DEV_NUM*2){
    fork_child_proc();
    launched_proc++;
    printf("Process %d launch.\n",launched_proc);
    sleep(9);
  }

  printf("First Step End...\n");

  while(launched_proc++ < PROC_NUM){
    pid_t res;
    int status;

    res = wait(&status);

    struct timeval end;
    gettimeofday(&end,NULL);

    int j;
    for(j = 0 ; j < PROC_NUM ; j ++){
      if(DATA[j].my_pid == res){
	DATA[j].end = end;
      }
    }
    
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

    struct timeval end;
    gettimeofday(&end,NULL);

    int j;
    for(j = 0 ; j < PROC_NUM ; j ++){
      if(DATA[j].my_pid == res){
	DATA[j].end = end;
      }
    }
    printf("Process %d finished.\n",received_proc);
    printf("pid == %lld finish ... \n",res);
  }

  printf("Every processes completed!\n");

  gettimeofday(&tv1,NULL);

  printf("Result time : %f[sec]\n",elapsed(tv0,tv1));

  int k;

  unsigned long long base = DATA[0].start.tv_sec;
  
  for(k = 0 ; k < PROC_NUM ; k ++){
    DATA[k].start.tv_sec -= base;
    DATA[k].end.tv_sec -= base;
  }

  for(k = 0 ; k < PROC_NUM ; k ++){
    printf("PID == %lld\n",DATA[k].my_pid);
    printf("Start : %d(%d)\n",DATA[k].start.tv_sec,(DATA[k].start.tv_sec*4)/5);
    printf("End   : %d\n",DATA[k].end.tv_sec);
    printf("Time  : %d(%d)\n",DATA[k].end.tv_sec - DATA[k].start.tv_sec,((DATA[k].end.tv_sec - DATA[k].start.tv_sec)*4)/5);
  }

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

  int random = rand()%100;

  pid_t child = fork();
  
  if(child == 0){
    if(random < 50)
      execl(PATH_TO_PROG,NULL);
    else
      //      execl(PATH_TO_PROG2,NULL);
      execl(PATH_TO_PROG2,NULL);
    exit(0);
  }else{
    struct timeval start;
    gettimeofday(&start,NULL);
    int i;
    for(i = 0 ; i < PROC_NUM ; i ++){
      if(DATA[i].my_pid == 0){
	DATA[i].my_pid = child;
	DATA[i].start = start;
	i = PROC_NUM;
      }
    }
  }
}
