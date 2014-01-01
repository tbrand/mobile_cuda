#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/time.h>

#define PROC_NUM 5
#define PARALLEL 5
#define PATH_TO_PROG   "../../../app/0_Simple/matrixMulSmall/matrixMulSmall"

int status;

int launched_proc = 0;
int received_proc = 0;

void fork_child_proc();

static float elapsed(struct timeval tv0,struct timeval tv1){
  return (float)(tv1.tv_sec - tv0.tv_sec);
    //    + (float)(tv1.tv_usec - tv0.tv_usec)
    //    * 0.000001f;
}

int main(){

  srand(110);

  struct timeval tv0,tv1;

  gettimeofday(&tv0,NULL);

  while(launched_proc < PARALLEL){
    fork_child_proc();
    launched_proc++;
    printf("Process %d launch.\n",launched_proc);
  }

  printf("First Step End...\n");

  while(launched_proc++ < PROC_NUM){
    pid_t res;
    int status;

    res = wait(&status);

    struct timeval end;
    gettimeofday(&end,NULL);
    
    received_proc++;

    printf("Process %d finished.\n",received_proc);

    fork_child_proc();

    printf("Process %d launch.\n",launched_proc);
  }

  printf("Second Step End...\n");

  while(received_proc++ < PROC_NUM){
    pid_t res;
    int status;

    res = wait(&status);

    struct timeval end;

    gettimeofday(&end,NULL);

    printf("Process %d finished.\n",received_proc);
  }

  printf("Every processes completed!\n");

  gettimeofday(&tv1,NULL);

  printf("Result time : %f[sec]\n",elapsed(tv0,tv1));
}

void fork_child_proc(){

  pid_t child = fork();
  
  if(child == 0){
    execl(PATH_TO_PROG,NULL);
    exit(0);
  }
}
