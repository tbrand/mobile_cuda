#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

#define PROC_NUM 20
#define PATH_TO_PROG "../../../app/0_Simple/matrixMul/matrixMul"

int status;
int counter = 0;
int fin = 0;

void fork_child_proc();
void wait_multi_proc();

static float elapsed(struct timeval tv0,struct timeval tv1){
	return (float)(tv1.tv_sec - tv0.tv_sec)
		+ (float)(tv1.tv_usec - tv0.tv_usec)
		* 0.000001f;
}

int main(){

  struct timeval tv0,tv1;

  gettimeofday(&tv0);
  
  int i;
  for(i = 0 ; i < PROC_NUM ; i ++){
    fork_child_proc();
    sleep(5);
  }

  wait_multi_proc();

  gettimeofday(&tv1);

  printf("Result time : %f[sec]\n",elapsed(tv0,tv1));

}

void fork_child_proc(){
  pid_t child = fork();
  
  if(child == 0){
    printf("This process is child.\n");
    execl(PATH_TO_PROG,NULL);
    exit(0);
  }
}

void wait_multi_proc(){

  int i;
  for(i = 0 ; i < PROC_NUM ; i ++){

    pid_t pid;
    int status = 0;

    pid = wait(&status);
  }
}
