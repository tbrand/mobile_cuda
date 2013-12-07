#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

#define PATH_TO_PROG "../app/0_Simple/matrixMul/matrixMul"
#define PROC_NUM 20
#define DEV_NUM 4

pid_t pids[DEV_NUM];
int status;
int proc_counter;

void fork_orig_proc(int);
pid_t wait_proc();

int main(){

  proc_counter = 0;

  int i;
  for(i = 0 ; i < DEV_NUM ; i ++){
    fork_orig_proc(i);
  }

  int fin = 0;

  while(!fin){
    pid_t res;
    int j;

    res = wait_proc();

    for(j = 0 ; j < DEV_NUM ; j ++){
      if(pids[j] == res){

	proc_counter ++;

	if(proc_counter + DEV_NUM <= PROC_NUM){
	  fork_orig_proc(j);
	  break;
	}else if(proc_counter == PROC_NUM){
	  fin = 1;
	  break;
	}

      }
    }
  }

  return 0;
}

void fork_orig_proc(int pos){
  pids[pos] = fork();
  if(pids[pos] < 0){
    exit(-1);
  }else if(pids[pos] == 0){

    printf("This Proc is %d\n",pos);

    if(pos == 0)
      putenv("CUDA_VISIBLE_DEVICES=0");
    else if(pos == 1)
      putenv("CUDA_VISIBLE_DEVICES=1");
    else if(pos == 2)
      putenv("CUDA_VISIBLE_DEVICES=2");
    else if(pos == 3)
      putenv("CUDA_VISIBLE_DEVICES=3");

    execl(PATH_TO_PROG,NULL);//execute matrixMul.cu

    exit(-1);
  }
}

pid_t wait_proc(){

  pid_t r = waitpid(-1,&status,0);

  if(r < 0){
    perror("waitpid");
    exit(-1);
  }

  if(WIFEXITED(status)){
    //child process complete.
    printf("child %d complete\n",proc_counter);
    return r;
  }
}
