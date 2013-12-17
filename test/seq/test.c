#include <stdio.h>		
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

#define PATH_TO_PROG "../../app/0_Simple/matrixMul/matrixMul"
#define PROC_NUM 20

pid_t my_pid;
int status;
int proc_counter;

void fork_orig_proc();
pid_t wait_proc();

static float elapsed(struct timeval tv0,struct timeval tv1){
	return (float)(tv1.tv_sec - tv0.tv_sec)
		+ (float)(tv1.tv_usec - tv0.tv_usec)
		* 0.000001f;
}

int main(){

  struct timeval tv0,tv1;

  gettimeofday(&tv0,NULL);

  proc_counter = 0;

  fork_orig_proc();

  int fin = 0;  

  while(!fin){
    pid_t res;

    res = wait_proc();

    if(my_pid == res){

      proc_counter ++;

      if(proc_counter < PROC_NUM){
	fork_orig_proc();
      }else if(proc_counter == PROC_NUM){
	fin = 1;
      }
    }
  }

  gettimeofday(&tv1);

  printf("Result time : %f[sec]\n",elapsed(tv0,tv1));

  return 0;
}

void fork_orig_proc(){
  my_pid = fork();
  if(my_pid < 0){
    exit(-1);
  }else if(my_pid == 0){

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
