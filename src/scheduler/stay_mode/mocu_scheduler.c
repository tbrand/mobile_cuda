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

int main(){
  
  int i;
  for(i = 0 ; i < PROC_NUM ; i ++){
    fork_child_proc();
    sleep(1);
  }

  wait_multi_proc();
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

    printf("-----------------------\n");
    printf(">From Parent\n");
    printf(" Child  : %d\n",pid);
    printf(" Status : %d\n",status);
    printf("-----------------------\n");
  }
}
