#include <stdio.h>		
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <time.h>

#define PATH_TO_PROG  "../app/0_Simple/matrixMul/matrixMul"
#define PATH_TO_PROG2 "../app/orig/memoryBound"
#define PROC_NUM 20
#define DEV_NUM 4

pid_t pids[DEV_NUM];
int status;
int proc_counter;

time_t tt;
struct tm *ts,*te;

typedef struct my_pid_time{
  pid_t my_pid;
  struct timeval start;
  struct timeval end;
  int pos;
  int proc;
} __pid;

void fork_orig_proc(int);
pid_t wait_proc();

__pid DATA[PROC_NUM];

static float elapsed(struct timeval tv0,struct timeval tv1){
  return (float)(tv1.tv_sec - tv0.tv_sec)
    + (float)(tv1.tv_usec - tv0.tv_usec)
    * 0.000001f;
}

int main(){

  tt = time(NULL);
  ts = localtime(&tt);
  printf("%02d:%02d:%02d\n", ts->tm_hour, ts->tm_min, ts->tm_sec);

  srand(110);

  struct timeval tv0,tv1;

  gettimeofday(&tv0,NULL);

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

  gettimeofday(&tv1,NULL);

  printf("Result time : %f[sec]\n",elapsed(tv0,tv1));

  tt = time(NULL);
  te = localtime(&tt);
  printf("%02d:%02d:%02d\n", te->tm_hour, te->tm_min, te->tm_sec);

  int k;

  unsigned long long base = DATA[0].start.tv_sec;
  
  for(k = 0 ; k < PROC_NUM ; k ++){
    DATA[k].start.tv_sec -= base;
    DATA[k].end.tv_sec -= base;
  }

  for(k = 0 ; k < PROC_NUM ; k ++){
    printf("PID == %lld\n",DATA[k].my_pid);
    printf("pos   : %d\n",DATA[k].pos);
    printf("proc  : %d\n",DATA[k].proc);
    printf("Start : %d(%d)\n",DATA[k].start.tv_sec,(DATA[k].start.tv_sec*4)/5);
    printf("End   : %d\n",DATA[k].end.tv_sec);
    printf("Time  : %d(%d)\n",DATA[k].end.tv_sec - DATA[k].start.tv_sec,((DATA[k].end.tv_sec - DATA[k].start.tv_sec)*4)/5);
  }

  return 0;
}

int switch_counter = 0;

void fork_orig_proc(int pos){

  int random = rand()%100;

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

    if(random < 50)
      execl(PATH_TO_PROG,NULL);//execute matrixMul.cu
    else
      execl(PATH_TO_PROG2,NULL);

    exit(-1);
  }else{
    struct timeval start;
    gettimeofday(&start,NULL);
    int i;
    for(i = 0 ; i < PROC_NUM ; i ++){
      if(DATA[i].my_pid == 0){
	if(random < 50){
	  DATA[i].proc = 0;
	}else{
	  DATA[i].proc = 1;
	}
	DATA[i].my_pid = pids[pos];
	DATA[i].start = start;
	DATA[i].pos = pos;
	i = PROC_NUM;
      }
    }
  }
}

pid_t wait_proc(){

  pid_t r = waitpid(-1,&status,0);

  if(r < 0){
    perror("waitpid");
    exit(-1);
  }

  struct timeval end;
  gettimeofday(&end,NULL);

  int j;
  for(j = 0 ; j < PROC_NUM ; j ++){
    if(DATA[j].my_pid == r){
      DATA[j].end = end;
    }
  }

  if(WIFEXITED(status)){
    //child process complete.
    printf("child %d complete\n",proc_counter);
    return r;
  }
}
