/**********************************************
 *  Mobile CUDA                               *
 *                                            *
 *  Author : Taichirou Suzuki                 *
 *                                            *
 *  using /usr/lib64/libcuda.so               *
 *                                            *
 *  https://github.com/tbrand/mobile_cuda.git *
 *                                            *
 **********************************************
 +--------------------------------------------+
 | Mobile CUDA has 2 modes                    |
 +====+=======================================+
 |MODE|Explanation                            |
 +----+---------------------------------------+
 |   0|Gathering all processes to device0.    |
 |   1|Diffuce processes from the beginning.  |
 +--------------------------------------------+*/

#define MODE 0

#include <stdio.h>
#include <dlfcn.h>
#include <string.h>

//for signal handler
#include <signal.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

//include original header file
#include <cuda.h>
#include <mocu.h>

//for shmem and semaphore
#include <sys/types.h>
#include <sys/ipc.h>
#include <unistd.h>
#include <sys/sem.h>
#include <sys/stat.h>
#include <sys/shm.h>

//for nvml
#include <nvml.h>

#define DEBUG         0
#define DEBUG_ERROR   0
#define PRINT_LOG     0
#define DEBUG_RESTORE 1
#define DEBUG_MIG 1
#define DEBUG_BACKUP 1

#define DEBUG_SEMAPHORE 0

#define D_CONTEXT  0
#define D_STREAM   0
#define D_EVENT    0
#define D_REGION   0
//#define D_ARRAY   0
#define D_TEXREF   0
#define D_FUNCTION 0
//#define D_SYMBOL   0
#define D_MODULE   0
#define D_APILOG   0

#define KEY 10103
#define PROC_NUM 64

#define KERNEL_MIG_INTERVEL 5

int mocuID = 0;

CUresult (*mocuInit)(unsigned int);
CUresult (*mocuDriverGetVersion)(int *);
CUresult (*mocuDeviceGet)(CUdevice *,int );
CUresult (*mocuDeviceGetCount)(int *);
CUresult (*mocuDeviceGetName)(char *,int ,CUdevice );
CUresult (*mocuDeviceTotalMem_v2)(size_t *,CUdevice );
CUresult (*mocuDeviceGetAttribute)(int *,CUdevice_attribute ,CUdevice );
CUresult (*mocuDeviceGetProperties)(CUdevprop *,CUdevice );
CUresult (*mocuDeviceComputeCapability)(int *,int *,CUdevice );
CUresult (*mocuCtxCreate_v2)(CUcontext *,unsigned int ,CUdevice );
CUresult (*mocuCtxDestroy_v2)(CUcontext );
CUresult (*mocuCtxPushCurrent_v2)(CUcontext );
CUresult (*mocuCtxPopCurrent_v2)(CUcontext *);
CUresult (*mocuCtxSetCurrent)(CUcontext );
CUresult (*mocuCtxGetCurrent)(CUcontext *);
CUresult (*mocuCtxGetDevice)(CUdevice *);
CUresult (*mocuCtxSynchronize)();
CUresult (*mocuCtxSetLimit)(CUlimit ,size_t );
CUresult (*mocuCtxGetLimit)(size_t *,CUlimit );
CUresult (*mocuCtxGetCacheConfig)(CUfunc_cache *);
CUresult (*mocuCtxSetCacheConfig)(CUfunc_cache);
CUresult (*mocuCtxGetSharedMemConfig)(CUsharedconfig *);
CUresult (*mocuCtxSetSharedMemConfig)(CUsharedconfig);
CUresult (*mocuCtxGetApiVersion)(CUcontext ,unsigned int *);
CUresult (*mocuCtxGetStreamPriorityRange)(int *,int *);
CUresult (*mocuCtxAttach)(CUcontext *,unsigned int);
CUresult (*mocuCtxDetach)(CUcontext);
CUresult (*mocuModuleLoad)(CUmodule *,const char *);
CUresult (*mocuModuleLoadData)(CUmodule *,const void *);
CUresult (*mocuModuleLoadDataEx)(CUmodule *,const void *,unsigned int ,CUjit_option *,void **);
CUresult (*mocuModuleLoadFatBinary)(CUmodule *,const void *);
CUresult (*mocuModuleUnload)(CUmodule);
CUresult (*mocuModuleGetFunction)(CUfunction *,CUmodule ,const char *);
CUresult (*mocuModuleGetGlobal_v2)(CUdeviceptr *,size_t *,CUmodule ,const char *);
CUresult (*mocuModuleGetTexRef)(CUtexref *,CUmodule ,const char *);
CUresult (*mocuModuleGetSurfRef)(CUsurfref *,CUmodule ,const char *);
CUresult (*mocuMemGetInfo_v2)(size_t *,size_t *);
CUresult (*mocuMemAlloc_v2)(CUdeviceptr *,size_t);
CUresult (*mocuMemAllocPitch_v2)(CUdeviceptr *,size_t *,size_t ,size_t ,unsigned int);
CUresult (*mocuMemFree_v2)(CUdeviceptr);
CUresult (*mocuMemGetAddressRange_v2)(CUdeviceptr *,size_t *,CUdeviceptr);
CUresult (*mocuMemAllocHost_v2)(void **,size_t);
CUresult (*mocuMemFreeHost)(void *);
CUresult (*mocuMemHostAlloc)(void **,size_t ,unsigned int );
CUresult (*mocuMemHostGetDevicePointer_v2)(CUdeviceptr *,void *,unsigned int);
CUresult (*mocuMemHostGetFlags)(unsigned int *,void *);
CUresult (*mocuDeviceGetByPCIBusId)(CUdevice *,char *);
CUresult (*mocuDeviceGetPCIBusId)(char *,int ,CUdevice);
CUresult (*mocuIpcGetEventHandle)(CUipcEventHandle *,CUevent);
CUresult (*mocuIpcOpenEventHandle)(CUevent *,CUipcEventHandle );
CUresult (*mocuIpcGetMemHandle)(CUipcMemHandle *,CUdeviceptr );
CUresult (*mocuIpcOpenMemHandle)(CUdeviceptr *,CUipcMemHandle ,unsigned int );
CUresult (*mocuIpcCloseMemHandle)(CUdeviceptr );
CUresult (*mocuMemHostRegister)(void *,size_t ,unsigned int );
CUresult (*mocuMemHostUnregister)(void *);
CUresult (*mocuMemcpy)(CUdeviceptr ,CUdeviceptr ,size_t );
CUresult (*mocuMemcpyPeer)(CUdeviceptr ,CUcontext ,CUdeviceptr ,CUcontext ,size_t );
CUresult (*mocuMemcpyHtoD_v2)(CUdeviceptr ,const void *,size_t );
CUresult (*mocuMemcpyDtoH_v2)(void *,CUdeviceptr ,size_t );
CUresult (*mocuMemcpyDtoD_v2)(CUdeviceptr ,CUdeviceptr ,size_t );
CUresult (*mocuMemcpyDtoA_v2)(CUarray ,size_t ,CUdeviceptr ,size_t );
CUresult (*mocuMemcpyAtoD_v2)(CUdeviceptr ,CUarray ,size_t ,size_t );
CUresult (*mocuMemcpyHtoA_v2)(CUarray ,size_t ,const void *,size_t );
CUresult (*mocuMemcpyAtoH_v2)(void *,CUarray ,size_t ,size_t );
CUresult (*mocuMemcpyAtoA_v2)(CUarray ,size_t ,CUarray ,size_t ,size_t );
CUresult (*mocuMemcpy2D_v2)(const CUDA_MEMCPY2D *);
CUresult (*mocuMemcpy2DUnaligned_v2)(const CUDA_MEMCPY2D *);
CUresult (*mocuMemcpy3D_v2)(const CUDA_MEMCPY3D *);
CUresult (*mocuMemcpy3DPeer)(const CUDA_MEMCPY3D_PEER *);
CUresult (*mocuMemcpyAsync)(CUdeviceptr ,CUdeviceptr ,size_t ,CUstream );
CUresult (*mocuMemcpyPeerAsync)(CUdeviceptr ,CUcontext ,CUdeviceptr ,CUcontext ,size_t ,CUstream );
CUresult (*mocuMemcpyHtoDAsync_v2)(CUdeviceptr ,const void *,size_t ,CUstream );
CUresult (*mocuMemcpyDtoHAsync_v2)(void *,CUdeviceptr ,size_t ,CUstream );
CUresult (*mocuMemcpyDtoDAsync_v2)(CUdeviceptr ,CUdeviceptr ,size_t ,CUstream );
CUresult (*mocuMemcpyHtoAAsync_v2)(CUarray ,size_t ,const void *,size_t ,CUstream );
CUresult (*mocuMemcpyAtoHAsync_v2)(void *,CUarray ,size_t ,size_t ,CUstream );
CUresult (*mocuMemcpy2DAsync_v2)(const CUDA_MEMCPY2D *,CUstream );
CUresult (*mocuMemcpy3DAsync_v2)(const CUDA_MEMCPY3D *,CUstream );
CUresult (*mocuMemcpy3DPeerAsync)(const CUDA_MEMCPY3D_PEER *,CUstream );
CUresult (*mocuMemsetD8_v2)(CUdeviceptr ,unsigned char ,size_t );
CUresult (*mocuMemsetD16_v2)(CUdeviceptr ,unsigned short ,size_t );
CUresult (*mocuMemsetD32_v2)(CUdeviceptr ,unsigned int ,size_t );
CUresult (*mocuMemsetD2D8_v2)(CUdeviceptr ,size_t ,unsigned char ,size_t ,size_t );
CUresult (*mocuMemsetD2D16_v2)(CUdeviceptr ,size_t ,unsigned short ,size_t ,size_t );
CUresult (*mocuMemsetD2D32_v2)(CUdeviceptr ,size_t ,unsigned int ,size_t ,size_t );
CUresult (*mocuMemsetD8Async)(CUdeviceptr ,unsigned char ,size_t ,CUstream );
CUresult (*mocuMemsetD16Async)(CUdeviceptr ,unsigned short ,size_t ,CUstream );
CUresult (*mocuMemsetD32Async)(CUdeviceptr ,unsigned int ,size_t ,CUstream );
CUresult (*mocuMemsetD2D8Async)(CUdeviceptr ,size_t ,unsigned char ,size_t ,size_t ,CUstream );
CUresult (*mocuMemsetD2D16Async)(CUdeviceptr ,size_t ,unsigned short ,size_t ,size_t ,CUstream );
CUresult (*mocuMemsetD2D32Async)(CUdeviceptr ,size_t ,unsigned int ,size_t ,size_t ,CUstream );
CUresult (*mocuArrayCreate_v2)(CUarray *,const CUDA_ARRAY_DESCRIPTOR *);
CUresult (*mocuArrayGetDescriptor_v2)(CUDA_ARRAY_DESCRIPTOR *,CUarray );
CUresult (*mocuArrayDestroy)(CUarray );
CUresult (*mocuArray3DCreate_v2)(CUarray *,const CUDA_ARRAY3D_DESCRIPTOR *);
CUresult (*mocuArray3DGetDescriptor_v2)(CUDA_ARRAY3D_DESCRIPTOR *,CUarray );
CUresult (*mocuMipmappedArrayCreate)(CUmipmappedArray *,const CUDA_ARRAY3D_DESCRIPTOR *,unsigned int );
CUresult (*mocuMipmappedArrayGetLevel)(CUarray *,CUmipmappedArray ,unsigned int );
CUresult (*mocuMipmappedArrayDestroy)(CUmipmappedArray );
CUresult (*mocuPointerGetAttribute)(void *,CUpointer_attribute ,CUdeviceptr );
CUresult (*mocuStreamCreate)(CUstream *,unsigned int );
CUresult (*mocuStreamCreateWithPriority)(CUstream *,unsigned int ,int );
CUresult (*mocuStreamGetPriority)(CUstream ,int *);
CUresult (*mocuStreamGetFlags)(CUstream ,unsigned int *);
CUresult (*mocuStreamWaitEvent)(CUstream ,CUevent ,unsigned int );
CUresult (*mocuStreamAddCallback)(CUstream ,CUstreamCallback ,void *,unsigned int );
CUresult (*mocuStreamQuery)(CUstream );
CUresult (*mocuStreamSynchronize)(CUstream );
CUresult (*mocuStreamDestroy_v2)(CUstream );
CUresult (*mocuEventCreate)(CUevent *,unsigned int );
CUresult (*mocuEventRecord)(CUevent ,CUstream );
CUresult (*mocuEventQuery)(CUevent );
CUresult (*mocuEventSynchronize)(CUevent );
CUresult (*mocuEventDestroy_v2)(CUevent );
CUresult (*mocuEventElapsedTime)(float *,CUevent ,CUevent );
CUresult (*mocuFuncGetAttribute)(int *,CUfunction_attribute ,CUfunction );
CUresult (*mocuFuncSetCacheConfig)(CUfunction ,CUfunc_cache );
CUresult (*mocuFuncSetSharedMemConfig)(CUfunction ,CUsharedconfig );
CUresult (*mocuLaunchKernel)(CUfunction ,unsigned int ,unsigned int ,unsigned int ,unsigned int ,unsigned int ,unsigned int ,unsigned int ,CUstream ,void **,void **);
CUresult (*mocuFuncSetBlockShape)(CUfunction ,int ,int ,int );
CUresult (*mocuFuncSetSharedSize)(CUfunction ,unsigned int );
CUresult (*mocuParamSetSize)(CUfunction ,unsigned int );
CUresult (*mocuParamSeti)(CUfunction ,int ,unsigned int );
CUresult (*mocuParamSetf)(CUfunction ,int ,float );
CUresult (*mocuParamSetv)(CUfunction ,int ,void *,unsigned int );
CUresult (*mocuLaunch)(CUfunction );
CUresult (*mocuLaunchGrid)(CUfunction ,int ,int );
CUresult (*mocuLaunchGridAsync)(CUfunction ,int ,int ,CUstream );
CUresult (*mocuParamSetTexRef)(CUfunction ,int ,CUtexref );
CUresult (*mocuTexRefSetArray)(CUtexref ,CUarray ,unsigned int );
CUresult (*mocuTexRefSetMipmappedArray)(CUtexref ,CUmipmappedArray ,unsigned int );
CUresult (*mocuTexRefSetAddress_v2)(size_t *,CUtexref ,CUdeviceptr ,size_t );
CUresult (*mocuTexRefSetAddress2D_v2)(CUtexref ,const CUDA_ARRAY_DESCRIPTOR *,CUdeviceptr ,size_t );
CUresult (*mocuTexRefSetFormat)(CUtexref ,CUarray_format ,int );
CUresult (*mocuTexRefSetAddressMode)(CUtexref ,int ,CUaddress_mode );
CUresult (*mocuTexRefSetFilterMode)(CUtexref ,CUfilter_mode );
CUresult (*mocuTexRefSetMipmapFilterMode)(CUtexref ,CUfilter_mode );
CUresult (*mocuTexRefSetMipmapLevelBias)(CUtexref ,float );
CUresult (*mocuTexRefSetMipmapLevelClamp)(CUtexref ,float ,float );
CUresult (*mocuTexRefSetMaxAnisotropy)(CUtexref ,unsigned int );
CUresult (*mocuTexRefSetFlags)(CUtexref ,unsigned int );
CUresult (*mocuTexRefGetAddress_v2)(CUdeviceptr *,CUtexref );
CUresult (*mocuTexRefGetArray)(CUarray *,CUtexref );
CUresult (*mocuTexRefGetMipmappedArray)(CUmipmappedArray *,CUtexref );
CUresult (*mocuTexRefGetAddressMode)(CUaddress_mode *,CUtexref ,int );
CUresult (*mocuTexRefGetFilterMode)(CUfilter_mode *,CUtexref );
CUresult (*mocuTexRefGetFormat)(CUarray_format *,int *,CUtexref );
CUresult (*mocuTexRefGetMipmapFilterMode)(CUfilter_mode *,CUtexref );
CUresult (*mocuTexRefGetMipmapLevelBias)(float *,CUtexref );
CUresult (*mocuTexRefGetMipmapLevelClamp)(float *,float *,CUtexref );
CUresult (*mocuTexRefGetMaxAnisotropy)(int *,CUtexref );
CUresult (*mocuTexRefGetFlags)(unsigned int *,CUtexref );
CUresult (*mocuTexRefCreate)(CUtexref *);
CUresult (*mocuTexRefDestroy)(CUtexref );
CUresult (*mocuSurfRefSetArray)(CUsurfref ,CUarray ,unsigned int );
CUresult (*mocuSurfRefGetArray)(CUarray *,CUsurfref );
CUresult (*mocuTexObjectCreate)(CUtexObject *,const CUDA_RESOURCE_DESC *,const CUDA_TEXTURE_DESC *,const CUDA_RESOURCE_VIEW_DESC *);
CUresult (*mocuTexObjectDestroy)(CUtexObject );
CUresult (*mocuTexObjectGetResourceDesc)(CUDA_RESOURCE_DESC *,CUtexObject );
CUresult (*mocuTexObjectGetTextureDesc)(CUDA_TEXTURE_DESC *,CUtexObject );
CUresult (*mocuTexObjectGetResourceViewDesc)(CUDA_RESOURCE_VIEW_DESC *,CUtexObject );
CUresult (*mocuSurfObjectCreate)(CUsurfObject *,const CUDA_RESOURCE_DESC *);
CUresult (*mocuSurfObjectDestroy)(CUsurfObject );
CUresult (*mocuSurfObjectGetResourceDesc)(CUDA_RESOURCE_DESC *,CUsurfObject );
CUresult (*mocuDeviceCanAccessPeer)(int *,CUdevice ,CUdevice );
CUresult (*mocuCtxEnablePeerAccess)(CUcontext ,unsigned int );
CUresult (*mocuCtxDisablePeerAccess)(CUcontext );
CUresult (*mocuGraphicsUnregisterResource)(CUgraphicsResource );
CUresult (*mocuGraphicsSubResourceGetMappedArray)(CUarray *,CUgraphicsResource ,unsigned int ,unsigned int );
CUresult (*mocuGraphicsResourceGetMappedMipmappedArray)(CUmipmappedArray *,CUgraphicsResource );
CUresult (*mocuGraphicsResourceGetMappedPointer_v2)(CUdeviceptr *,size_t *,CUgraphicsResource );
CUresult (*mocuGraphicsResourceSetMapFlags)(CUgraphicsResource ,unsigned int );
CUresult (*mocuGraphicsMapResources)(unsigned int ,CUgraphicsResource *,CUstream );
CUresult (*mocuGraphicsUnmapResources)(unsigned int ,CUgraphicsResource *,CUstream );
CUresult (*mocuGetExportTable)(const void **,const CUuuid *);
CUresult (*mocuTexRefSetAddress2D_v2)(CUtexref ,const CUDA_ARRAY_DESCRIPTOR *,CUdeviceptr ,size_t );

#if 1
CUresult (*mocuLinkCreate)(unsigned int, CUjit_option *, void **, CUlinkState *);
CUresult (*mocuLinkAddData)(CUlinkState , CUjitInputType , void *, size_t, const char *, unsigned int, CUjit_option *, void **);
CUresult (*mocuLinkAddFile)(CUlinkState , CUjitInputType , const char *, unsigned int , CUjit_option *, void **);
CUresult (*mocuLinkComplete)(CUlinkState , void **, size_t *);
CUresult (*mocuLinkDestroy)(CUlinkState );
#endif

MOCU mocu;

static int initialized = 0;

int sem_id;

void _init_smph(){
  sem_id = semget(KEY,1,IPC_CREAT|S_IRUSR|S_IWUSR|IPC_EXCL);
  if(sem_id == -1){
#if DEBUG_SEMAPHORE
    printf("Already Initialized ...\n");
#endif
    sem_id = semget(KEY,1,S_IRUSR|S_IWUSR);
  }else{
    int res = semctl(sem_id,0,SETVAL,1);
#if DEBUG_SEMAPHORE
    printf("Initialized Semaphore \n");
    printf("Result Code : %d\n",res);
#endif
  }
}


int _num_of_proc_at_device(int pos){
  int infoCount = PROC_NUM;
  nvmlReturn_t res;
  nvmlProcessInfo_t* infos;

  infos = (nvmlProcessInfo_t*)malloc(sizeof(nvmlProcessInfo_t)*infoCount);

  res = nvmlDeviceGetComputeRunningProcesses(mocu.nvml_dev[pos],&infoCount,infos);

  if(res != NVML_SUCCESS){
    printf("Failed to get compute running processes @ device%d\n",pos);
  }

  return infoCount;
}

int _actual_num_of_proc_at_device0(){
  int zero = _num_of_proc_at_device(0);
  int i;
  for(i = 1 ; i < mocu.ndev ; i ++){
    zero -= _num_of_proc_at_device(i);
  }

  if(zero < 0)zero = 0;

  return zero;
}

int _get_optimum_device_pos(){
  int pos = 0;
  int procs = _actual_num_of_proc_at_device0();
  int i;
  for(i = 1 ; i < mocu.ndev ; i ++){
    int _num = _num_of_proc_at_device(i);
    if(procs > _num){
      procs = _num;
      pos = i;
    }
  }
  return pos;
}

int _is_optimum_pos(int pos){
  int yes = 1;
  int my_num;
  int i;

  if(pos == 0)
    my_num = _actual_num_of_proc_at_device0();
  else
    my_num = _num_of_proc_at_device(pos);

  if(my_num <= 1)return yes;

  for(i = 0 ; i < mocu.ndev ; i ++){

    if(i == pos)continue;
    else{
      int num;
      if(i == 0){
	num = _actual_num_of_proc_at_device0();
      }else{
	num = _num_of_proc_at_device(i);
      }
      if(num < my_num)yes = 0;
    }
  }

  return yes;
}

void _detach_smph(){
  int proc_num = _num_of_proc_at_device(0);

  if(proc_num == 1){
#if DEBUG_SEMAPHORE
    printf("This Process is the last one.\n");
    printf("So detach the semaphore...\n");
#endif
    semctl(sem_id,0,IPC_RMID);
  }
}


void lock_other_proc(){
  struct sembuf sops;

  sops.sem_num = 0;
  sops.sem_op = -1;
  sops.sem_flg = SEM_UNDO;

  int res = semop(sem_id , &sops , 1);
#if DEBUG_SEMAPHORE  
  printf("Lock Other Proc : %d\n",res);
#endif
}

void unlock_other_proc(){
  struct sembuf sops;

  sops.sem_num = 0;
  sops.sem_op = 1;
  sops.sem_flg = SEM_UNDO;

  int res = semop(sem_id , &sops , 1);
#if DEBUG_SEMAPHORE
  printf("Unlock Other Proc : %d\n",res);
#endif
}

void _init_apilog(context* cp){
  cp->alog = (apilog*)malloc(sizeof(apilog));
  cp->alast = cp->alog;
  cp->alog->prev = NULL;
  cp->alog->next = NULL;
  cp->alog->type = -1;
}

thread* create_new_thread(pthread_t new_tid){

  thread* tp;
  
  tp = (thread*)malloc(sizeof(thread));
  tp->mode = 2;
  tp->prev = mocu.t1->prev;
  tp->next = mocu.t1;
  tp->prev->next = tp;
  tp->next->prev = tp;
  tp->tid = new_tid;

  tp->ctx = (context*)malloc(sizeof(context));
  tp->ctx->stack = NULL;
  tp->ctx->mode = -1;

  return tp;
}

thread* get_now_thread(pthread_t ntid){

  thread* tp;
  pthread_t tid;

  tid = ntid;
  tp = mocu.t0->next;

  while(1){

    if(pthread_equal(tp->tid,tid)){
      break;
    }

    if(tp->next == NULL){
      tp = create_new_thread(pthread_self());
      break;
    }else{
      tp = tp->next;
    }
  }
  return tp;
}

context* get_current_context(){
  thread* tp;
  context* cp;
  pthread_t tid;

  tid = pthread_self();
  tp = get_now_thread(tid);
  
  cp = tp->ctx;
  while(cp->stack != NULL){
    cp = cp->stack;
  }

  return cp;
}

void _init_context(){    
  mocu.c0 = (context *)malloc(sizeof(context));
  mocu.c1 = (context *)malloc(sizeof(context));
  mocu.c0->prev = NULL;
  mocu.c1->next = NULL;
  mocu.c0->next = mocu.c1;
  mocu.c1->prev = mocu.c0;
  mocu.c0->mode = -1;
  mocu.c1->mode = -1;
}

void _init_region(context* cp){
  cp->d0 = (region*)malloc(sizeof(region));
  cp->d1 = (region*)malloc(sizeof(region));
  cp->d0->mode = cp->d1->mode = -1;
  cp->d0->next = cp->d1;
  cp->d1->prev = cp->d0;
  cp->d1->next = NULL;
  cp->d0->prev = NULL;
}

void _init_stream(context* cp){
  cp->s0 = (stream*)malloc(sizeof(stream));
  cp->s1 = (stream*)malloc(sizeof(stream));
  cp->s0->mode = cp->s1->mode = -1;
  cp->s0->next = cp->s1;
  cp->s1->prev = cp->s0;
  cp->s1->next = NULL;
  cp->s0->prev = NULL;
}

void _init_texref(context* cp){
  cp->t0 = (texref*)malloc(sizeof(texref));
  cp->t1 = (texref*)malloc(sizeof(texref));
  cp->t0->mode = cp->t1->mode = -1;
  cp->t0->next = cp->t1;
  cp->t1->prev = cp->t0;
  cp->t1->next = NULL;
  cp->t0->prev = NULL;
}


void _init_module(context* cp){
  cp->m0 = (module*)malloc(sizeof(module));
  cp->m1 = (module*)malloc(sizeof(module));
  cp->m0->mode = cp->m1->mode = -1;
  cp->m0->next = cp->m1;
  cp->m1->prev = cp->m0;
  cp->m1->next = NULL;
  cp->m0->prev = NULL;  
}

void _init_thread(){
  mocu.t0 = (thread *)malloc(sizeof(thread));
  mocu.t1 = (thread *)malloc(sizeof(thread));
  mocu.t0->prev = NULL;
  mocu.t1->next = NULL;
  mocu.t0->next = mocu.t1;
  mocu.t1->prev = mocu.t0;
  mocu.t0->mode = -1;
  mocu.t1->mode = -1;
  mocu.t0->tid = -1;
  mocu.t1->tid = -1;

  create_new_thread(pthread_self());
}

void stack_context(context* cp){
  pthread_t tid;
  thread* tp;
  context* scp;
  
  tid = pthread_self();
  tp = get_now_thread(tid);

  scp = tp->ctx;
  while(scp->stack != NULL){
    scp = scp->stack;
  }

  scp->stack = cp;
  cp->stack = NULL;
}

void _init_event(context* cp){
  cp->e0 = (event*)malloc(sizeof(event));
  cp->e1 = (event*)malloc(sizeof(event));
  cp->e0->mode = cp->e1->mode = -1;
  cp->e0->next = cp->e1;
  cp->e1->prev = cp->e0;
  cp->e1->next = NULL;
  cp->e0->prev = NULL;
  mocuEventCreate(&cp->e1->e,0);
}

int create_context(CUcontext ctx){

  context *cp,*temp;
  int found = 0;

  temp = mocu.c0->next;

  while(temp->next != NULL){
    if(temp->user == ctx){
      found = 1;
      break;
    }else{
      temp = temp->next;
    }
  }

  if(!found){

    cp = (context*)malloc(sizeof(context));
    cp->mode = 1;
    cp->user = ctx;

    unsigned int flags = 0;

    CUresult res;
    
    res = mocuCtxCreate_v2(&cp->ctx,flags,mocu.dev[mocuID]);

    if(res != CUDA_SUCCESS){
      printf("Failed to call mocuCtxCreate_v2(CUcontext,flag,CUdevice) with %d\n",res);
    }

    cp->prev = mocu.c1->prev;
    cp->next = mocu.c1;
    cp->prev->next = cp;
    cp->next->prev = cp;
    cp->flags = flags;
    cp->tid = pthread_self();

    _init_module(cp);
    _init_apilog(cp);
    _init_stream(cp);
    _init_event(cp);
    _init_region(cp);
    _init_texref(cp);
    stack_context(cp);

    return 1;
  }else{
    cp = temp; 
    stack_context(cp);

    return 0;
  }
}

void _init_env(){
  int i;
  mocuDeviceGetCount(&(mocu.ndev));
  mocu.devid = (int*)malloc(sizeof(int)*(mocu.ndev));
  for(i = 0 ; i < mocu.ndev; i ++){
    mocu.devid[i] = i;
  }
  mocu.dev = (CUdevice*)malloc(sizeof(CUdevice)*(mocu.ndev));
}


void _init_mocu(){
  CUresult res;
  
  mocu.version[0] = 0;
  res = mocuInit(0);

  if(res != CUDA_SUCCESS){
    printf("Failed to initialize CUDA.\n");
    exit(0);
  }

  _init_env();

  nvmlReturn_t _res;

  _res = nvmlInit();

  if(_res != NVML_SUCCESS){
    printf("Failed to initialize nvml\n");
  }

  mocu.nvml_dev = (nvmlDevice_t*)malloc(sizeof(nvmlDevice_t)*mocu.ndev);

  int i;
  for(i = 0; i < mocu.ndev; i ++){
    res = mocuDeviceGet(&mocu.dev[i],i);
    
    if(res != CUDA_SUCCESS){
      printf("Failed to get device.%d\n",i);
      exit(0);
    }

    _res = nvmlDeviceGetHandleByIndex(i,&mocu.nvml_dev[i]);

    if(_res != NVML_SUCCESS){
      printf("Failed to get device%d by using nvml\n",i);
    }
  }
  
  _init_context();
  _init_thread();
}


static int lock = 0;

void print_all_log(){

#if PRINT_LOG
  
  while(lock);

  lock = 1;

  printf("\n>Print All Logs\n");

  thread* tp;
  context* cp;
  int tc = 0,cc = 0;
  
  tp = mocu.t0->next;
  while(tp->next != NULL){
    tc ++;
    printf("--Thread(%3d) : %p  >tid %u\n",tc,tp,tp->tid);
    cp = tp->ctx->stack;
    while(cp != NULL){
      cc ++;
      printf("---Stacked context(%3d) : %p\n",cc,cp);
      cp = cp->stack;
    }

    tp = tp->next;
  }

  context* _c;
  apilog* a;
  region* d;
  event* e;
  stream* s;
  module* m;
  int _cc = 0,ac = 0,dc = 0,ec = 0,sc = 0,mc = 0;

  _c = mocu.c0->next;
  while(_c->next != NULL){
    _cc++;
    printf("--Context(%3d) : %p\n",_cc,_c);
    printf("----- user : %p\n",_c->user);
    printf("----- ctx  : %p\n",_c->ctx);

    a = _c->alog;
    while(a->next != NULL){
      ac++;
      printf("---Apilog(%3d) : %p\n",ac,a);
      a = a->next;
    }

    d = _c->d0->next;
    while(d->next != NULL){
      dc++;
      printf("---Region(%3d) : %p   >dptr : %p\n",dc,d,d->base);
      d = d->next;
    }

    s = _c->s0->next;
    while(s->next != NULL){
      sc++;
      printf("---Stream(%3d) : %p\n",sc,s);
      s = s->next;
    }

    e = _c->e0->next;
    while(e->next != NULL){
      ec++;
      printf("---Event (%3d) : %p\n",ec,e);
      e = e->next;
    }

    m = _c->m0->next;
    while(m->next != NULL){
      mc++;
      printf("---Module(%3d) : %p\n",mc,m);
      m = m->next;
    }

    _c = _c->next;
  }

  printf("\n");

  lock = 0;

#endif
}

static int module_init(module* mp){

  mp->f0 = (function *)malloc(sizeof(function));
  mp->f1 = (function *)malloc(sizeof(function));
  mp->f0->mode = mp->f1->mode = -1;
  mp->f0->prev = mp->f1->next = NULL;
  mp->f0->next = mp->f1;
  mp->f1->prev = mp->f0;

  mp->s0 = (symbol *)malloc(sizeof(symbol));
  mp->s1 = (symbol *)malloc(sizeof(symbol));
  mp->s0->mode = mp->s1->mode = -1;
  mp->s0->prev = mp->s1->next = NULL;
  mp->s0->next = mp->s1;
  mp->s1->prev = mp->s0;

  return 0;
}

static int module_collect_symbols(module *mp)
{
  symbol *s;

  s = mp->s0->next;
  while (s->mode >= 0) {
    if (s->type == SYMBOL_DEVICE || s->type == SYMBOL_CONST) {
      mocuModuleGetGlobal(&s->addr, &s->size, mp->m, s->name);
    }
    s = s->next;
  }

  return 0;
}

module* get_module_correspond_to_hmod(context* cp,CUmodule hmod,int _flag){
  module* mp;
  
  mp = cp->m0->next;
  while(mp->next != NULL){
    if(mp->m == hmod || mp == (module*)hmod){
      return mp;
    }
    mp = mp->next;
  }

  if(_flag){
    mp = (module*)malloc(sizeof(module));
    mp->m = hmod;
    mp->mode = 8;
    mp->prev = cp->m1->prev;
    mp->next = cp->m1;
    mp->prev->next = mp;
    mp->next->prev = mp;
    module_init(mp);

    return mp;
  }else{
    return NULL;
  }
}

size_t check_memory_amount_used(){

#if DEBUG_MIG
  printf("\n");
  printf("+--------------------------+\n");
  printf("|Amount Used of Memory     |\n");
  printf("+==========================+\n");;
#endif

  context* cp;
  region* r;
  size_t _size = 0;
  int counter = 0;

  cp = mocu.c0->next;

  while(cp->next != NULL){

    r = cp->d0->next;

    while(r->next != NULL){
      
#if DEBUG_MIG
      printf("|Region %3d  %10ld[KB]|\n",++counter,r->size >> 10);
      printf("+--------------------------+\n");
#endif
      
      _size += r->size;

      r = r->next;
    }
    
    cp = cp->next;
  }

#if DEBUG_MIG
  printf("|Total :     %10ld[KB]|\n",_size >> 10);
  printf("+--------------------------+\n");
#endif

  return _size;
}

__attribute__((constructor())) void __init_taichirou(){

#if DEBUG
  printf("[MOCU] constructor is called.\n");
#endif

  if(initialized) return;

  void* handle = dlopen("/usr/lib64/libcuda.so",RTLD_LAZY|RTLD_LOCAL);

  mocuInit = (CUresult (*)(unsigned int ))dlsym(handle,"cuInit");
  mocuDriverGetVersion = (CUresult (*)(int *))dlsym(handle,"cuDriverGetVersion");
  mocuDeviceGet = (CUresult (*)(CUdevice *,int ))dlsym(handle,"cuDeviceGet");
  mocuDeviceGetCount = (CUresult (*)(int *))dlsym(handle,"cuDeviceGetCount");
  mocuDeviceGetName = (CUresult (*)(char *,int ,CUdevice ))dlsym(handle,"cuDeviceGetName");
  mocuDeviceTotalMem_v2 = (CUresult (*)(size_t *,CUdevice ))dlsym(handle,"cuDeviceTotalMem_v2");
  mocuDeviceGetAttribute = (CUresult (*)(int *,CUdevice_attribute ,CUdevice ))dlsym(handle,"cuDeviceGetAttribute");
  mocuDeviceGetProperties = (CUresult (*)(CUdevprop *,CUdevice ))dlsym(handle,"cuDeviceGetProperties");
  mocuDeviceComputeCapability = (CUresult (*)(int *,int *,CUdevice ))dlsym(handle,"cuDeviceComputeCapability");
  mocuCtxCreate_v2 = (CUresult (*)(CUcontext *,unsigned int ,CUdevice ))dlsym(handle,"cuCtxCreate_v2");
  mocuCtxDestroy_v2 = (CUresult (*)(CUcontext ))dlsym(handle,"cuCtxDestroy_v2");
  mocuCtxPushCurrent_v2 = (CUresult (*)(CUcontext ))dlsym(handle,"cuCtxPushCurrent_v2");
  mocuCtxPopCurrent_v2 = (CUresult (*)(CUcontext *))dlsym(handle,"cuCtxPopCurrent_v2");
  mocuCtxSetCurrent = (CUresult (*)(CUcontext ))dlsym(handle,"cuCtxSetCurrent");
  mocuCtxGetCurrent = (CUresult (*)(CUcontext *))dlsym(handle,"cuCtxGetCurrent");
  mocuCtxGetDevice = (CUresult (*)(CUdevice *))dlsym(handle,"cuCtxGetDevice");
  mocuCtxSynchronize = (CUresult (*)())dlsym(handle,"cuCtxSynchronize");
  mocuCtxSetLimit = (CUresult (*)(CUlimit ,size_t ))dlsym(handle,"cuCtxSetLimit");
  mocuCtxGetLimit = (CUresult (*)(size_t *,CUlimit ))dlsym(handle,"cuCtxGetLimit");
  mocuCtxGetCacheConfig = (CUresult (*)(CUfunc_cache *))dlsym(handle,"cuCtxGetCacheConfig");
  mocuCtxSetCacheConfig = (CUresult (*)(CUfunc_cache ))dlsym(handle,"cuCtxSetCacheConfig");
  mocuCtxGetSharedMemConfig = (CUresult (*)(CUsharedconfig *))dlsym(handle,"cuCtxGetSharedMemConfig");
  mocuCtxSetSharedMemConfig = (CUresult (*)(CUsharedconfig ))dlsym(handle,"cuCtxSetSharedMemConfig");
  mocuCtxGetApiVersion = (CUresult (*)(CUcontext ,unsigned int *))dlsym(handle,"cuCtxGetApiVersion");
  mocuCtxGetStreamPriorityRange = (CUresult (*)(int *,int *))dlsym(handle,"cuCtxGetStreamPriorityRange");
  mocuCtxAttach = (CUresult (*)(CUcontext *,unsigned int ))dlsym(handle,"cuCtxAttach");
  mocuCtxDetach = (CUresult (*)(CUcontext ))dlsym(handle,"cuCtxDetach");
  mocuModuleLoad = (CUresult (*)(CUmodule *,const char *))dlsym(handle,"cuModuleLoad");
  mocuModuleLoadData = (CUresult (*)(CUmodule *,const void *))dlsym(handle,"cuModuleLoadData");
  mocuModuleLoadDataEx = (CUresult (*)(CUmodule *,const void *,unsigned int ,CUjit_option *,void **))dlsym(handle,"cuModuleLoadDataEx");
  mocuModuleLoadFatBinary = (CUresult (*)(CUmodule *,const void *))dlsym(handle,"cuModuleLoadFatBinary");
  mocuModuleUnload = (CUresult (*)(CUmodule ))dlsym(handle,"cuModuleUnload");
  mocuModuleGetFunction = (CUresult (*)(CUfunction *,CUmodule ,const char *))dlsym(handle,"cuModuleGetFunction");
  mocuModuleGetGlobal_v2 = (CUresult (*)(CUdeviceptr *,size_t *,CUmodule ,const char *))dlsym(handle,"cuModuleGetGlobal_v2");
  mocuModuleGetTexRef = (CUresult (*)(CUtexref *,CUmodule ,const char *))dlsym(handle,"cuModuleGetTexRef");
  mocuModuleGetSurfRef = (CUresult (*)(CUsurfref *,CUmodule ,const char *))dlsym(handle,"cuModuleGetSurfRef");
  mocuMemGetInfo_v2 = (CUresult (*)(size_t *,size_t *))dlsym(handle,"cuMemGetInfo_v2");
  mocuMemAlloc_v2 = (CUresult (*)(CUdeviceptr *,size_t ))dlsym(handle,"cuMemAlloc_v2");
  mocuMemAllocPitch_v2 = (CUresult (*)(CUdeviceptr *,size_t *,size_t ,size_t ,unsigned int ))dlsym(handle,"cuMemAllocPitch_v2");
  mocuMemFree_v2 = (CUresult (*)(CUdeviceptr ))dlsym(handle,"cuMemFree_v2");
  mocuMemGetAddressRange_v2 = (CUresult (*)(CUdeviceptr *,size_t *,CUdeviceptr ))dlsym(handle,"cuMemGetAddressRange_v2");
  mocuMemAllocHost_v2 = (CUresult (*)(void **,size_t ))dlsym(handle,"cuMemAllocHost_v2");
  mocuMemFreeHost = (CUresult (*)(void *))dlsym(handle,"cuMemFreeHost");
  mocuMemHostAlloc = (CUresult (*)(void **,size_t ,unsigned int ))dlsym(handle,"cuMemHostAlloc");
  mocuMemHostGetDevicePointer_v2 = (CUresult (*)(CUdeviceptr *,void *,unsigned int ))dlsym(handle,"cuMemHostGetDevicePointer_v2");
  mocuMemHostGetFlags = (CUresult (*)(unsigned int *,void *))dlsym(handle,"cuMemHostGetFlags");
  mocuDeviceGetByPCIBusId = (CUresult (*)(CUdevice *,char *))dlsym(handle,"cuDeviceGetByPCIBusId");
  mocuDeviceGetPCIBusId = (CUresult (*)(char *,int ,CUdevice ))dlsym(handle,"cuDeviceGetPCIBusId");
  mocuIpcGetEventHandle = (CUresult (*)(CUipcEventHandle *,CUevent ))dlsym(handle,"cuIpcGetEventHandle");
  mocuIpcOpenEventHandle = (CUresult (*)(CUevent *,CUipcEventHandle ))dlsym(handle,"cuIpcOpenEventHandle");
  mocuIpcGetMemHandle = (CUresult (*)(CUipcMemHandle *,CUdeviceptr ))dlsym(handle,"cuIpcGetMemHandle");
  mocuIpcOpenMemHandle = (CUresult (*)(CUdeviceptr *,CUipcMemHandle ,unsigned int ))dlsym(handle,"cuIpcOpenMemHandle");
  mocuIpcCloseMemHandle = (CUresult (*)(CUdeviceptr ))dlsym(handle,"cuIpcCloseMemHandle");
  mocuMemHostRegister = (CUresult (*)(void *,size_t ,unsigned int ))dlsym(handle,"cuMemHostRegister");
  mocuMemHostUnregister = (CUresult (*)(void *))dlsym(handle,"cuMemHostUnregister");
  mocuMemcpy = (CUresult (*)(CUdeviceptr ,CUdeviceptr ,size_t ))dlsym(handle,"cuMemcpy");
  mocuMemcpyPeer = (CUresult (*)(CUdeviceptr ,CUcontext ,CUdeviceptr ,CUcontext ,size_t ))dlsym(handle,"cuMemcpyPeer");
  mocuMemcpyHtoD_v2 = (CUresult (*)(CUdeviceptr ,const void *,size_t ))dlsym(handle,"cuMemcpyHtoD_v2");
  mocuMemcpyDtoH_v2 = (CUresult (*)(void *,CUdeviceptr ,size_t ))dlsym(handle,"cuMemcpyDtoH_v2");
  mocuMemcpyDtoD_v2 = (CUresult (*)(CUdeviceptr ,CUdeviceptr ,size_t ))dlsym(handle,"cuMemcpyDtoD_v2");
  mocuMemcpyDtoA_v2 = (CUresult (*)(CUarray ,size_t ,CUdeviceptr ,size_t ))dlsym(handle,"cuMemcpyDtoA_v2");
  mocuMemcpyAtoD_v2 = (CUresult (*)(CUdeviceptr ,CUarray ,size_t ,size_t ))dlsym(handle,"cuMemcpyAtoD_v2");
  mocuMemcpyHtoA_v2 = (CUresult (*)(CUarray ,size_t ,const void *,size_t ))dlsym(handle,"cuMemcpyHtoA_v2");
  mocuMemcpyAtoH_v2 = (CUresult (*)(void *,CUarray ,size_t ,size_t ))dlsym(handle,"cuMemcpyAtoH_v2");
  mocuMemcpyAtoA_v2 = (CUresult (*)(CUarray ,size_t ,CUarray ,size_t ,size_t ))dlsym(handle,"cuMemcpyAtoA_v2");
  mocuMemcpy2D_v2 = (CUresult (*)(const CUDA_MEMCPY2D *))dlsym(handle,"cuMemcpy2D_v2");
  mocuMemcpy2DUnaligned_v2 = (CUresult (*)(const CUDA_MEMCPY2D *))dlsym(handle,"cuMemcpy2DUnaligned_v2");
  mocuMemcpy3D_v2 = (CUresult (*)(const CUDA_MEMCPY3D *))dlsym(handle,"cuMemcpy3D_v2");
  mocuMemcpy3DPeer = (CUresult (*)(const CUDA_MEMCPY3D_PEER *))dlsym(handle,"cuMemcpy3DPeer");
  mocuMemcpyAsync = (CUresult (*)(CUdeviceptr ,CUdeviceptr ,size_t ,CUstream ))dlsym(handle,"cuMemcpyAsync");
  mocuMemcpyPeerAsync = (CUresult (*)(CUdeviceptr ,CUcontext ,CUdeviceptr ,CUcontext ,size_t ,CUstream ))dlsym(handle,"cuMemcpyPeerAsync");
  mocuMemcpyHtoDAsync_v2 = (CUresult (*)(CUdeviceptr ,const void *,size_t ,CUstream ))dlsym(handle,"cuMemcpyHtoDAsync_v2");
  mocuMemcpyDtoHAsync_v2 = (CUresult (*)(void *,CUdeviceptr ,size_t ,CUstream ))dlsym(handle,"cuMemcpyDtoHAsync_v2");
  mocuMemcpyDtoDAsync_v2 = (CUresult (*)(CUdeviceptr ,CUdeviceptr ,size_t ,CUstream ))dlsym(handle,"cuMemcpyDtoDAsync_v2");
  mocuMemcpyHtoAAsync_v2 = (CUresult (*)(CUarray ,size_t ,const void *,size_t ,CUstream ))dlsym(handle,"cuMemcpyHtoAAsync_v2");
  mocuMemcpyAtoHAsync_v2 = (CUresult (*)(void *,CUarray ,size_t ,size_t ,CUstream ))dlsym(handle,"cuMemcpyAtoHAsync_v2");
  mocuMemcpy2DAsync_v2 = (CUresult (*)(const CUDA_MEMCPY2D *,CUstream ))dlsym(handle,"cuMemcpy2DAsync_v2");
  mocuMemcpy3DAsync_v2 = (CUresult (*)(const CUDA_MEMCPY3D *,CUstream ))dlsym(handle,"cuMemcpy3DAsync_v2");
  mocuMemcpy3DPeerAsync = (CUresult (*)(const CUDA_MEMCPY3D_PEER *,CUstream ))dlsym(handle,"cuMemcpy3DPeerAsync");
  mocuMemsetD8_v2 = (CUresult (*)(CUdeviceptr ,unsigned char ,size_t ))dlsym(handle,"cuMemsetD8_v2");
  mocuMemsetD16_v2 = (CUresult (*)(CUdeviceptr ,unsigned short ,size_t ))dlsym(handle,"cuMemsetD16_v2");
  mocuMemsetD32_v2 = (CUresult (*)(CUdeviceptr ,unsigned int ,size_t ))dlsym(handle,"cuMemsetD32_v2");
  mocuMemsetD2D8_v2 = (CUresult (*)(CUdeviceptr ,size_t ,unsigned char ,size_t ,size_t ))dlsym(handle,"cuMemsetD2D8_v2");
  mocuMemsetD2D16_v2 = (CUresult (*)(CUdeviceptr ,size_t ,unsigned short ,size_t ,size_t ))dlsym(handle,"cuMemsetD2D16_v2");
  mocuMemsetD2D32_v2 = (CUresult (*)(CUdeviceptr ,size_t ,unsigned int ,size_t ,size_t ))dlsym(handle,"cuMemsetD2D32_v2");
  mocuMemsetD8Async = (CUresult (*)(CUdeviceptr ,unsigned char ,size_t ,CUstream ))dlsym(handle,"cuMemsetD8Async");
  mocuMemsetD16Async = (CUresult (*)(CUdeviceptr ,unsigned short ,size_t ,CUstream ))dlsym(handle,"cuMemsetD16Async");
  mocuMemsetD32Async = (CUresult (*)(CUdeviceptr ,unsigned int ,size_t ,CUstream ))dlsym(handle,"cuMemsetD32Async");
  mocuMemsetD2D8Async = (CUresult (*)(CUdeviceptr ,size_t ,unsigned char ,size_t ,size_t ,CUstream ))dlsym(handle,"cuMemsetD2D8Async");
  mocuMemsetD2D16Async = (CUresult (*)(CUdeviceptr ,size_t ,unsigned short ,size_t ,size_t ,CUstream ))dlsym(handle,"cuMemsetD2D16Async");
  mocuMemsetD2D32Async = (CUresult (*)(CUdeviceptr ,size_t ,unsigned int ,size_t ,size_t ,CUstream ))dlsym(handle,"cuMemsetD2D32Async");
  mocuArrayCreate_v2 = (CUresult (*)(CUarray *,const CUDA_ARRAY_DESCRIPTOR *))dlsym(handle,"cuArrayCreate_v2");
  mocuArrayGetDescriptor_v2 = (CUresult (*)(CUDA_ARRAY_DESCRIPTOR *,CUarray ))dlsym(handle,"cuArrayGetDescriptor_v2");
  mocuArrayDestroy = (CUresult (*)(CUarray ))dlsym(handle,"cuArrayDestroy");
  mocuArray3DCreate_v2 = (CUresult (*)(CUarray *,const CUDA_ARRAY3D_DESCRIPTOR *))dlsym(handle,"cuArray3DCreate_v2");
  mocuArray3DGetDescriptor_v2 = (CUresult (*)(CUDA_ARRAY3D_DESCRIPTOR *,CUarray ))dlsym(handle,"cuArray3DGetDescriptor_v2");
  mocuMipmappedArrayCreate = (CUresult (*)(CUmipmappedArray *,const CUDA_ARRAY3D_DESCRIPTOR *,unsigned int ))dlsym(handle,"cuMipmappedArrayCreate");
  mocuMipmappedArrayGetLevel = (CUresult (*)(CUarray *,CUmipmappedArray ,unsigned int ))dlsym(handle,"cuMipmappedArrayGetLevel");
  mocuMipmappedArrayDestroy = (CUresult (*)(CUmipmappedArray ))dlsym(handle,"cuMipmappedArrayDestroy");
  mocuPointerGetAttribute = (CUresult (*)(void *,CUpointer_attribute ,CUdeviceptr ))dlsym(handle,"cuPointerGetAttribute");
  mocuStreamCreate = (CUresult (*)(CUstream *,unsigned int ))dlsym(handle,"cuStreamCreate");
  mocuStreamCreateWithPriority = (CUresult (*)(CUstream *,unsigned int ,int ))dlsym(handle,"cuStreamCreateWithPriority");
  mocuStreamGetPriority = (CUresult (*)(CUstream ,int *))dlsym(handle,"cuStreamGetPriority");
  mocuStreamGetFlags = (CUresult (*)(CUstream ,unsigned int *))dlsym(handle,"cuStreamGetFlags");
  mocuStreamWaitEvent = (CUresult (*)(CUstream ,CUevent ,unsigned int ))dlsym(handle,"cuStreamWaitEvent");
  mocuStreamAddCallback = (CUresult (*)(CUstream ,CUstreamCallback ,void *,unsigned int ))dlsym(handle,"cuStreamAddCallback");
  mocuStreamQuery = (CUresult (*)(CUstream ))dlsym(handle,"cuStreamQuery");
  mocuStreamSynchronize = (CUresult (*)(CUstream ))dlsym(handle,"cuStreamSynchronize");
  mocuStreamDestroy_v2 = (CUresult (*)(CUstream ))dlsym(handle,"cuStreamDestroy_v2");
  mocuEventCreate = (CUresult (*)(CUevent *,unsigned int ))dlsym(handle,"cuEventCreate");
  mocuEventRecord = (CUresult (*)(CUevent ,CUstream ))dlsym(handle,"cuEventRecord");
  mocuEventQuery = (CUresult (*)(CUevent ))dlsym(handle,"cuEventQuery");
  mocuEventSynchronize = (CUresult (*)(CUevent ))dlsym(handle,"cuEventSynchronize");
  mocuEventDestroy_v2 = (CUresult (*)(CUevent ))dlsym(handle,"cuEventDestroy_v2");
  mocuEventElapsedTime = (CUresult (*)(float *,CUevent ,CUevent ))dlsym(handle,"cuEventElapsedTime");
  mocuFuncGetAttribute = (CUresult (*)(int *,CUfunction_attribute ,CUfunction ))dlsym(handle,"cuFuncGetAttribute");
  mocuFuncSetCacheConfig = (CUresult (*)(CUfunction ,CUfunc_cache ))dlsym(handle,"cuFuncSetCacheConfig");
  mocuFuncSetSharedMemConfig = (CUresult (*)(CUfunction ,CUsharedconfig ))dlsym(handle,"cuFuncSetSharedMemConfig");
  mocuLaunchKernel = (CUresult (*)(CUfunction ,unsigned int ,unsigned int ,unsigned int ,unsigned int ,unsigned int ,unsigned int ,unsigned int ,CUstream ,void **,void **))dlsym(handle,"cuLaunchKernel");
  mocuFuncSetBlockShape = (CUresult (*)(CUfunction ,int ,int ,int ))dlsym(handle,"cuFuncSetBlockShape");
  mocuFuncSetSharedSize = (CUresult (*)(CUfunction ,unsigned int ))dlsym(handle,"cuFuncSetSharedSize");
  mocuParamSetSize = (CUresult (*)(CUfunction ,unsigned int ))dlsym(handle,"cuParamSetSize");
  mocuParamSeti = (CUresult (*)(CUfunction ,int ,unsigned int ))dlsym(handle,"cuParamSeti");
  mocuParamSetf = (CUresult (*)(CUfunction ,int ,float ))dlsym(handle,"cuParamSetf");
  mocuParamSetv = (CUresult (*)(CUfunction ,int ,void *,unsigned int ))dlsym(handle,"cuParamSetv");
  mocuLaunch = (CUresult (*)(CUfunction ))dlsym(handle,"cuLaunch");
  mocuLaunchGrid = (CUresult (*)(CUfunction ,int ,int ))dlsym(handle,"cuLaunchGrid");
  mocuLaunchGridAsync = (CUresult (*)(CUfunction ,int ,int ,CUstream ))dlsym(handle,"cuLaunchGridAsync");
  mocuParamSetTexRef = (CUresult (*)(CUfunction ,int ,CUtexref ))dlsym(handle,"cuParamSetTexRef");
  mocuTexRefSetArray = (CUresult (*)(CUtexref ,CUarray ,unsigned int ))dlsym(handle,"cuTexRefSetArray");
  mocuTexRefSetMipmappedArray = (CUresult (*)(CUtexref ,CUmipmappedArray ,unsigned int ))dlsym(handle,"cuTexRefSetMipmappedArray");
  mocuTexRefSetAddress_v2 = (CUresult (*)(size_t *,CUtexref ,CUdeviceptr ,size_t ))dlsym(handle,"cuTexRefSetAddress_v2");
  mocuTexRefSetAddress2D_v2 = (CUresult (*)(CUtexref ,const CUDA_ARRAY_DESCRIPTOR *,CUdeviceptr ,size_t ))dlsym(handle,"cuTexRefSetAddress2D_v2");
  mocuTexRefSetFormat = (CUresult (*)(CUtexref ,CUarray_format ,int ))dlsym(handle,"cuTexRefSetFormat");
  mocuTexRefSetAddressMode = (CUresult (*)(CUtexref ,int ,CUaddress_mode ))dlsym(handle,"cuTexRefSetAddressMode");
  mocuTexRefSetFilterMode = (CUresult (*)(CUtexref ,CUfilter_mode ))dlsym(handle,"cuTexRefSetFilterMode");
  mocuTexRefSetMipmapFilterMode = (CUresult (*)(CUtexref ,CUfilter_mode ))dlsym(handle,"cuTexRefSetMipmapFilterMode");
  mocuTexRefSetMipmapLevelBias = (CUresult (*)(CUtexref ,float ))dlsym(handle,"cuTexRefSetMipmapLevelBias");
  mocuTexRefSetMipmapLevelClamp = (CUresult (*)(CUtexref ,float ,float ))dlsym(handle,"cuTexRefSetMipmapLevelClamp");
  mocuTexRefSetMaxAnisotropy = (CUresult (*)(CUtexref ,unsigned int ))dlsym(handle,"cuTexRefSetMaxAnisotropy");
  mocuTexRefSetFlags = (CUresult (*)(CUtexref ,unsigned int ))dlsym(handle,"cuTexRefSetFlags");
  mocuTexRefGetAddress_v2 = (CUresult (*)(CUdeviceptr *,CUtexref ))dlsym(handle,"cuTexRefGetAddress_v2");
  mocuTexRefGetArray = (CUresult (*)(CUarray *,CUtexref ))dlsym(handle,"cuTexRefGetArray");
  mocuTexRefGetMipmappedArray = (CUresult (*)(CUmipmappedArray *,CUtexref ))dlsym(handle,"cuTexRefGetMipmappedArray");
  mocuTexRefGetAddressMode = (CUresult (*)(CUaddress_mode *,CUtexref ,int ))dlsym(handle,"cuTexRefGetAddressMode");
  mocuTexRefGetFilterMode = (CUresult (*)(CUfilter_mode *,CUtexref ))dlsym(handle,"cuTexRefGetFilterMode");
  mocuTexRefGetFormat = (CUresult (*)(CUarray_format *,int *,CUtexref ))dlsym(handle,"cuTexRefGetFormat");
  mocuTexRefGetMipmapFilterMode = (CUresult (*)(CUfilter_mode *,CUtexref ))dlsym(handle,"cuTexRefGetMipmapFilterMode");
  mocuTexRefGetMipmapLevelBias = (CUresult (*)(float *,CUtexref ))dlsym(handle,"cuTexRefGetMipmapLevelBias");
  mocuTexRefGetMipmapLevelClamp = (CUresult (*)(float *,float *,CUtexref ))dlsym(handle,"cuTexRefGetMipmapLevelClamp");
  mocuTexRefGetMaxAnisotropy = (CUresult (*)(int *,CUtexref ))dlsym(handle,"cuTexRefGetMaxAnisotropy");
  mocuTexRefGetFlags = (CUresult (*)(unsigned int *,CUtexref ))dlsym(handle,"cuTexRefGetFlags");
  mocuTexRefCreate = (CUresult (*)(CUtexref *))dlsym(handle,"cuTexRefCreate");
  mocuTexRefDestroy = (CUresult (*)(CUtexref ))dlsym(handle,"cuTexRefDestroy");
  mocuSurfRefSetArray = (CUresult (*)(CUsurfref ,CUarray ,unsigned int ))dlsym(handle,"cuSurfRefSetArray");
  mocuSurfRefGetArray = (CUresult (*)(CUarray *,CUsurfref ))dlsym(handle,"cuSurfRefGetArray");
  mocuTexObjectCreate = (CUresult (*)(CUtexObject *,const CUDA_RESOURCE_DESC *,const CUDA_TEXTURE_DESC *,const CUDA_RESOURCE_VIEW_DESC *))dlsym(handle,"cuTexObjectCreate");
  mocuTexObjectDestroy = (CUresult (*)(CUtexObject ))dlsym(handle,"cuTexObjectDestroy");
  mocuTexObjectGetResourceDesc = (CUresult (*)(CUDA_RESOURCE_DESC *,CUtexObject ))dlsym(handle,"cuTexObjectGetResourceDesc");
  mocuTexObjectGetTextureDesc = (CUresult (*)(CUDA_TEXTURE_DESC *,CUtexObject ))dlsym(handle,"cuTexObjectGetTextureDesc");
  mocuTexObjectGetResourceViewDesc = (CUresult (*)(CUDA_RESOURCE_VIEW_DESC *,CUtexObject ))dlsym(handle,"cuTexObjectGetResourceViewDesc");
  mocuSurfObjectCreate = (CUresult (*)(CUsurfObject *,const CUDA_RESOURCE_DESC *))dlsym(handle,"cuSurfObjectCreate");
  mocuSurfObjectDestroy = (CUresult (*)(CUsurfObject ))dlsym(handle,"cuSurfObjectDestroy");
  mocuSurfObjectGetResourceDesc = (CUresult (*)(CUDA_RESOURCE_DESC *,CUsurfObject ))dlsym(handle,"cuSurfObjectGetResourceDesc");
  mocuDeviceCanAccessPeer = (CUresult (*)(int *,CUdevice ,CUdevice ))dlsym(handle,"cuDeviceCanAccessPeer");
  mocuCtxEnablePeerAccess = (CUresult (*)(CUcontext ,unsigned int ))dlsym(handle,"cuCtxEnablePeerAccess");
  mocuCtxDisablePeerAccess = (CUresult (*)(CUcontext ))dlsym(handle,"cuCtxDisablePeerAccess");
  mocuGraphicsUnregisterResource = (CUresult (*)(CUgraphicsResource ))dlsym(handle,"cuGraphicsUnregisterResource");
  mocuGraphicsSubResourceGetMappedArray = (CUresult (*)(CUarray *,CUgraphicsResource ,unsigned int ,unsigned int ))dlsym(handle,"cuGraphicsSubResourceGetMappedArray");
  mocuGraphicsResourceGetMappedMipmappedArray = (CUresult (*)(CUmipmappedArray *,CUgraphicsResource ))dlsym(handle,"cuGraphicsResourceGetMappedMipmappedArray");
  mocuGraphicsResourceGetMappedPointer_v2 = (CUresult (*)(CUdeviceptr *,size_t *,CUgraphicsResource ))dlsym(handle,"cuGraphicsResourceGetMappedPointer_v2");
  mocuGraphicsResourceSetMapFlags = (CUresult (*)(CUgraphicsResource ,unsigned int ))dlsym(handle,"cuGraphicsResourceSetMapFlags");
  mocuGraphicsMapResources = (CUresult (*)(unsigned int ,CUgraphicsResource *,CUstream ))dlsym(handle,"cuGraphicsMapResources");
  mocuGraphicsUnmapResources = (CUresult (*)(unsigned int ,CUgraphicsResource *,CUstream ))dlsym(handle,"cuGraphicsUnmapResources");
  mocuGetExportTable = (CUresult (*)(const void **,const CUuuid *))dlsym(handle,"cuGetExportTable");
  mocuTexRefSetAddress2D_v2 = (CUresult (*)(CUtexref ,const CUDA_ARRAY_DESCRIPTOR *,CUdeviceptr ,size_t ))dlsym(handle,"cuTexRefSetAddress2D_v2");

#if 0
  mocuLinkCreate = (CUresult (*) (unsigned int, CUjit_option *, void **, CUlinkState *))dlsym(handle,"cuLinkCreate");
  mocuLinkAddData = (CUresult (*) (CUlinkState , CUjitInputType , void *, size_t, const char *, unsigned int, CUjit_option *, void **))dlsym(handle,"cuLinkAddData");
  mocuLinkAddFile = (CUresult (*) (CUlinkState , CUjitInputType , const char *, unsigned int , CUjit_option *, void **))dlsym(handle,"cuLinkAddFile");
  mocuLinkComplete = (CUresult (*) (CUlinkState , void **, size_t *))dlsym(handle,"cuLinkComplete");
  mocuLinkDestroy = (CUresult (*) (CUlinkState ))dlsym(handle,"cuLinkDestroy");
#endif

#if DEBUG
  printf("[MOCU] constructor closed.\n");
#endif

  _init_mocu();
  _init_smph();

#if MODE

  //Process can enter here when MODE == 1.
    lock_other_proc();

    int mocu_pos = _get_optimum_device_pos();
  
    if(mocu_pos != mocuID){
      mocu_backup();
      mocu_migrate(mocu_pos);
    }

    unlock_other_proc();

#endif

  initialized = 1;
}


CUresult cuInit(unsigned int Flags)
{
#if DEBUG
  printf("[MOCU] cuInit is called.\n");
#endif
  CUresult res;

  res = mocuInit(Flags);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuInit\n");
#endif
  }

  return res;
}


CUresult cuDriverGetVersion(int *driverVersion)
{
#if DEBUG
  printf("[MOCU] cuDriverGetVersion is called.\n");
#endif
  CUresult res;

  res = mocuDriverGetVersion(driverVersion);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuDriverGetVersion\n");
#endif
  }

  return res;
}

CUresult cuDeviceGet(CUdevice *device,int ordinal)
{
#if DEBUG
  printf("[MOCU] cuDeviceGet is called with devID==%d.\n",ordinal);
#endif
  CUresult res;

  res = mocuDeviceGet(device,ordinal);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuDeviceGet\n");
#endif
  }

  return res;
}


CUresult cuDeviceGetCount(int *count)
{
#if DEBUG
  printf("[MOCU] cuDeviceGetCount is called.\n");
#endif
  CUresult res;
  int _count;

  res = mocuDeviceGetCount(&_count);
  
  *count = _count;

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuDeviceGetCount\n");
#endif
  }

  return res;
}


CUresult cuDeviceGetName(char *name,int len,CUdevice dev)
{
#if DEBUG
  printf("[MOCU] cuDeviceGetName is called.\n");
#endif
  CUresult res;

  res = mocuDeviceGetName(name,len,dev);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuDeviceGetName\n");
#endif
  }

  return res;
}


CUresult cuDeviceTotalMem_v2(size_t *bytes,CUdevice dev)
{
#if DEBUG
  printf("[MOCU] cuDeviceTotalMem_v2 is called.\n");
#endif
  CUresult res;

  res = mocuDeviceTotalMem_v2(bytes,dev);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuDeviceTotalMem_v2\n");
#endif
  }

  return res;
}


CUresult cuDeviceGetAttribute(int *pi,CUdevice_attribute attrib,CUdevice dev)
{
#if DEBUG
  printf("[MOCU] cuDeviceGetAttribute is called.\n");
#endif
  CUresult res;

  res = mocuDeviceGetAttribute(pi,attrib,dev);

  if(res == CUDA_SUCCESS){
    return res;
    
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuDeviceGetAttribute\n");
#endif
  }

  return res;
}


CUresult cuDeviceGetProperties(CUdevprop *prop,CUdevice dev)
{
#if DEBUG
  printf("[MOCU] cuDeviceGetProperties is called.\n");
#endif
  CUresult res;

  res = mocuDeviceGetProperties(prop,dev);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuDeviceGetProperties\n");
#endif
  }

  return res;
}


CUresult cuDeviceComputeCapability(int *major,int *minor,CUdevice dev)
{
#if DEBUG
  printf("[MOCU] cuDeviceComputeCapability is called.\n");
#endif
  CUresult res;

  res = mocuDeviceComputeCapability(major,minor,dev);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuDeviceComputeCapability\n");
#endif
  }

  return res;
}

CUresult cuCtxCreate_v2(CUcontext *pctx,unsigned int flags,CUdevice dev)
{
#if DEBUG||D_CONTEXT
  printf("[MOCU] cuCtxCreate_v2 is called.\n");
#endif
  CUresult res;
  CUcontext ctx;
  context* cp;

  res = mocuCtxCreate_v2(&ctx,flags,dev);

  if(res == CUDA_SUCCESS){
    create_context(ctx);
    
    cp = get_current_context();
    *pctx = (CUcontext)cp;

    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuCtxCreate\n");
#endif
  }

  return res;
}


CUresult cuCtxDestroy_v2(CUcontext ctx)
{
#if DEBUG||D_CONTEXT||D_REGION||D_APILOG
  printf("[MOCU] cuCtxDestroy_v2 is called.\n");
#endif

  CUresult res;
  context* cp;
  thread* tp;;
  event* ep;
  region *r;
  apilog *ap;
  stream *sp;

  cp = (context*)ctx;
  if(cp == NULL) return CUDA_ERROR_INVALID_VALUE;

  res = mocuCtxDestroy_v2(cp->ctx);

  if(res == CUDA_SUCCESS){

    cp->next->prev = cp->prev;
    cp->prev->next = cp->next;

    ap = cp->alog->next;
    while (ap) {
      free(ap->prev);
      ap = ap->next;
    }
    free(cp->alast);

    r = cp->d0->next;
    while (r) {
      free(r->prev);
      r = r->next;
    }
    free(cp->d1);

    /*
      mp = cp->m0->next;
      while (mp) {
      free(mp->prev);
      mp = mp->next;
      }
      free(cp->m1);
    */

    ep = cp->e0->next;
    while (ep) {
      free(ep->prev);
      ep = ep->next;
    }
    free(cp->e1);

    sp = cp->s0->next;
    while (sp) {
      free(sp->prev);
      sp = sp->next;
    }
    free(cp->s1);

    /*
      a = cp->a0->next;
      while (a) {
      free(a->prev);
      a = a->next;
      }
      free(cp->a1);
    */
    free(cp);

    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuCtxDestroy\n");
#endif
  }
  return res;
}


CUresult cuCtxPushCurrent_v2(CUcontext ctx)
{
#if DEBUG||D_CONTEXT
  printf("[MOCU] cuCtxPushCurrent_v2 is called.\n");
#endif
  CUresult res;

  res = mocuCtxPushCurrent_v2(ctx);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuCtxPushCurrent_v2\n");
#endif
  }

  return res;
}


CUresult cuCtxPopCurrent_v2(CUcontext *pctx)
{
#if DEBUG||D_CONTEXT
  printf("[MOCU] cuCtxPopCurrent_v2 is called.\n");
#endif
  CUresult res;

  res = mocuCtxPopCurrent_v2(pctx);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuCtxPopCurrent_v2\n");
#endif
  }

  return res;
}


CUresult cuCtxSetCurrent(CUcontext ctx)
{
#if DEBUG||D_CONTEXT
  printf("[MOCU] cuCtxSetCurrent is called.\n");
#endif
  
  CUresult res;
  context* cp;
  int f_res;

#if DEBUG||D_CONTEXT
  printf("     Target is %p\n",ctx);
#endif

  f_res = create_context(ctx);

#if DEBUG||D_CONTEXT
  if(!f_res)printf("     Success to find context @ cuCtxSetCurrent\n");
  else      printf("     Failed  to find context @ cuCtxSetCurrent\n");
#endif

  cp = get_current_context();
  res = mocuCtxSetCurrent(cp->ctx);

  if(res != CUDA_SUCCESS){
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuCtxSetCurrent\n");
#endif
  }
  return res;
}


CUresult cuCtxGetCurrent(CUcontext *pctx)
{
#if DEBUG||D_CONTEXT
  printf("[MOCU] cuCtxGetCurrent is called.\n");
#endif
  
  CUresult res;
  context* cp;
  CUcontext mctx;

  res = mocuCtxGetCurrent(&mctx);

  if(res == CUDA_SUCCESS){

    if(mctx == NULL)return CUDA_SUCCESS;

    cp = mocu.c0->next;
    while(cp->next != NULL){
      if(cp->ctx == mctx){
	*pctx = cp->user;
	return CUDA_SUCCESS;
      }else{
	cp = cp->next;
      }
    }
    exit(0);
  }else{
    printf("[MOCU] Error @ cuCtxGetCurrent(%d)\n",res);
    return res;
  }
}


CUresult cuCtxGetDevice(CUdevice *device)
{
#if DEBUG
  printf("[MOCU] cuCtxGetDevice is called.\n");
#endif
  CUresult res;

  res = mocuCtxGetDevice(device);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuCtxGetDevice(%d)\n",res);
#endif
  }

  return res;
}


CUresult cuCtxSynchronize()
{
#if DEBUG
  printf("[MOCU] cuCtxSynchronize is called.\n");
#endif
  CUresult res;

  res = mocuCtxSynchronize();

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuCtxSynchronize\n");
#endif
  }

  return res;
}


CUresult cuCtxSetLimit(CUlimit limit,size_t value)
{
#if DEBUG
  printf("[MOCU] cuCtxSetLimit is called.\n");
#endif
  CUresult res;

  res = mocuCtxSetLimit(limit,value);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuCtxSetLimit\n");
#endif
  }

  return res;
}


CUresult cuCtxGetLimit(size_t *pvalue,CUlimit limit)
{
#if DEBUG
  printf("[MOCU] cuCtxGetLimit is called.\n");
#endif
  CUresult res;

  res = mocuCtxGetLimit(pvalue,limit);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuCtxGetLimit\n");
#endif
  }

  return res;
}


CUresult cuCtxGetCacheConfig(CUfunc_cache *pconfig)
{
#if DEBUG
  printf("[MOCU] cuCtxGetCacheConfig is called.\n");
#endif
  CUresult res;

  res = mocuCtxGetCacheConfig(pconfig);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuCtxGetCacheConfig\n");
#endif
  }

  return res;
}


CUresult cuCtxSetCacheConfig(CUfunc_cache config)
{
#if DEBUG
  printf("[MOCU] cuCtxSetCacheConfig is called.\n");
#endif
  CUresult res;

  res = mocuCtxSetCacheConfig(config);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuCtxSetCacheConfig\n");
#endif
  }

  return res;
}


CUresult cuCtxGetSharedMemConfig(CUsharedconfig *pConfig)
{
#if DEBUG
  printf("[MOCU] cuCtxGetSharedMemConfig is called.\n");
#endif
  CUresult res;

  res = mocuCtxGetSharedMemConfig(pConfig);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuCtxGetSharedMemConfig\n");
#endif
  }

  return res;
}


CUresult cuCtxSetSharedMemConfig(CUsharedconfig config)
{
#if DEBUG
  printf("[MOCU] cuCtxSetSharedMemConfig is called.\n");
#endif
  CUresult res;

  res = mocuCtxSetSharedMemConfig(config);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuCtxSetSharedMemConfig\n");
#endif
  }

  return res;
}


CUresult cuCtxGetApiVersion(CUcontext ctx,unsigned int *version)
{
#if DEBUG||D_CONTEXT
  printf("[MOCU] cuCtxGetApiVersion is called.\n");
#endif
  CUresult res;
  context *cp;

  cp = (context*)ctx;

  res = mocuCtxGetApiVersion(cp->ctx,version);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuCtxGetApiVersion\n");
#endif
  }

  return res;
}


CUresult cuCtxGetStreamPriorityRange(int *leastPriority,int *greatestPriority)
{
#if DEBUG
  printf("[MOCU] cuCtxGetStreamPriorityRange is called.\n");
#endif
  CUresult res;

  res = mocuCtxGetStreamPriorityRange(leastPriority,greatestPriority);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuCtxGetStreamPriorityRange\n");
#endif
  }

  return res;
}


CUresult cuCtxAttach(CUcontext *pctx,unsigned int flags)
{
#if DEBUG||D_CONTEXT
  printf("[MOCU] cuCtxAttach is called.\n");
#endif
  CUresult res;

  res = mocuCtxAttach(pctx,flags);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuCtxAttach\n");
#endif
  }

  return res;
}


CUresult cuCtxDetach(CUcontext ctx)
{
#if DEBUG||D_CONTEXT
  printf("[MOCU] cuCtxDetach is called.\n");
#endif
  CUresult res;

  res = mocuCtxDetach(ctx);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuCtxDetach\n");
#endif
  }

  return res;
}


CUresult cuModuleLoad(CUmodule *_module,const char *fname)
{
  //#if DEBUG||D_MODULE||D_APILOG
#if 1
  printf("[MOCU] cuModuleLoad is called.\n");
#endif

#if 1

  CUresult res;
  context* cp;
  CUmodule m;
  module* mp;
  apilog* a;
  char* image;
  int fd;

  res = mocuModuleLoad(&m, fname);

  if(res == CUDA_SUCCESS){

    cp = get_current_context();

    mp = (module*)malloc(sizeof(module));
    mp->m = m;
    mp->mode = 2;
    mp->prev = cp->m1->prev;
    mp->next = cp->m1;
    mp->prev->next = mp;
    mp->next->prev = mp;
    module_init(mp);

    a = (apilog*)malloc(sizeof(apilog));
    a->type = MODULE_LOAD;
    a->data.moduleLoad.mod = mp;
    a->next = NULL;
    a->prev = cp->alast;
    cp->alast->next = a;
    cp->alast = a;

    *_module = (CUmodule)mp;

  }

  return res;
#endif

#if 0
 
  CUresult res;

  res = mocuModuleLoad(_module,fname);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuModuleLoad\n");
#endif
  }

  return res;
#endif
 
}


CUresult cuModuleLoadData(CUmodule *_module,const void *vimage)
{
  //#if DEBUG||D_MODULE||D_APILOG
#if 1
  printf("[MOCU] cuModuleLoadData is called.\n");
#endif

#if 1

  CUresult res;
  context *cp;
  CUmodule m;
  module *mp;
  apilog *a;
  char *image;

  image = (char*)vimage;

  res = mocuModuleLoadData(&m,image);

  if(res == CUDA_SUCCESS){

    cp = get_current_context();

    mp = (module*)malloc(sizeof(module));
    mp->m = m;
    mp->mode = 3;
    mp->prev = cp->m1->prev;
    mp->next = cp->m1;
    mp->prev->next = mp;
    mp->next->prev = mp;
    module_init(mp);

    a = (apilog*)malloc(sizeof(apilog));
    a->type = MODULE_LOAD_DATA;
    a->data.moduleLoadData.mod = mp;
    a->next = NULL;
    a->prev = cp->alast;
    cp->alast->next = a;
    cp->alast = a;

    //nvcr_module_save_source(mp, image);
    //nvcr_collect_symbols(mp);
    
    *_module = (CUmodule)mp;
  }

  return res;

#endif

#if 0

  CUresult res;

  res = mocuModuleLoadData(_module,vimage);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuModuleLoadData\n");
#endif
  }

  return res;

#endif
}


CUresult cuModuleLoadDataEx(CUmodule *_module,const void *vimage,unsigned int numOptions,CUjit_option *options,void **optionValues)
{
  //#if DEBUG||D_MODULE||D_APILOG
#if 1
  printf("[MOCU] cuModuleLoadDataEx is called.\n");
#endif

#if 1

  CUresult res;
  context *cp;
  CUmodule m;
  module *mp;
  apilog *a;
  char *image;

  image = (char*)vimage;
  
  res = mocuModuleLoadDataEx(&m,image,numOptions,options,optionValues);

  if(res == CUDA_SUCCESS){

    cp = get_current_context();

    mp = (module *)malloc(sizeof(module));
    mp->m = m;
    mp->mode = 4;
    mp->prev = cp->m1->prev;
    mp->next = cp->m1;
    mp->prev->next = mp;
    mp->next->prev = mp;
    module_init(mp);

    a = (apilog *)malloc(sizeof(apilog));
    a->type = MODULE_LOAD_DATA_EX;
    a->data.moduleLoadDataEx.mod = mp;
    a->next = NULL;
    a->prev = cp->alast;
    cp->alast->next = a;
    cp->alast = a;

    *_module = (CUmodule)mp;

  }

#endif

#if 0

  CUresult res;

  res = mocuModuleLoadDataEx(_module,vimage,numOptions,options,optionValues);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuModuleLoadDataEx\n");
#endif
  }

  return res;

#endif
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr,size_t *bytes,CUmodule hmod,const char *name)
{
  //#if DEBUG||D_MODULE
#if 1
  printf("[MOCU] cuModuleGetGlobal_v2 is called.\n");
#endif

#if 1

  CUresult res;
  module *mp;
  context* cp;

  //  mp = (module*)hmod;
  cp = get_current_context();
  mp = get_module_correspond_to_hmod(cp,hmod,1);

  if(mp==NULL)return CUDA_ERROR_INVALID_VALUE;

  res = mocuModuleGetGlobal_v2(dptr,bytes,mp->m,name);

#endif

#if 0

  CUresult res;

  res = mocuModuleGetGlobal_v2(dptr,bytes,hmod,name);
  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuModuleGetGlobal_v2\n");
#endif
  }

#endif

  return res;
}

CUresult cuModuleLoadFatBinary(CUmodule *_module,const void *fatCubin)
{
#if DEBUG||D_MODULE||D_APILOG
  printf("[MOCU] cuModuleLoadFatBinary is called.\n");
#endif

#if 1

  CUresult res;
  context *cp;
  CUmodule m;
  module *mp;
  apilog *a;
  symbol *s;
  
  int fd;
  int len;
  //  __cudaFatCudaBinary *fb;

  res = mocuModuleLoadFatBinary(&m,fatCubin);

  if(res == CUDA_SUCCESS){

    cp = get_current_context();

    mp = (module*)malloc(sizeof(module));
    mp->m = m;
    mp->mode = 5;
    mp->next = cp->m1;
    mp->prev = cp->m1->prev;
    mp->prev->next = mp;
    mp->next->prev = mp;
    mp->len = 0;
    mp->source = (char *)fatCubin;    
    module_init(mp);

    a = (apilog*)malloc(sizeof(apilog));
    a->type = MODULE_LOAD_FAT_BINARY;
    a->data.moduleLoadFatBinary.mod = mp;
    a->next = NULL;
    a->prev = cp->alast;
    cp->alast->next = a;
    cp->alast = a;
    
    //    fb = (__cudaFatCudaBinary *)fatCubin;

    *_module = (CUmodule)mp;

    /*
      if (fb->cubin->cubin)
      nvcr_cubin_symbols(mp, fb->cubin->cubin);
      else
      nvcr_ptx_symbols(mp, fb->ptx->ptx);
    */

    s = mp->s0->next;
    while(s->mode >= 0){
      if(s->type == SYMBOL_DEVICE || s->type == SYMBOL_CONST){
	mocuModuleGetGlobal_v2(&s->addr,&s->size,mp->m,s->name);
      }
      s = s->next;
    }

  }

  return res;

#endif

#if 0

  CUresult res;

  res = mocuModuleLoadFatBinary(_module,fatCubin);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuModuleLoadFatBinary\n");
#endif
  }

  return res;

#endif
}


CUresult cuModuleUnload(CUmodule hmod)
{
#if DEBUG||D_MODULE||D_APILOG
  printf("[MOCU] cuModuleUnload is called.\n");
#endif

  CUresult res;
  context *cp;
  module *mp;
  apilog *a;

  cp = get_current_context();
  mp = get_module_correspond_to_hmod(cp,hmod,0);

  if(mp == NULL){
#if DEBUG||D_MODULE||D_APILOG    
    printf("  module is NULL\n");
#endif
    return mocuModuleUnload(hmod);
  }

  res = mocuModuleUnload(mp->m);

  if(res == CUDA_SUCCESS){

    a = (apilog*)malloc(sizeof(apilog));
    a->type = MODULE_UNLOAD;
    a->data.moduleUnload.mod = mp;
    a->next = NULL;
    a->prev = cp->alast;
    cp->alast->next = a;
    cp->alast = a;

    mp->next = mp->prev->next;
    mp->prev = mp->next->prev;
    free(mp); 
    
  }else{
#if DEBUG_ERROR
    //    printf("[MOCU] Error @ cuModuleUnload(%d)\n",res);
#endif
  }

  _detach_smph();

  return res;

}

CUresult cuModuleGetFunction(CUfunction *hfunc,CUmodule hmod,const char *name)
{
#if DEBUG||D_FUNCTION||D_MODULE
  printf("[MOCU] cuModuleGetFunction is called. (module : %p)\n",hmod);
#endif

  CUresult res;
  context* cp;
  module* mp;
  function* fp;
  CUfunction f;
  int len;

  cp = get_current_context();
  mp = get_module_correspond_to_hmod(cp,hmod,1);

  /*
    CUcontext ctx;
    CUdevice dev;

    mocuDeviceGet(&dev, 1);
    mocuCtxCreate_v2(&ctx, 0, dev);
    res = mocuModuleGetFunction(&f, mp->m, name);
    printf(" ======>  res=%d\n", res);
    mocuCtxDestroy_v2(ctx);
  */

  //  res = mocuCtxSetCurrent(cp->user);printf("@@@@@@@@@@@@@@@@@@@@ Get Funtion(user) : %d\n",res);
  res = mocuModuleGetFunction(&f,mp->m,name);
  //  printf("                  : %p\n",f);

  //  res = mocuCtxSetCurrent(cp->ctx);printf("@@@@@@@@@@@@@@@@@@@@ Get Funtion(ctx)  : %d\n",res);
  res = mocuModuleGetFunction(&f,mp->m,name);
  //  printf("                  : %p\n",f);


  if(res == CUDA_SUCCESS){
    fp = (function*)malloc(sizeof(function));
    fp->f = f;
    fp->mode = 10;
    len = strlen(name);
    fp->name = malloc(len+1);
    fp->shared_size = 0;
    fp->cache = CU_FUNC_CACHE_PREFER_NONE;
    strcpy(fp->name, name);
    fp->prev = mp->f1->prev;
    fp->next = mp->f1;
    fp->prev->next = fp;
    fp->next->prev = fp;

    fp->t0 = (texref *)malloc(sizeof(texref));
    fp->t1 = (texref *)malloc(sizeof(texref));
    fp->t0->prev = fp->t1->next = NULL;
    fp->t0->mode = fp->t1->mode = -1;
    fp->t0->t = fp->t1->t = NULL;
    fp->t0->next = fp->t1;
    fp->t1->prev = fp->t0;

    *hfunc = (CUfunction)fp;

  }

  return res;

#if 0

  CUresult res;

  res = mocuModuleGetFunction(hfunc,hmod,name);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuModuleGetFunction\n");
#endif
  }

  return res;

#endif
}


CUresult cuModuleGetTexRef(CUtexref *pTexRef,CUmodule hmod,const char *name)
{
#if DEBUG||D_TEXREF||D_MODULE
  printf("[MOCU] cuModuleGetTexRef is called.\n");
#endif

#if 1

  CUresult res;
  texref *tp;
  module *mp;
  context *cp;
  CUtexref t;
  int len;

  cp = get_current_context();
  mp = get_module_correspond_to_hmod(cp,hmod,1);

  res = mocuModuleGetTexRef(&t,mp->m,name);

  if(res == CUDA_SUCCESS){
    tp = (texref*)malloc(sizeof(texref));
    tp->t = t;
    tp->mode = 0;
    //    tp->m = mp;
    tp->type = 0;
    tp->name = (char*)malloc(strlen(name) + 1);
    strcpy(tp->name,name);
    tp->prev = cp->t1->prev;
    tp->next = cp->t1;
    tp->prev->next = tp;
    tp->next->prev = tp;

    *pTexRef = (CUtexref)tp;
  }

  return res;

#endif

#if 0

  CUresult res;

  res = mocuModuleGetTexRef(pTexRef,hmod,name);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuModuleGetTexRef\n");
#endif
  }

  return res;

#endif
}


CUresult cuModuleGetSurfRef(CUsurfref *pSurfRef,CUmodule hmod,const char *name)
{
#if DEBUG||D_MODULE
  printf("[MOCU] cuModuleGetSurfRef is called.\n");
#endif
  CUresult res;

#if DEBUG

  module *mp;

  mp = (module*)hmod;

  if(mp == NULL) return CUDA_ERROR_INVALID_VALUE;

  res = mocuModuleGetSurfRef(pSurfRef,mp->m,name);
#endif

#if 0
  res = mocuModuleGetSurfRef(pSurfRef,hmod,name);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuModuleGetSurfRef\n");
#endif
  }

#endif

  return res;
}


CUresult cuMemGetInfo_v2(size_t *free,size_t *total)
{
#if DEBUG
  printf("[MOCU] cuMemGetInfo_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemGetInfo_v2(free,total);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemGetInfo_v2\n");
#endif
  }

  return res;
}


CUresult cuMemAlloc_v2(CUdeviceptr *dptr,size_t bytesize)
{
#if DEBUG||D_REGION||D_APILOG
  printf("[MOCU] cuMemAlloc_v2 is called.\n");
#endif

  CUresult res;
  apilog* a;
  region* r;
  context* cp;

  res = mocuMemAlloc_v2(dptr,bytesize);

  if(res == CUDA_ERROR_OUT_OF_MEMORY){

    size_t memSize;
    nvmlMemory_t mem_info;
    nvmlReturn_t _res;
    unsigned long long freeMem;

    memSize = check_memory_amount_used();

    mocu_backup();

    int optimum_pos = _get_optimum_device_pos();

    int i;
    for(i = optimum_pos ; i < mocu.ndev + optimum_pos ; i ++){

      if(i%mocu.ndev == mocuID)continue;

      lock_other_proc();

      _res = nvmlDeviceGetMemoryInfo(mocu.nvml_dev[i%mocu.ndev],&mem_info);

      if(_res != NVML_SUCCESS){
	printf("Failed to get memory information ... @device%d\n",i);
      }

      freeMem = mem_info.free;

#if DEBUG_MIG
      printf("+--------------------------------------------+\n");
      printf("|Device %2d                                   |\n",i%mocu.ndev);
      printf("+============================================+\n");
      printf("| Free  memory region %10lld[byte]       |\n",freeMem);
      printf("| Used  memory region %10lld[byte]       |\n",mem_info.used);
      printf("| Total memory region %10lld[byte]       |\n",mem_info.total);
      printf("+--------------------------------------------+\n");
#endif

      if(freeMem > memSize + bytesize + 64*1024*1024){

	mocu_migrate(i%mocu.ndev);

	res = mocuMemAlloc_v2(dptr,bytesize);

	if(res == CUDA_SUCCESS){
	  i = mocu.ndev + optimum_pos;
	}else{
	  printf("Failed to allocate memory\n");
	  if(res != CUDA_ERROR_OUT_OF_MEMORY){
	    printf("Mobile CUDA exit with ERROR CODE : %d\n",res);
	    exit(1);
	  }
	}
      }else if(i == mocu.ndev + optimum_pos - 1){
	printf("+---------------------------+\n");
	printf("|       *  Warning  *       |\n");
	printf("+---------------------------+\n");
	printf("| There is no enough region |\n");
	printf("|   This Process will exit  |\n");
	printf("+---------------------------+\n");
	unlock_other_proc();
	exit(-1);
      }
      unlock_other_proc();
    }
  }
  
  if(res == CUDA_SUCCESS){

    cp = get_current_context();

    a = (apilog*)malloc(sizeof(apilog));
    a->type = MEM_ALLOC;
    a->data.memAlloc.size = bytesize;
    a->data.memAlloc.addr = *dptr;

    a->next = NULL;
    a->prev = cp->alast;
    cp->alast->next = a;
    cp->alast = a;

    r = (region*)malloc(sizeof(region));
    r->size = bytesize;
    r->base = *dptr;
    r->mode = 0;
    r->prev = cp->d1->prev;
    r->next = cp->d1;
    r->prev->next = r;
    r->next->prev = r;

    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemAlloc%d\n",res);
#endif
  }

  return res;
}


CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr,size_t *pPitch,size_t WidthInBytes,size_t Height,unsigned int ElementSizeBytes)
{
#if DEBUG
  printf("[MOCU] cuMemAllocPitch_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemAllocPitch_v2(dptr,pPitch,WidthInBytes,Height,ElementSizeBytes);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemAllocPitch_v2\n");
#endif
  }

  return res;
}


CUresult cuMemFree_v2(CUdeviceptr dptr)
{
#if DEBUG||D_REGION||D_APILOG
  printf("[MOCU] cuMemFree_v2 is called.\n");
#endif

  CUresult res;
  apilog* a;
  region* r;
  context* cp;

  res = mocuMemFree_v2(dptr);

  if(res == CUDA_SUCCESS){
    
    cp = get_current_context();

    a = (apilog*)malloc(sizeof(apilog));
    a->type = MEM_FREE;
    a->data.memFree.addr = dptr;

    a->next = NULL;
    a->prev = cp->alast;
    cp->alast->next = a;
    cp->alast = a;

    cp = mocu.c0->next;
    while(cp->next != NULL){
      int suc = 0;
      region* r = cp->d0->next;
      while(r->next != NULL){
	if(r->base == dptr){
	  suc = 1;
	  r->next->prev = r->prev;
	  r->prev->next = r->next;
	  free(r);
	  break;
	}else{
	  r = r->next;
	}
      }
      if(suc) break;
      cp = cp->next;
    }


    /*
      int counter = 0;
      r = cp->d0->next;
      while(r->mode >= 0){
      if(r->base == dptr) break;
      r = r->next;
      counter ++;
      }

      if(r->mode < 0){
      printf("Unknown global memory region.(%d) @ %p\n",counter+1,dptr);
      exit(0);
      }

      r->next->prev = r->prev;
      r->prev->next = r->next;
      free(r);

    */

    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemFree\n");
#endif
  }

  return res;

}


CUresult cuMemGetAddressRange_v2(CUdeviceptr *pbase,size_t *psize,CUdeviceptr dptr)
{
#if DEBUG
  printf("[MOCU] cuMemGetAddressRange_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemGetAddressRange_v2(pbase,psize,dptr);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemGetAddressRange_v2\n");
#endif
  }

  return res;
}


CUresult cuMemAllocHost_v2(void **pp,size_t bytesize)
{
#if DEBUG
  printf("[MOCU] cuMemAllocHost_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemAllocHost_v2(pp,bytesize);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemAllocHost_v2\n");
#endif
  }

  return res;
}


CUresult cuMemFreeHost(void *p)
{
#if DEBUG
  printf("[MOCU] cuMemFreeHost is called.\n");
#endif
  CUresult res;

  res = mocuMemFreeHost(p);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemFreeHost\n");
#endif
  }

  return res;
}


CUresult cuMemHostAlloc(void **pp,size_t bytesize,unsigned int Flags)
{
#if DEBUG
  printf("[MOCU] cuMemHostAlloc is called.\n");
#endif
  CUresult res;

  res = mocuMemHostAlloc(pp,bytesize,Flags | CU_MEMHOSTALLOC_PORTABLE);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemHostAlloc\n");
#endif
  }

  return res;
}


CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr,void *p,unsigned int Flags)
{
#if DEBUG
  printf("[MOCU] cuMemHostGetDevicePointer_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemHostGetDevicePointer_v2(pdptr,p,Flags);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemHostGetDevicePointer_v2\n");
#endif
  }

  return res;
}


CUresult cuMemHostGetFlags(unsigned int *pFlags,void *p)
{
#if DEBUG
  printf("[MOCU] cuMemHostGetFlags is called.\n");
#endif
  CUresult res;

  res = mocuMemHostGetFlags(pFlags,p);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemHostGetFlags\n");
#endif
  }

  return res;
}


CUresult cuDeviceGetByPCIBusId(CUdevice *dev,char *pciBusId)
{
#if DEBUG
  printf("[MOCU] cuDeviceGetByPCIBusId is called.\n");
#endif
  CUresult res;

  res = mocuDeviceGetByPCIBusId(dev,pciBusId);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuDeviceGetByPCIBusId\n");
#endif
  }

  return res;
}


CUresult cuDeviceGetPCIBusId(char *pciBusId,int len,CUdevice dev)
{
#if DEBUG
  printf("[MOCU] cuDeviceGetPCIBusId is called.\n");
#endif
  CUresult res;

  res = mocuDeviceGetPCIBusId(pciBusId,len,dev);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuDeviceGetPCIBusId\n");
#endif
  }

  return res;
}


CUresult cuIpcGetEventHandle(CUipcEventHandle *pHandle,CUevent event)
{
#if DEBUG||D_EVENT
  printf("[MOCU] cuIpcGetEventHandle is called.\n");
#endif
  CUresult res;

  res = mocuIpcGetEventHandle(pHandle,event);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuIpcGetEventHandle\n");
#endif
  }

  return res;
}


CUresult cuIpcOpenEventHandle(CUevent *phEvent,CUipcEventHandle handle)
{
#if DEBUG||D_EVENT
  printf("[MOCU] cuIpcOpenEventHandle is called.\n");
#endif
  CUresult res;

  res = mocuIpcOpenEventHandle(phEvent,handle);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuIpcOpenEventHandle\n");
#endif
  }

  return res;
}


CUresult cuIpcGetMemHandle(CUipcMemHandle *pHandle,CUdeviceptr dptr)
{
#if DEBUG
  printf("[MOCU] cuIpcGetMemHandle is called.\n");
#endif
  CUresult res;

  res = mocuIpcGetMemHandle(pHandle,dptr);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuIpcGetMemHandle\n");
#endif
  }

  return res;
}


CUresult cuIpcOpenMemHandle(CUdeviceptr *pdptr,CUipcMemHandle handle,unsigned int Flags)
{
#if DEBUG
  printf("[MOCU] cuIpcOpenMemHandle is called.\n");
#endif
  CUresult res;

  res = mocuIpcOpenMemHandle(pdptr,handle,Flags);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuIpcOpenMemHandle\n");
#endif
  }

  return res;
}


CUresult cuIpcCloseMemHandle(CUdeviceptr dptr)
{
#if DEBUG
  printf("[MOCU] cuIpcCloseMemHandle is called.\n");
#endif
  CUresult res;

  res = mocuIpcCloseMemHandle(dptr);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuIpcCloseMemHandle\n");
#endif
  }

  return res;
}


CUresult cuMemHostRegister(void *p,size_t bytesize,unsigned int Flags)
{
#if DEBUG
  printf("[MOCU] cuMemHostRegister is called.\n");
#endif
  CUresult res;

  res = mocuMemHostRegister(p,bytesize,Flags | CU_MEMHOSTALLOC_PORTABLE);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemHostRegister\n");
#endif
  }

  return res;
}


CUresult cuMemHostUnregister(void *p)
{
#if DEBUG
  printf("[MOCU] cuMemHostUnregister is called.\n");
#endif
  CUresult res;

  res = mocuMemHostUnregister(p);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemHostUnregister\n");
#endif
  }

  return res;
}


CUresult cuMemcpy(CUdeviceptr dst,CUdeviceptr src,size_t ByteCount)
{
#if DEBUG
  printf("[MOCU] cuMemcpy is called.\n");
#endif
  CUresult res;

  res = mocuMemcpy(dst,src,ByteCount);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpy\n");
#endif
  }

  return res;
}


CUresult cuMemcpyPeer(CUdeviceptr dstDevice,CUcontext dstContext,CUdeviceptr srcDevice,CUcontext srcContext,size_t ByteCount)
{
#if DEBUG||D_CONTEXT
  printf("[MOCU] cuMemcpyPeer is called.\n");
#endif
  CUresult res;

  res = mocuMemcpyPeer(dstDevice,dstContext,srcDevice,srcContext,ByteCount);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpyPeer\n");
#endif
  }

  return res;
}


CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice,const void *srcHost,size_t ByteCount)
{
#if DEBUG
  printf("[MOCU] cuMemcpyHtoD_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemcpyHtoD_v2(dstDevice,srcHost,ByteCount);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpyHtoD_v2\n");
#endif
  }

  return res;
}


CUresult cuMemcpyDtoH_v2(void *dstHost,CUdeviceptr srcDevice,size_t ByteCount)
{
#if DEBUG
  printf("[MOCU] cuMemcpyDtoH_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemcpyDtoH_v2(dstHost,srcDevice,ByteCount);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpyDtoH_v2\n");
#endif
  }

  return res;
}


CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice,CUdeviceptr srcDevice,size_t ByteCount)
{
#if DEBUG
  printf("[MOCU] cuMemcpyDtoD_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemcpyDtoD_v2(dstDevice,srcDevice,ByteCount);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpyDtoD_v2\n");
#endif
  }

  return res;
}


CUresult cuMemcpyDtoA_v2(CUarray dstArray,size_t dstOffset,CUdeviceptr srcDevice,size_t ByteCount)
{
#if DEBUG
  printf("[MOCU] cuMemcpyDtoA_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemcpyDtoA_v2(dstArray,dstOffset,srcDevice,ByteCount);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpyDtoA_v2\n");
#endif
  }

  return res;
}


CUresult cuMemcpyAtoD_v2(CUdeviceptr dstDevice,CUarray srcArray,size_t srcOffset,size_t ByteCount)
{
#if DEBUG
  printf("[MOCU] cuMemcpyAtoD_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemcpyAtoD_v2(dstDevice,srcArray,srcOffset,ByteCount);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpyAtoD_v2\n");
#endif
  }

  return res;
}


CUresult cuMemcpyHtoA_v2(CUarray dstArray,size_t dstOffset,const void *srcHost,size_t ByteCount)
{
#if DEBUG
  printf("[MOCU] cuMemcpyHtoA_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemcpyHtoA_v2(dstArray,dstOffset,srcHost,ByteCount);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpyHtoA_v2\n");
#endif
  }

  return res;
}


CUresult cuMemcpyAtoH_v2(void *dstHost,CUarray srcArray,size_t srcOffset,size_t ByteCount)
{
#if DEBUG
  printf("[MOCU] cuMemcpyAtoH_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemcpyAtoH_v2(dstHost,srcArray,srcOffset,ByteCount);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpyAtoH_v2\n");
#endif
  }

  return res;
}


CUresult cuMemcpyAtoA_v2(CUarray dstArray,size_t dstOffset,CUarray srcArray,size_t srcOffset,size_t ByteCount)
{
#if DEBUG
  printf("[MOCU] cuMemcpyAtoA_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemcpyAtoA_v2(dstArray,dstOffset,srcArray,srcOffset,ByteCount);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpyAtoA_v2\n");
#endif
  }

  return res;
}


CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy)
{
#if DEBUG
  printf("[MOCU] cuMemcpy2D_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemcpy2D_v2(pCopy);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpy2D_v2\n");
#endif
  }

  return res;
}


CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy)
{
#if DEBUG
  printf("[MOCU] cuMemcpy2DUnaligned_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemcpy2DUnaligned_v2(pCopy);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpy2DUnaligned_v2\n");
#endif
  }

  return res;
}


CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy)
{
#if DEBUG
  printf("[MOCU] cuMemcpy3D_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemcpy3D_v2(pCopy);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpy3D_v2\n");
#endif
  }

  return res;
}


CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy)
{
#if DEBUG
  printf("[MOCU] cuMemcpy3DPeer is called.\n");
#endif
  CUresult res;

  res = mocuMemcpy3DPeer(pCopy);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpy3DPeer\n");
#endif
  }

  return res;
}


CUresult cuMemcpyAsync(CUdeviceptr dst,CUdeviceptr src,size_t ByteCount,CUstream hStream)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuMemcpyAsync is called.\n");
#endif
  CUresult res;

  res = mocuMemcpyAsync(dst,src,ByteCount,hStream);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpyAsync\n");
#endif
  }

  return res;
}


CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice,CUcontext dstContext,CUdeviceptr srcDevice,CUcontext srcContext,size_t ByteCount,CUstream hStream)
{
#if DEBUG||D_CONTEXT||D_STREAM
  printf("[MOCU] cuMemcpyPeerAsync is called.\n");
#endif
  CUresult res;

  res = mocuMemcpyPeerAsync(dstDevice,dstContext,srcDevice,srcContext,ByteCount,hStream);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpyPeerAsync\n");
#endif
  }

  return res;
}


CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice,const void *srcHost,size_t ByteCount,CUstream hStream)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuMemcpyHtoDAsync_v2 is called.\n");
#endif
  CUresult res;
  stream* sp;

  sp = (stream*)hStream;

  res = mocuMemcpyHtoDAsync_v2(dstDevice,srcHost,ByteCount, sp == NULL ? NULL : sp->s);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpyHtoDAsync_v2\n");
#endif
  }

  return res;
}


CUresult cuMemcpyDtoHAsync_v2(void *dstHost,CUdeviceptr srcDevice,size_t ByteCount,CUstream hStream)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuMemcpyDtoHAsync_v2 is called.\n");
#endif
  CUresult res;
  stream* sp;

  sp = (stream*)hStream;

  res = mocuMemcpyDtoHAsync_v2(dstHost,srcDevice,ByteCount,sp == NULL ? NULL : sp->s);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpyDtoHAsync_v2\n");
#endif
  }

  return res;
}


CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice,CUdeviceptr srcDevice,size_t ByteCount,CUstream hStream)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuMemcpyDtoDAsync_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemcpyDtoDAsync_v2(dstDevice,srcDevice,ByteCount,hStream);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpyDtoDAsync_v2\n");
#endif
  }

  return res;
}


CUresult cuMemcpyHtoAAsync_v2(CUarray dstArray,size_t dstOffset,const void *srcHost,size_t ByteCount,CUstream hStream)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuMemcpyHtoAAsync_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemcpyHtoAAsync_v2(dstArray,dstOffset,srcHost,ByteCount,hStream);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpyHtoAAsync_v2\n");
#endif
  }

  return res;
}


CUresult cuMemcpyAtoHAsync_v2(void *dstHost,CUarray srcArray,size_t srcOffset,size_t ByteCount,CUstream hStream)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuMemcpyAtoHAsync_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemcpyAtoHAsync_v2(dstHost,srcArray,srcOffset,ByteCount,hStream);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpyAtoHAsync_v2\n");
#endif
  }

  return res;
}


CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy,CUstream hStream)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuMemcpy2DAsync_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemcpy2DAsync_v2(pCopy,hStream);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpy2DAsync_v2\n");
#endif
  }

  return res;
}


CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy,CUstream hStream)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuMemcpy3DAsync_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemcpy3DAsync_v2(pCopy,hStream);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpy3DAsync_v2\n");
#endif
  }

  return res;
}


CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy,CUstream hStream)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuMemcpy3DPeerAsync is called.\n");
#endif
  CUresult res;

  res = mocuMemcpy3DPeerAsync(pCopy,hStream);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemcpy3DPeerAsync\n");
#endif
  }

  return res;
}


CUresult cuMemsetD8_v2(CUdeviceptr dstDevice,unsigned char uc,size_t N)
{
#if DEBUG
  printf("[MOCU] cuMemsetD8_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemsetD8_v2(dstDevice,uc,N);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemsetD8_v2\n");
#endif
  }

  return res;
}


CUresult cuMemsetD16_v2(CUdeviceptr dstDevice,unsigned short us,size_t N)
{
#if DEBUG
  printf("[MOCU] cuMemsetD16_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemsetD16_v2(dstDevice,us,N);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemsetD16_v2\n");
#endif
  }

  return res;
}


CUresult cuMemsetD32_v2(CUdeviceptr dstDevice,unsigned int ui,size_t N)
{
#if DEBUG
  printf("[MOCU] cuMemsetD32_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemsetD32_v2(dstDevice,ui,N);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemsetD32_v2\n");
#endif
  }

  return res;
}


CUresult cuMemsetD2D8_v2(CUdeviceptr dstDevice,size_t dstPitch,unsigned char uc,size_t Width,size_t Height)
{
#if DEBUG
  printf("[MOCU] cuMemsetD2D8_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemsetD2D8_v2(dstDevice,dstPitch,uc,Width,Height);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemsetD2D8_v2\n");
#endif
  }

  return res;
}


CUresult cuMemsetD2D16_v2(CUdeviceptr dstDevice,size_t dstPitch,unsigned short us,size_t Width,size_t Height)
{
#if DEBUG
  printf("[MOCU] cuMemsetD2D16_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemsetD2D16_v2(dstDevice,dstPitch,us,Width,Height);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemsetD2D16_v2\n");
#endif
  }

  return res;
}


CUresult cuMemsetD2D32_v2(CUdeviceptr dstDevice,size_t dstPitch,unsigned int ui,size_t Width,size_t Height)
{
#if DEBUG
  printf("[MOCU] cuMemsetD2D32_v2 is called.\n");
#endif
  CUresult res;

  res = mocuMemsetD2D32_v2(dstDevice,dstPitch,ui,Width,Height);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemsetD2D32_v2\n");
#endif
  }

  return res;
}


CUresult cuMemsetD8Async(CUdeviceptr dstDevice,unsigned char uc,size_t N,CUstream hStream)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuMemsetD8Async is called.\n");
#endif
  CUresult res;

  res = mocuMemsetD8Async(dstDevice,uc,N,hStream);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemsetD8Async\n");
#endif
  }

  return res;
}


CUresult cuMemsetD16Async(CUdeviceptr dstDevice,unsigned short us,size_t N,CUstream hStream)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuMemsetD16Async is called.\n");
#endif
  CUresult res;

  res = mocuMemsetD16Async(dstDevice,us,N,hStream);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemsetD16Async\n");
#endif
  }

  return res;
}


CUresult cuMemsetD32Async(CUdeviceptr dstDevice,unsigned int ui,size_t N,CUstream hStream)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuMemsetD32Async is called.\n");
#endif
  CUresult res;

  res = mocuMemsetD32Async(dstDevice,ui,N,hStream);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemsetD32Async\n");
#endif
  }

  return res;
}


CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice,size_t dstPitch,unsigned char uc,size_t Width,size_t Height,CUstream hStream)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuMemsetD2D8Async is called.\n");
#endif
  CUresult res;

  res = mocuMemsetD2D8Async(dstDevice,dstPitch,uc,Width,Height,hStream);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemsetD2D8Async\n");
#endif
  }

  return res;
}


CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice,size_t dstPitch,unsigned short us,size_t Width,size_t Height,CUstream hStream)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuMemsetD2D16Async is called.\n");
#endif
  CUresult res;

  res = mocuMemsetD2D16Async(dstDevice,dstPitch,us,Width,Height,hStream);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemsetD2D16Async\n");
#endif
  }

  return res;
}


CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice,size_t dstPitch,unsigned int ui,size_t Width,size_t Height,CUstream hStream)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuMemsetD2D32Async is called.\n");
#endif
  CUresult res;

  res = mocuMemsetD2D32Async(dstDevice,dstPitch,ui,Width,Height,hStream);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMemsetD2D32Async\n");
#endif
  }

  return res;
}


CUresult cuArrayCreate_v2(CUarray *pHandle,const CUDA_ARRAY_DESCRIPTOR *pAllocateArray)
{
#if DEBUG
  printf("[MOCU] cuArrayCreate_v2 is called.\n");
#endif
  CUresult res;

  res = mocuArrayCreate_v2(pHandle,pAllocateArray);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuArrayCreate_v2\n");
#endif
  }

  return res;
}


CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor,CUarray hArray)
{
#if DEBUG
  printf("[MOCU] cuArrayGetDescriptor_v2 is called.\n");
#endif
  CUresult res;

  res = mocuArrayGetDescriptor_v2(pArrayDescriptor,hArray);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuArrayGetDescriptor_v2\n");
#endif
  }

  return res;
}


CUresult cuArrayDestroy(CUarray hArray)
{
#if DEBUG
  printf("[MOCU] cuArrayDestroy is called.\n");
#endif
  CUresult res;

  res = mocuArrayDestroy(hArray);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuArrayDestroy\n");
#endif
  }

  return res;
}


CUresult cuArray3DCreate_v2(CUarray *pHandle,const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray)
{
#if DEBUG
  printf("[MOCU] cuArray3DCreate_v2 is called.\n");
#endif
  CUresult res;

  res = mocuArray3DCreate_v2(pHandle,pAllocateArray);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuArray3DCreate_v2\n");
#endif
  }

  return res;
}


CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor,CUarray hArray)
{
#if DEBUG
  printf("[MOCU] cuArray3DGetDescriptor_v2 is called.\n");
#endif
  CUresult res;

  res = mocuArray3DGetDescriptor_v2(pArrayDescriptor,hArray);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuArray3DGetDescriptor_v2\n");
#endif
  }

  return res;
}


CUresult cuMipmappedArrayCreate(CUmipmappedArray *pHandle,const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,unsigned int numMipmapLevels)
{
#if DEBUG
  printf("[MOCU] cuMipmappedArrayCreate is called.\n");
#endif
  CUresult res;

  res = mocuMipmappedArrayCreate(pHandle,pMipmappedArrayDesc,numMipmapLevels);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMipmappedArrayCreate\n");
#endif
  }

  return res;
}


CUresult cuMipmappedArrayGetLevel(CUarray *pLevelArray,CUmipmappedArray hMipmappedArray,unsigned int level)
{
#if DEBUG
  printf("[MOCU] cuMipmappedArrayGetLevel is called.\n");
#endif
  CUresult res;

  res = mocuMipmappedArrayGetLevel(pLevelArray,hMipmappedArray,level);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMipmappedArrayGetLevel\n");
#endif
  }

  return res;
}


CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray)
{
#if DEBUG
  printf("[MOCU] cuMipmappedArrayDestroy is called.\n");
#endif
  CUresult res;

  res = mocuMipmappedArrayDestroy(hMipmappedArray);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuMipmappedArrayDestroy\n");
#endif
  }

  return res;
}


CUresult cuPointerGetAttribute(void *data,CUpointer_attribute attribute,CUdeviceptr ptr)
{
#if DEBUG
  printf("[MOCU] cuPointerGetAttribute is called.\n");
#endif
  CUresult res;

  res = mocuPointerGetAttribute(data,attribute,ptr);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuPointerGetAttribute\n");
#endif
  }

  return res;
}


CUresult cuStreamCreate(CUstream *phStream,unsigned int flags)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuStreamCreate is called.\n");
#endif

  CUresult res;
  CUstream s;
  stream* sp;
  context* cp;

  res = mocuStreamCreate(&s,flags);

  if(res == CUDA_SUCCESS){

    cp = get_current_context();
    
    sp = (stream*)malloc(sizeof(stream));
    sp->s = s;
    sp->flags = flags;
    sp->mode = 0;
    sp->next = cp->s1;
    sp->prev = cp->s1->prev;
    sp->prev->next = sp;
    sp->next->prev = sp;

    *phStream = (CUstream)sp;

    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuStreamCreate\n");
#endif
  }

  return res;

}


CUresult cuStreamCreateWithPriority(CUstream *phStream,unsigned int flags,int priority)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuStreamCreateWithPriority is called.\n");
#endif
  CUresult res;

  res = mocuStreamCreateWithPriority(phStream,flags,priority);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuStreamCreateWithPriority\n");
#endif
  }

  return res;
}


CUresult cuStreamGetPriority(CUstream hStream,int *priority)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuStreamGetPriority is called.\n");
#endif
  CUresult res;

  res = mocuStreamGetPriority(hStream,priority);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuStreamGetPriority\n");
#endif
  }

  return res;
}


CUresult cuStreamGetFlags(CUstream hStream,unsigned int *flags)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuStreamGetFlags is called.\n");
#endif
  CUresult res;

  res = mocuStreamGetFlags(hStream,flags);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuStreamGetFlags\n");
#endif
  }

  return res;
}


CUresult cuStreamWaitEvent(CUstream hStream,CUevent hEvent,unsigned int Flags)
{
#if DEBUG||D_STREAM||D_EVENT
  printf("[MOCU] cuStreamWaitEvent is called.\n");
#endif
  CUresult res;

  res = mocuStreamWaitEvent(hStream,hEvent,Flags);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuStreamWaitEvent\n");
#endif
  }

  return res;
}


CUresult cuStreamAddCallback(CUstream hStream,CUstreamCallback callback,void *userData,unsigned int flags)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuStreamAddCallback is called.\n");
#endif
  CUresult res;
  stream* sp;
  
  sp = (stream*)hStream;

  res = mocuStreamAddCallback(sp == NULL ? NULL : sp->s,callback,userData,flags);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuStreamAddCallback\n");
#endif
  }

  return res;
}


CUresult cuStreamQuery(CUstream hStream)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuStreamQuery is called.\n");
#endif
  CUresult res;
  stream* sp;

  sp = (stream*)hStream;
  if(sp == NULL) return CUDA_ERROR_INVALID_VALUE;

  res = mocuStreamQuery(sp->s);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuStreamQuery\n");
#endif
  }

  return res;

}


CUresult cuStreamSynchronize(CUstream hStream)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuStreamSynchronize is called.\n");
#endif
  CUresult res;
  stream* sp;

  sp = (stream*)hStream;
  if(sp == NULL) return CUDA_ERROR_INVALID_VALUE;

  res = mocuStreamSynchronize(sp->s);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuStreamSynchronize\n");
#endif
  }

  return res;

}


CUresult cuStreamDestroy_v2(CUstream hStream)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuStreamDestroy_v2 is called.\n");
#endif

  CUresult res;
  stream *sp;
  
  sp = (stream *)hStream;
  if(sp == NULL)return CUDA_ERROR_INVALID_VALUE;

  res = mocuStreamDestroy_v2(sp->s);

  if(res == CUDA_SUCCESS){

    sp->next->prev = sp->prev;
    sp->prev->next = sp->next;
    free(sp);

    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuStreamDestroy\n");
#endif
    return res;
  }
}


CUresult cuEventCreate(CUevent *phEvent,unsigned int flags)
{
#if DEBUG||D_EVENT
  printf("[MOCU] cuEventCreate is called.\n");
#endif
  CUresult res;
  event* ep;
  CUevent e;
  context* cp;

  res = mocuEventCreate(&e,flags);

  if(res == CUDA_SUCCESS){

    cp = get_current_context();

    ep = (event*)malloc(sizeof(event));
    ep->e = e;
    ep->mode = 0;
    ep->flags = flags;
    ep->next = cp->e1;
    ep->prev = cp->e1->prev;
    ep->prev->next = ep;
    ep->next->prev = ep;
    
    *phEvent = (CUevent)ep;

    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuEventCreate\n");
#endif
  }

  return res;

}


CUresult cuEventRecord(CUevent hEvent,CUstream hStream)
{
#if DEBUG||D_STREAM||D_EVENT
  printf("[MOCU] cuEventRecord is called.\n");
#endif
  CUresult res;
  stream* sp;
  event* ep;

  ep = (event*)hEvent;
  sp = (stream*)hStream;

  res = mocuEventRecord(ep->e,(sp == NULL) ? NULL : sp->s);

  if(res == CUDA_SUCCESS){
    ep->mode = 1;
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuEventRecord\n");
#endif
  }

  return res;

}


CUresult cuEventQuery(CUevent hEvent)
{
#if DEBUG||D_EVENT
  printf("[MOCU] cuEventQuery is called.\n");
#endif
  CUresult res;
  event *ep;

  ep = (event*)hEvent;

  res = mocuEventQuery(ep->e);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuEventQuery\n");
#endif
  }

  return res;

}


CUresult cuEventSynchronize(CUevent hEvent)
{
#if DEBUG||D_EVENT
  printf("[MOCU] cuEventSynchronize is called.\n");
#endif
  CUresult res;
  event *ep;

  ep = (event*)hEvent;

  res = mocuEventSynchronize(ep->e);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuEventSynchronize\n");
#endif
  }

  return res;

}


CUresult cuEventDestroy_v2(CUevent hEvent)
{
#if DEBUG||D_EVENT
  printf("[MOCU] cuEventDestroy_v2 is called.\n");
#endif

  CUresult res;
  event* ep;

  ep = (event*)hEvent;

  res = mocuEventDestroy_v2(ep->e);

  if(res == CUDA_SUCCESS){

    ep->next->prev = ep->prev;
    ep->prev->next = ep->next;
    free(ep);

    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuEventDestroy\n");
#endif
  }

  return res;

}


CUresult cuEventElapsedTime(float *pMilliseconds,CUevent hStart,CUevent hEnd)
{
#if DEBUG||D_EVENT
  printf("[MOCU] cuEventElapsedTime is called.\n");
#endif


  float f;
  event *ep0,*ep1;
  CUresult res;
  context* cp;

  cp = get_current_context();

  ep0 = (event*)hStart;
  ep1 = (event*)hEnd;

  if(ep0->mode == 0 || ep1->mode == 0){
    return CUDA_ERROR_NOT_READY;
  }

  switch (ep0->mode * 2 + ep1->mode) {
  case 3:
    return mocuEventElapsedTime(pMilliseconds, ep0->e, ep1->e);
  case 4:
    res = mocuEventElapsedTime(&f, cp->e0->e, ep0->e);
    if (res == CUDA_SUCCESS){
      *pMilliseconds = - ep1->charge - f;
    }
    return res;
  case 5:
    res = mocuEventElapsedTime(&f, cp->e1->e, ep1->e);
    if (res == CUDA_SUCCESS){
      *pMilliseconds = ep0->charge + f;
    }
    return res;
  case 6:
    *pMilliseconds = ep0->charge - ep1->charge;
    return CUDA_SUCCESS;
  }


  //  *pMilliseconds = 1.0;
  //  return CUDA_SUCCESS;

}


CUresult cuFuncGetAttribute(int *pi,CUfunction_attribute attrib,CUfunction hfunc)
{
#if DEBUG||D_FUNCTION
  printf("[MOCU] cuFuncGetAttribute is called.\n");
#endif
  CUresult res;

  res = mocuFuncGetAttribute(pi,attrib,hfunc);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuFuncGetAttribute\n");
#endif
  }

  return res;
}


CUresult cuFuncSetCacheConfig(CUfunction hfunc,CUfunc_cache config)
{
#if DEBUG||D_FUNCTION
  printf("[MOCU] cuFuncSetCacheConfig is called.\n");
#endif
  CUresult res;
  function *fp;

  //  res = mocuFuncSetCacheConfig(hfunc,config);
  res = mocuFuncSetCacheConfig(fp->f,config);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuFuncSetCacheConfig\n");
#endif
  }

  return res;
}


CUresult cuFuncSetSharedMemConfig(CUfunction hfunc,CUsharedconfig config)
{
#if DEBUG||D_FUNCTION
  printf("[MOCU] cuFuncSetSharedMemConfig is called.\n");
#endif
  CUresult res;

  res = mocuFuncSetSharedMemConfig(hfunc,config);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuFuncSetSharedMemConfig\n");
#endif
  }

  return res;
}

CUresult cuLaunchKernel(CUfunction f,unsigned int gridDimX,unsigned int gridDimY,unsigned int gridDimZ,unsigned int blockDimX,unsigned int blockDimY,unsigned int blockDimZ,unsigned int sharedMemBytes,CUstream hStream,void **kernelParams,void **extra)
{
#if DEBUG||D_STREAM||D_FUNCTION
  printf("[MOCU] cuLaunchKernel is called. : @device %d\n",mocuID);
#endif

  CUresult res;
  stream* sp;
  function* fp;

  sp = (stream*)hStream;
  fp = (function*)f;

  res = mocuLaunchKernel(fp->f,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,sharedMemBytes,sp == NULL ? NULL : sp->s,kernelParams,extra);

  if(res != CUDA_SUCCESS){
#if DEBUG_ERROR
    //printf("[MOCU] Error(%d) @ cuLaunchKernel\n",res);
#endif
  }
  return res;
}


CUresult cuFuncSetBlockShape(CUfunction hfunc,int x,int y,int z)
{
#if DEBUG||D_FUNCTION
  printf("[MOCU] cuFuncSetBlockShape is called.\n");
#endif
  CUresult res;

  res = mocuFuncSetBlockShape(hfunc,x,y,z);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuFuncSetBlockShape\n");
#endif
  }

  return res;
}


CUresult cuFuncSetSharedSize(CUfunction hfunc,unsigned int bytes)
{
#if DEBUG||D_FUNCTION
  printf("[MOCU] cuFuncSetSharedSize is called.\n");
#endif
  CUresult res;

  res = mocuFuncSetSharedSize(hfunc,bytes);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuFuncSetSharedSize\n");
#endif
  }

  return res;
}


CUresult cuParamSetSize(CUfunction hfunc,unsigned int numbytes)
{
#if DEBUG||D_FUNCTION
  printf("[MOCU] cuParamSetSize is called.\n");
#endif
  CUresult res;

  res = mocuParamSetSize(hfunc,numbytes);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuParamSetSize\n");
#endif
  }

  return res;
}


CUresult cuParamSeti(CUfunction hfunc,int offset,unsigned int value)
{
#if DEBUG||D_FUNCTION
  printf("[MOCU] cuParamSeti is called.\n");
#endif
  CUresult res;

  res = mocuParamSeti(hfunc,offset,value);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuParamSeti\n");
#endif
  }

  return res;
}


CUresult cuParamSetf(CUfunction hfunc,int offset,float value)
{
#if DEBUG||D_FUNCTION
  printf("[MOCU] cuParamSetf is called.\n");
#endif
  CUresult res;

  res = mocuParamSetf(hfunc,offset,value);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuParamSetf\n");
#endif
  }

  return res;
}


CUresult cuParamSetv(CUfunction hfunc,int offset,void *ptr,unsigned int numbytes)
{
#if DEBUG||D_FUNCTION
  printf("[MOCU] cuParamSetv is called.\n");
#endif
  CUresult res;

  res = mocuParamSetv(hfunc,offset,ptr,numbytes);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuParamSetv\n");
#endif
  }

  return res;
}


CUresult cuLaunch(CUfunction f)
{
#if DEBUG||D_FUNCTION
  printf("[MOCU] cuLaunch is called.\n");
#endif
  CUresult res;

  res = mocuLaunch(f);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuLaunch\n");
#endif
  }

  return res;
}


CUresult cuLaunchGrid(CUfunction f,int grid_width,int grid_height)
{
#if DEBUG||D_FUNCTION
  printf("[MOCU] cuLaunchGrid is called.\n");
#endif
  CUresult res;

  res = mocuLaunchGrid(f,grid_width,grid_height);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuLaunchGrid\n");
#endif
  }

  return res;
}


CUresult cuLaunchGridAsync(CUfunction f,int grid_width,int grid_height,CUstream hStream)
{
#if DEBUG||D_STREAM||D_FUNCTION
  printf("[MOCU] cuLaunchGridAsync is called.\n");
#endif
  CUresult res;

  res = mocuLaunchGridAsync(f,grid_width,grid_height,hStream);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuLaunchGridAsync\n");
#endif
  }

  return res;
}


CUresult cuParamSetTexRef(CUfunction hfunc,int texunit,CUtexref hTexRef)
{
#if DEBUG||D_TEXREF||D_FUNCTION
  printf("[MOCU] cuParamSetTexRef is called.\n");
#endif
  CUresult res;

  res = mocuParamSetTexRef(hfunc,texunit,hTexRef);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuParamSetTexRef\n");
#endif
  }

  return res;
}


CUresult cuTexRefSetArray(CUtexref hTexRef,CUarray hArray,unsigned int Flags)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefSetArray is called.\n");
#endif
  CUresult res;

  res = mocuTexRefSetArray(hTexRef,hArray,Flags);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefSetArray\n");
#endif
  }

  return res;
}


CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef,CUmipmappedArray hMipmappedArray,unsigned int Flags)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefSetMipmappedArray is called.\n");
#endif
  CUresult res;

  res = mocuTexRefSetMipmappedArray(hTexRef,hMipmappedArray,Flags);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefSetMipmappedArray\n");
#endif
  }

  return res;
}


CUresult cuTexRefSetAddress_v2(size_t *ByteOffset,CUtexref hTexRef,CUdeviceptr dptr,size_t bytes)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefSetAddress_v2 is called.\n");
#endif
  CUresult res;
  texref* tp;

  tp = (texref*)hTexRef;

  res = mocuTexRefSetAddress_v2(ByteOffset,tp->t,dptr,bytes);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefSetAddress_v2\n");
#endif
  }

  return res;
}


CUresult cuTexRefSetAddress2D_v2(CUtexref hTexRef,const CUDA_ARRAY_DESCRIPTOR *desc,CUdeviceptr dptr,size_t Pitch)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefSetAddress2D_v2 is called.\n");
#endif
  CUresult res;

  res = mocuTexRefSetAddress2D_v2(hTexRef,desc,dptr,Pitch);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefSetAddress2D_v2\n");
#endif
  }

  return res;
}


CUresult cuTexRefSetFormat(CUtexref hTexRef,CUarray_format fmt,int NumPackedComponents)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefSetFormat is called.\n");
#endif
  CUresult res;
  texref* tp;

  tp = (texref*)hTexRef;

  res = mocuTexRefSetFormat(tp->t,fmt,NumPackedComponents);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefSetFormat\n");
#endif
  }

  return res;
}


CUresult cuTexRefSetAddressMode(CUtexref hTexRef,int dim,CUaddress_mode am)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefSetAddressMode is called.\n");
#endif
  CUresult res;
  texref* tp;

  tp = (texref*)hTexRef;

  res = mocuTexRefSetAddressMode(tp->t,dim,am);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefSetAddressMode\n");
#endif
  }

  return res;
}


CUresult cuTexRefSetFilterMode(CUtexref hTexRef,CUfilter_mode fm)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefSetFilterMode is called.\n");
#endif
  CUresult res;
  texref* tp;

  tp = (texref*)hTexRef;

  res = mocuTexRefSetFilterMode(tp->t,fm);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefSetFilterMode\n");
#endif
  }

  return res;
}


CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef,CUfilter_mode fm)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefSetMipmapFilterMode is called.\n");
#endif
  CUresult res;
  texref* tp;

  tp = (texref*)hTexRef;

  res = mocuTexRefSetMipmapFilterMode(tp->t,fm);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefSetMipmapFilterMode\n");
#endif
  }

  return res;
}


CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef,float bias)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefSetMipmapLevelBias is called.\n");
#endif
  CUresult res;
  texref* tp;

  tp = (texref*)hTexRef;

  res = mocuTexRefSetMipmapLevelBias(tp->t,bias);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefSetMipmapLevelBias\n");
#endif
  }

  return res;
}


CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef,float minMipmapLevelClamp,float maxMipmapLevelClamp)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefSetMipmapLevelClamp is called.\n");
#endif
  CUresult res;
  texref* tp;

  tp = (texref*)hTexRef;

  res = mocuTexRefSetMipmapLevelClamp(tp->t,minMipmapLevelClamp,maxMipmapLevelClamp);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefSetMipmapLevelClamp\n");
#endif
  }

  return res;
}


CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef,unsigned int maxAniso)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefSetMaxAnisotropy is called.\n");
#endif
  CUresult res;
  texref* tp;

  tp = (texref*)hTexRef;

  res = mocuTexRefSetMaxAnisotropy(tp->t,maxAniso);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefSetMaxAnisotropy\n");
#endif
  }

  return res;
}


CUresult cuTexRefSetFlags(CUtexref hTexRef,unsigned int Flags)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefSetFlags is called.\n");
#endif
  CUresult res;
  texref* tp;

  tp = (texref*)hTexRef;

  res = mocuTexRefSetFlags(tp->t,Flags);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefSetFlags\n");
#endif
  }

  return res;
}


CUresult cuTexRefGetAddress_v2(CUdeviceptr *pdptr,CUtexref hTexRef)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefGetAddress_v2 is called.\n");
#endif
  CUresult res;
  texref* tp;

  res = mocuTexRefGetAddress_v2(pdptr,tp->t);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefGetAddress_v2\n");
#endif
  }

  return res;
}


CUresult cuTexRefGetArray(CUarray *phArray,CUtexref hTexRef)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefGetArray is called.\n");
#endif
  CUresult res;
  texref* tp;

  tp = (texref*)hTexRef;

  res = mocuTexRefGetArray(phArray,tp->t);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefGetArray\n");
#endif
  }

  return res;
}


CUresult cuTexRefGetMipmappedArray(CUmipmappedArray *phMipmappedArray,CUtexref hTexRef)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefGetMipmappedArray is called.\n");
#endif
  CUresult res;
  texref* tp;

  tp = (texref*)hTexRef;

  res = mocuTexRefGetMipmappedArray(phMipmappedArray,tp->t);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefGetMipmappedArray\n");
#endif
  }

  return res;
}


CUresult cuTexRefGetAddressMode(CUaddress_mode *pam,CUtexref hTexRef,int dim)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefGetAddressMode is called.\n");
#endif
  CUresult res;
  texref* tp;

  tp = (texref*)hTexRef;

  res = mocuTexRefGetAddressMode(pam,tp->t,dim);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefGetAddressMode\n");
#endif
  }

  return res;
}


CUresult cuTexRefGetFilterMode(CUfilter_mode *pfm,CUtexref hTexRef)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefGetFilterMode is called.\n");
#endif
  CUresult res;
  texref* tp;

  tp = (texref*)hTexRef;

  res = mocuTexRefGetFilterMode(pfm,tp->t);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefGetFilterMode\n");
#endif
  }

  return res;
}


CUresult cuTexRefGetFormat(CUarray_format *pFormat,int *pNumChannels,CUtexref hTexRef)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefGetFormat is called.\n");
#endif
  CUresult res;
  texref* tp;

  tp = (texref*)hTexRef;

  res = mocuTexRefGetFormat(pFormat,pNumChannels,tp->t);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefGetFormat\n");
#endif
  }

  return res;
}


CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode *pfm,CUtexref hTexRef)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefGetMipmapFilterMode is called.\n");
#endif
  CUresult res;
  texref* tp;

  tp = (texref*)hTexRef;

  res = mocuTexRefGetMipmapFilterMode(pfm,tp->t);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefGetMipmapFilterMode\n");
#endif
  }

  return res;
}


CUresult cuTexRefGetMipmapLevelBias(float *pbias,CUtexref hTexRef)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefGetMipmapLevelBias is called.\n");
#endif
  CUresult res;
  texref* tp;

  tp = (texref*)hTexRef;

  res = mocuTexRefGetMipmapLevelBias(pbias,tp->t);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefGetMipmapLevelBias\n");
#endif
  }

  return res;
}


CUresult cuTexRefGetMipmapLevelClamp(float *pminMipmapLevelClamp,float *pmaxMipmapLevelClamp,CUtexref hTexRef)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefGetMipmapLevelClamp is called.\n");
#endif
  CUresult res;
  texref* tp;

  tp = (texref*)hTexRef;

  res = mocuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp,pmaxMipmapLevelClamp,tp->t);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefGetMipmapLevelClamp\n");
#endif
  }

  return res;
}


CUresult cuTexRefGetMaxAnisotropy(int *pmaxAniso,CUtexref hTexRef)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefGetMaxAnisotropy is called.\n");
#endif
  CUresult res;
  texref* tp;

  tp = (texref*)hTexRef;

  res = mocuTexRefGetMaxAnisotropy(pmaxAniso,tp->t);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefGetMaxAnisotropy\n");
#endif
  }

  return res;
}


CUresult cuTexRefGetFlags(unsigned int *pFlags,CUtexref hTexRef)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefGetFlags is called.\n");
#endif
  CUresult res;
  texref* tp;

  tp = (texref*)hTexRef;

  res = mocuTexRefGetFlags(pFlags,tp->t);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefGetFlags\n");
#endif
  }

  return res;
}


CUresult cuTexRefCreate(CUtexref *pTexRef)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefCreate is called.\n");
#endif
  CUresult res;
  texref* tp;
  context* cp;
  CUtexref p;

  res = mocuTexRefCreate(&p);

  if(res == CUDA_SUCCESS){

    cp = get_current_context();

    tp = (texref*)malloc(sizeof(texref));
    tp->t = p;
    tp->mode = 1;
    tp->next = cp->t1;
    tp->prev = cp->t1->prev;
    tp->next->prev = tp;
    tp->prev->next = tp;

    *pTexRef = (CUtexref)tp;

    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefCreate\n");
#endif
  }

  return res;
}


CUresult cuTexRefDestroy(CUtexref hTexRef)
{
#if DEBUG||D_TEXREF
  printf("[MOCU] cuTexRefDestroy is called.\n");
#endif
  CUresult res;
  texref* tp;

  tp = (texref*)hTexRef;

  res = mocuTexRefDestroy(tp->t);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexRefDestroy\n");
#endif
  }

  return res;
}


CUresult cuSurfRefSetArray(CUsurfref hSurfRef,CUarray hArray,unsigned int Flags)
{
#if DEBUG
  printf("[MOCU] cuSurfRefSetArray is called.\n");
#endif
  CUresult res;

  res = mocuSurfRefSetArray(hSurfRef,hArray,Flags);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuSurfRefSetArray\n");
#endif
  }

  return res;
}


CUresult cuSurfRefGetArray(CUarray *phArray,CUsurfref hSurfRef)
{
#if DEBUG
  printf("[MOCU] cuSurfRefGetArray is called.\n");
#endif
  CUresult res;

  res = mocuSurfRefGetArray(phArray,hSurfRef);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuSurfRefGetArray\n");
#endif
  }

  return res;
}


CUresult cuTexObjectCreate(CUtexObject *pTexObject,const CUDA_RESOURCE_DESC *pResDesc,const CUDA_TEXTURE_DESC *pTexDesc,const CUDA_RESOURCE_VIEW_DESC *pResViewDesc)
{
#if DEBUG
  printf("[MOCU] cuTexObjectCreate is called.\n");
#endif
  CUresult res;

  res = mocuTexObjectCreate(pTexObject,pResDesc,pTexDesc,pResViewDesc);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexObjectCreate\n");
#endif
  }

  return res;
}


CUresult cuTexObjectDestroy(CUtexObject texObject)
{
#if DEBUG
  printf("[MOCU] cuTexObjectDestroy is called.\n");
#endif
  CUresult res;

  res = mocuTexObjectDestroy(texObject);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexObjectDestroy\n");
#endif
  }

  return res;
}


CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,CUtexObject texObject)
{
#if DEBUG
  printf("[MOCU] cuTexObjectGetResourceDesc is called.\n");
#endif
  CUresult res;

  res = mocuTexObjectGetResourceDesc(pResDesc,texObject);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexObjectGetResourceDesc\n");
#endif
  }

  return res;
}


CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc,CUtexObject texObject)
{
#if DEBUG
  printf("[MOCU] cuTexObjectGetTextureDesc is called.\n");
#endif
  CUresult res;

  res = mocuTexObjectGetTextureDesc(pTexDesc,texObject);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexObjectGetTextureDesc\n");
#endif
  }

  return res;
}


CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC *pResViewDesc,CUtexObject texObject)
{
#if DEBUG
  printf("[MOCU] cuTexObjectGetResourceViewDesc is called.\n");
#endif
  CUresult res;

  res = mocuTexObjectGetResourceViewDesc(pResViewDesc,texObject);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuTexObjectGetResourceViewDesc\n");
#endif
  }

  return res;
}


CUresult cuSurfObjectCreate(CUsurfObject *pSurfObject,const CUDA_RESOURCE_DESC *pResDesc)
{
#if DEBUG
  printf("[MOCU] cuSurfObjectCreate is called.\n");
#endif
  CUresult res;

  res = mocuSurfObjectCreate(pSurfObject,pResDesc);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuSurfObjectCreate\n");
#endif
  }

  return res;
}


CUresult cuSurfObjectDestroy(CUsurfObject surfObject)
{
#if DEBUG
  printf("[MOCU] cuSurfObjectDestroy is called.\n");
#endif
  CUresult res;

  res = mocuSurfObjectDestroy(surfObject);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuSurfObjectDestroy\n");
#endif
  }

  return res;
}


CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,CUsurfObject surfObject)
{
#if DEBUG
  printf("[MOCU] cuSurfObjectGetResourceDesc is called.\n");
#endif
  CUresult res;

  res = mocuSurfObjectGetResourceDesc(pResDesc,surfObject);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuSurfObjectGetResourceDesc\n");
#endif
  }

  return res;
}


CUresult cuDeviceCanAccessPeer(int *canAccessPeer,CUdevice dev,CUdevice peerDev)
{
#if DEBUG
  printf("[MOCU] cuDeviceCanAccessPeer is called.\n");
#endif
  CUresult res;

  res = mocuDeviceCanAccessPeer(canAccessPeer,dev,peerDev);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuDeviceCanAccessPeer\n");
#endif
  }

  return res;
}


CUresult cuCtxEnablePeerAccess(CUcontext peerContext,unsigned int Flags)
{
#if DEBUG||D_CONTEXT
  printf("[MOCU] cuCtxEnablePeerAccess is called.\n");
#endif
  CUresult res;

  res = mocuCtxEnablePeerAccess(peerContext,Flags);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuCtxEnablePeerAccess\n");
#endif
  }

  return res;
}


CUresult cuCtxDisablePeerAccess(CUcontext peerContext)
{
#if DEBUG||D_CONTEXT
  printf("[MOCU] cuCtxDisablePeerAccess is called.\n");
#endif
  CUresult res;

  res = mocuCtxDisablePeerAccess(peerContext);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuCtxDisablePeerAccess\n");
#endif
  }

  return res;
}


CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource)
{
#if DEBUG
  printf("[MOCU] cuGraphicsUnregisterResource is called.\n");
#endif
  CUresult res;

  res = mocuGraphicsUnregisterResource(resource);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuGraphicsUnregisterResource\n");
#endif
  }

  return res;
}


CUresult cuGraphicsSubResourceGetMappedArray(CUarray *pArray,CUgraphicsResource resource,unsigned int arrayIndex,unsigned int mipLevel)
{
#if DEBUG
  printf("[MOCU] cuGraphicsSubResourceGetMappedArray is called.\n");
#endif
  CUresult res;

  res = mocuGraphicsSubResourceGetMappedArray(pArray,resource,arrayIndex,mipLevel);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuGraphicsSubResourceGetMappedArray\n");
#endif
  }

  return res;
}


CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray *pMipmappedArray,CUgraphicsResource resource)
{
#if DEBUG
  printf("[MOCU] cuGraphicsResourceGetMappedMipmappedArray is called.\n");
#endif
  CUresult res;

  res = mocuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray,resource);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuGraphicsResourceGetMappedMipmappedArray\n");
#endif
  }

  return res;
}


CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr *pDevPtr,size_t *pSize,CUgraphicsResource resource)
{
#if DEBUG
  printf("[MOCU] cuGraphicsResourceGetMappedPointer_v2 is called.\n");
#endif
  CUresult res;

  res = mocuGraphicsResourceGetMappedPointer_v2(pDevPtr,pSize,resource);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuGraphicsResourceGetMappedPointer_v2\n");
#endif
  }

  return res;
}


CUresult cuGraphicsResourceSetMapFlags(CUgraphicsResource resource,unsigned int flags)
{
#if DEBUG
  printf("[MOCU] cuGraphicsResourceSetMapFlags is called.\n");
#endif
  CUresult res;

  res = mocuGraphicsResourceSetMapFlags(resource,flags);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuGraphicsResourceSetMapFlags\n");
#endif
  }

  return res;
}


CUresult cuGraphicsMapResources(unsigned int count,CUgraphicsResource *resources,CUstream hStream)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuGraphicsMapResources is called.\n");
#endif
  CUresult res;

  res = mocuGraphicsMapResources(count,resources,hStream);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuGraphicsMapResources\n");
#endif
  }

  return res;
}


CUresult cuGraphicsUnmapResources(unsigned int count,CUgraphicsResource *resources,CUstream hStream)
{
#if DEBUG||D_STREAM
  printf("[MOCU] cuGraphicsUnmapResources is called.\n");
#endif
  CUresult res;

  res = mocuGraphicsUnmapResources(count,resources,hStream);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuGraphicsUnmapResources\n");
#endif
  }

  return res;
}


CUresult cuGetExportTable(const void **ppExportTable,const CUuuid *pExportTableId)
{
#if DEBUG
  printf("[MOCU] cuGetExportTable is called\n");
#endif
  CUresult res;

  res = mocuGetExportTable(ppExportTable,pExportTableId);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuGetExportTable\n");
#endif
  }

  return res;
}

#if 1

CUresult cuLinkCreate(unsigned int numOptions, CUjit_option *options, void **optionValues, CUlinkState *stateOut){
#if DEBUG
  printf("[MOCU] cuLinkCreate is called.\n");
#endif
  CUresult res;

  res = mocuLinkCreate(numOptions,options,optionValues,stateOut);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuLinkCreate\n");
#endif
  }

  return res;
}

CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name, unsigned int numOptions, CUjit_option *options, void **optionValues){
#if DEBUG
  printf("[MOCU] cuLinkAddData is called.\n");
#endif
  CUresult res;

  res = mocuLinkAddData(state,type,data,size,name,numOptions,options,optionValues);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuLinkAddData\n");
#endif
  }

  return res;
}

CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type, const char *path, unsigned int numOptions, CUjit_option *options, void **optionValues){
#if DEBUG
  printf("[MOCU] cuLinkAddFile is called.\n");
#endif
  CUresult res;

  res = mocuLinkAddFile(state,type,path,numOptions,options,optionValues);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuLinkAddFile\n");
#endif
  }

  return res;
}

CUresult cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut){
#if DEBUG
  printf("[MOCU] cuLinkComplete is called.\n");
#endif
  CUresult res;

  res = mocuLinkComplete(state,cubinOut,sizeOut);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuLinkComplete\n");
#endif
  }

  return res;
}

CUresult cuLinkDestroy (CUlinkState state){
#if DEBUG
  printf("[MOCU] cuLinkDestroy is called.\n");
#endif
  CUresult res;

  res = mocuLinkDestroy(state);

  if(res == CUDA_SUCCESS){
    return res;
  }else{
#if DEBUG_ERROR
    printf("[MOCU] Error @ cuLinkDestroy\n");
#endif
  }

  return res;
}

#endif

#undef cuCtxCreate
CUresult cuCtxCreate(CUcontext *pctx, unsigned int flag, CUdevice dev)
{
  printf("********************************\n");
  exit(1);

  return CUDA_SUCCESS;
}


/**
   ---------- Backup Part -------------
**/

void mocu_event_update(context* cp){

#if DEBUG_BACKUP
  printf("|  start events update.                                    |\n");
#endif

  event* ep;
  float f;

  mocuEventRecord(cp->e1->e,0);
  mocuEventSynchronize(cp->e1->e);
  
  ep = cp->e0->next;
  int counter = 0;

  while(ep->mode >= 0){
    switch(ep->mode){
    case 0:
      break;
    case 1:
      mocuEventElapsedTime(&ep->charge, ep->e, cp->e1->e);
      ep->mode = 2;
      break;
    case 2:
      mocuEventElapsedTime(&f,cp->e0->e,cp->e1->e);
      ep->charge += f;
      break;
    }
    ep = ep->next;
  }

#if DEBUG_BACKUP
  printf("|  Finish ...                                              |\n");
  printf("+----------------------------------------------------------+\n");
#endif

}

void mocu_device_backup(context* cp){

#if DEBUG_BACKUP
  printf("|  backup for Device start.                                |\n");
#endif

  module *mp;
  symbol *sp;
  region *r;
  r = cp->d0->next;
  while (r->mode >= 0) {
    r->backup = (char *)malloc(r->size);
#if DEBUG_BACKUP
    printf("|  [MOCU -- BACKUP] allocate %10ld KB.                |\n", r->size >> 10);
#endif
    if (r->backup == NULL) {
      printf("|  Failed to allocate memory for backup.\n");
      exit(0);
    }
    CUresult res1,res2;

    res1 = mocuMemcpyDtoH_v2(r->backup, r->base, r->size);
    res2 = mocuMemFree_v2(r->base);

    if(res1 != CUDA_SUCCESS || res2 != CUDA_SUCCESS){
      printf("Failed to backup and free memory phase ...\n");
      exit(1);
    }
    r = r->next;
  }

#if DEBUG_BACKUP
  printf("|  Finish ...                                              |\n");
  printf("+----------------------------------------------------------+\n");
#endif
  /*
    mp = cp->m0->next;
    while (mp->mode >= 0) {
    sp = mp->s0->next;
    while (sp->mode >= 0) {
    if (sp->type == SYMBOL_DEVICE ||
    sp->type == SYMBOL_CONST) {
    sp->backup = (char *)malloc(sp->size);
    if (sp->backup == NULL) {
    printf("Failed to allocate memory for backup.\n");
    exit(0);
    }
    //			printf("Backup symbol %d MB.\n", sp->size);
    nvcr.cuMemcpyDtoH(sp->backup, sp->addr, sp->size);
    nvcr.cuMemFree(sp->addr);
    }
    sp = sp->next;
    }
    mp = mp->next;
    }
  */
}

/**
   ----- texref.c
**/

void mocu_texref_restore(texref* tp,int mode){
#if DEBUG_RESTORE
  printf("[MOCU -- RESTORE] mocu_texref_restore\n");
#endif
  CUresult res;

  if (tp->m->mode == 0) return;

  if (mode == 0) {
    res = mocuModuleGetTexRef(&tp->t, tp->m->m, tp->name);
    if (res != CUDA_SUCCESS) {
      printf("Failed to restore TexRef. res=%d\n", res);
      exit(0);
    }
    return;
  }
		
  if (tp->type & TEXTURE_SET_ARRAY)
    mocuTexRefSetArray(tp->t, tp->a->a,CU_TRSA_OVERRIDE_FORMAT);
  if (tp->type & TEXTURE_SET_ADDRESS)
    mocuTexRefSetAddress_v2(NULL, tp->t, tp->ptr, tp->bytes);
  if (tp->type & TEXTURE_SET_ADDRESS2D)
    mocuTexRefSetAddress2D_v2(tp->t, &tp->desc, tp->ptr,tp->Pitch);
  if (tp->type & TEXTURE_SET_FORMAT)
    mocuTexRefSetFormat(tp->t, tp->fmt,tp->NumPackedComponents);
  if (tp->type & TEXTURE_SET_ADDRESS_MODE)
    mocuTexRefSetAddressMode(tp->t, tp->dim, tp->am);
  if (tp->type & TEXTURE_SET_FILTER_MODE)
    mocuTexRefSetFilterMode(tp->t, tp->fm);
  if (tp->type & TEXTURE_SET_FLAGS)
    mocuTexRefSetFlags(tp->t, tp->Flags);
}

void recreate_context(context* cp){

#if DEBUG_RESTORE
  printf("| [MOCU -- RESTORE] recreate_context                       |\n");
#endif

  //  CUdevice dev;
  unsigned int flags = 0;
  CUresult res1;//,res2;

  /*

  res1 = mocuDeviceGet(&dev,mocuID);
  res2 = mocuCtxCreate_v2(&cp->ctx,flags,dev);

  if(res1 != CUDA_SUCCESS || res2 != CUDA_SUCCESS){
    printf("  Failed to recreate_context with error code (mocuDeviceGet:%d mocuCtxCreate_v2%d) \n",res1,res2);
    exit(1);
  }

  */

  res1 = mocuCtxCreate_v2(&cp->ctx,flags,mocu.dev[mocuID]);

  if(res1 != CUDA_SUCCESS){
    printf("  Failed to recreate_context\n");
    exit(1);
  }

}

/**
   ----- device.c
**/

void mocu_replay_mem_alloc(apilog* a){
#if DEBUG_MIG
  printf("| [MOCU -- APILOG] mocu_replay_mem_alloc                   |\n");
#endif
  CUresult res;
  CUdeviceptr ptr;
  res = mocuMemAlloc_v2(&ptr, a->data.memAlloc.size);
  if (res != CUDA_SUCCESS || ptr != a->data.memAlloc.addr) {
    printf("Replay failed: in cuMemAlloc.(%p==%p\?\?\? : %d) (res == %d)\n",ptr,a->data.memAlloc.addr,a->data.memAlloc.size,res);
    exit(0);
  }
}

void mocu_replay_mem_free(apilog* a){
#if DEBUG_MIG
  printf("[MOCU -- APILOG] mocu_replay_mem_free\n");
#endif
  CUresult res;

  res = mocuMemFree_v2(a->data.memFree.addr);
  if (res != CUDA_SUCCESS) {
    printf("Replay failed: in cuMemFree.\n");
    exit(0);
  }
}

void mocu_replay_mem_alloc_pitch(apilog* a){
#if DEBUG_MIG
  printf("[MOCU -- APILOG] mocu_replay_mem_alloc_pitch\n");
#endif
  CUresult res;
  CUdeviceptr ptr;
  size_t Pitch;
  res = mocuMemAllocPitch_v2(&ptr, &Pitch,
			     a->data.memAllocPitch.WidthInBytes,
			     a->data.memAllocPitch.Height,
			     a->data.memAllocPitch.ElementSizeBytes);
  if (res != CUDA_SUCCESS || ptr != a->data.memAllocPitch.ptr || a->data.memAllocPitch.Pitch != Pitch) {
    printf("Replay failed: in cuMemAllocPitch.\n");
    exit(0);
  }
}

void mocu_device_restore(context* cp){
#if DEBUG_RESTORE
  printf("| [MOCU -- RESTORE] mocu_device_restore                    |\n");
#endif

  CUresult res;
  module* mp;
  symbol* sp;
  function* fp;
  texref* tp;
  region* r;

  r = cp->d0->next;
  while (r->mode >= 0) {

    res = mocuMemcpyHtoD_v2(r->base, r->backup, r->size);

#if DEBUG_RESTORE
    printf("| [MOCU -- RESTORE] Restored device memory %10ld KB.  |\n", r->size >> 10);
#endif

    if(res == CUDA_SUCCESS){
      free(r->backup);
      r = r->next; 
    }else{
      printf("Failed to restore. (ERROR CODE : %d)\n",res);
      printf("MOCU closed.\n");
      exit(0);
    }
  }

  tp = cp->t0->next;
  while (tp->mode >= 0) {
    mocu_texref_restore(tp, 0);
    tp = tp->next;
  }

  mp = cp->m0->next;
  while (mp->mode >= 0) {
    if (mp->mode == 0) {
      mp = mp->next;
      continue;
    }

    /*
      sp = mp->s0->next;
      while (sp->mode >= 0) {
      if (sp->type == SYMBOL_DEVICE ||
      sp->type == SYMBOL_CONST) {
      //				printf("Restored symbol %d B.\n", sp->size);
      nvcr.cuMemcpyHtoD(sp->addr, sp->backup,
      sp->size);
      free(sp->backup);
      }
      sp = sp->next;
      }
    */
    
    
//    fp = mp->f0->next;
//    while (fp->mode >= 0) {
//      printf("shinkou 4\n");
//      //(CUfunction *,CUmodule ,const char *);
//      printf("INFO CUfunction fp->f %p\n",fp->f);
//      printf("     CUmodule   mp->m %p\n",mp->m);
//      printf("     const char name  %s\n",fp->name);
//      mocuModuleGetFunction(&fp->f, mp->m, fp->name);
//      printf("shinkou 5\n");
//      if (fp->cache != CU_FUNC_CACHE_PREFER_NONE){
// 	mocuFuncSetCacheConfig(fp->f, fp->cache);
//      }
//      printf("shinkou 6\n");
//      /*
//      tp = fp->t0->next;
//      while (tp->mode >= 0) {
// 	tp->t = tp->master->t;
// 	mocu_texref_restore(tp, 1);
// 	mocuParamSetTexRef(fp->f, tp->texunit, tp->t);
// 	tp = tp->next;
//      }
//      */
// 					
//      fp = fp->next;
//    }
 
    mp = mp->next;

  }

  tp = cp->t0->next;
  while (tp->mode >= 0) {
    mocu_texref_restore(tp, 2);
    tp = tp->next;
  }

#if DEBUG_RESTORE
  printf("+----------------------------------------------------------+\n");
#endif

}

/**
   ----- event.c
**/

void mocu_event_restore(context* cp){
#if DEBUG_RESTORE
  printf("| [MOCU -- RESTORE] mocu_event_restore                     |\n");
#endif

  event *ep;
  CUresult res;

  mocuEventCreate(&cp->e0->e, 0);
  mocuEventCreate(&cp->e1->e, 0);
  mocuEventRecord(cp->e1->e,0);

  ep = cp->e0->next;

  while (ep->mode >= 0) {
    res = mocuEventCreate(&ep->e, ep->flags);
    if (res != CUDA_SUCCESS) {
      printf("Failed to re create event.\n");
      exit(0);
    }
    ep = ep->next;
  }

#if DEBUG_RESTORE
  printf("+----------------------------------------------------------+\n");
#endif

}

/**
   ----- module.c
**/

void mocu_replay_module_load(apilog* a){
#if DEBUG_MIG
  printf("[MOCU -- APILOG] mocu_replay_module_load\n");
#endif
  CUresult res;
  module *mp;

  mp = a->data.moduleLoad.mod;

  res = mocuModuleLoadData(&mp->m, mp->source);
  if (res != CUDA_SUCCESS) {
    printf("Replay failed: in cuModuleLoad.\n");
    exit(0);
  }
}

void mocu_replay_module_load_data(apilog* a){
#if DEBUG_MIG
  printf("[MOCU -- APILOG] mocu_replay_module_load_data\n");
#endif
  CUresult res;
  module *mp;

  mp = a->data.moduleLoadData.mod;

  res = mocuModuleLoadData(&mp->m, mp->source);
  if (res != CUDA_SUCCESS) {
    printf("Replay failed: in cuModuleLoadData.\n");
    exit(0);
  }
}

void mocu_replay_module_load_data_ex(apilog* a){
#if DEBUG_MIG
  printf("[MOCU -- APILOG] mocu_replay_module_load_data_ex\n");
#endif
  CUresult res;
  module *mp;

  mp = a->data.moduleLoadDataEx.mod;

  res = mocuModuleLoadDataEx(&mp->m, mp->source,mp->njit_option, mp->jit_option, mp->jit_option_value);
  if (res != CUDA_SUCCESS) {
    printf("Replay failed: in cuModuleLoadDataEx.\n");
    exit(0);
  }
}

void mocu_replay_module_load_fat_binary(apilog* a){
#if DEBUG_MIG
  printf("[MOCU -- APILOG] mocu_replay_module_load_fat_binary\n");
#endif
  CUresult res;
  module *mp;

  mp = a->data.moduleLoadFatBinary.mod;

  res = mocuModuleLoadFatBinary(&mp->m, mp->source);
  if (res != CUDA_SUCCESS) {
    printf("Replay failed: in cuModuleLoadFatBinary.\n");
    exit(0);
  }
}

void mocu_replay_module_unload(apilog* a){
#if DEBUG_MIG
  printf("[MOCU -- APILOG] mocu_replay_module_unload\n");
#endif
  CUresult res;
  module *mp;

  mp = a->data.moduleUnload.mod;

  res = mocuModuleUnload(mp->m);
  if (res != CUDA_SUCCESS) {
    printf("Replay failed: in cuModuleUnload.\n");
    exit(0);
  }
}

void mocu_apilog_free_array(apilog* a){
#if DEBUG_MIG
  printf("[MOCU -- APILOG] mocu_apilog_free_array\n");
#endif
  free(a->data.arrayCreate.a);
}

void mocu_apilog_free_array3d(apilog* a){
#if DEBUG_MIG
  printf("[MOCU -- APILOG] mocu_apilog_free_array3d\n");
#endif
  free(a->data.array3DCreate.a);
}

void mocu_apilog_free_load(context* cp,apilog* a){
#if DEBUG_MIG
  printf("[MOCU -- APILOG] mocu_apilog_free_load\n");
#endif
}

void mocu_apilog_free_unload(apilog* a){
#if DEBUG_MIG
  printf("[MOCU -- APILOG] mocu_apilog_free_unload\n");
#endif
}

/**
   ----- replay.c
**/

void replay(apilog* a){
#if DEBUG_MIG
  printf("| [MOCU -- APILOG] replay...                               |\n");
#endif
  switch (a->type) {
  case MEM_ALLOC:
    mocu_replay_mem_alloc(a);
    break;

  case MEM_FREE:
    mocu_replay_mem_free(a);
    break;

  case MEM_ALLOC_PITCH:
    mocu_replay_mem_alloc_pitch(a);
    break;

  case ARRAY_CREATE:
    //    mocu_replay_array_create(a);
    printf("[MOCU -- APILOG] This Function is NOT supported ...\n");
    exit(0);
    break;

  case ARRAY_DESTROY:
    //    mocu_replay_array_destroy(a);
    printf("[MOCU -- APILOG] This Function is NOT supported ...\n");
    exit(0);
    break;

  case ARRAY3D_CREATE:
    //    mocu_replay_array3d_create(a);
    printf("[MOCU -- APILOG] This Function is NOT supported ...\n");
    exit(0);
    break;

  case MODULE_LOAD:
    mocu_replay_module_load(a);
    break;

  case MODULE_LOAD_DATA:
    mocu_replay_module_load_data(a);
    break;

  case MODULE_LOAD_DATA_EX:
    mocu_replay_module_load_data_ex(a);
    break;

  case MODULE_LOAD_FAT_BINARY:
    mocu_replay_module_load_fat_binary(a);
    break;

  case MODULE_UNLOAD:
    mocu_replay_module_unload(a);
    break;

  default:
    printf("Unknown type %d.\n", a->type);
    exit(0);
  }

}

/**
   ----- stream.c
**/

void mocu_stream_restore(context* cp){
#if DEBUG_RESTORE
  printf("| [MOCU -- RESTORE] mocu_stream_restore                    |\n");
#endif

  CUresult res;
  stream *sp;

  sp = cp->s0->next;
  while (sp->mode >= 0) {
    res = mocuStreamCreate(&sp->s, sp->flags);
    if (res != CUDA_SUCCESS) {
      printf("Failed to re create event.\n");
      exit(0);
    }
    sp = sp->next;
  }

#if DEBUG_RESTORE
  printf("+----------------------------------------------------------+\n");
#endif

}

static float elapsed(int i, int j){
	return (float)(mocu.tv[j].tv_sec - mocu.tv[i].tv_sec)
		+ (float)(mocu.tv[j].tv_usec - mocu.tv[i].tv_usec)
		* 0.000001f;
}

void mocu_backup(){
#if DEBUG_BACKUP
  printf("\n");
  printf("+----------------------------------------------------------+\n");
  printf("|> BACKUP START                                            |\n");
  printf("+==========================================================+\n");
#endif
  
  gettimeofday(&mocu.tv[0],NULL);

  CUresult res;
  context *cp;
  CUcontext ctx;

  cp = mocu.c0->next;
  while (cp->mode >= 0) {

    res = mocuCtxSynchronize();

    mocu_event_update(cp);
    mocu_device_backup(cp);

    res = mocuCtxDestroy_v2(cp->ctx);

    if(res != CUDA_SUCCESS){
      printf("|  [MOCU] Failed...(ERROR CODE : %d)\n",res);
      exit(1);
    }

    cp = cp->next;

  }

  gettimeofday(&mocu.tv[1],NULL);

#if DEBUG_BACKUP
  printf("|  Finish ...                                              |\n");
  printf("+----------------------------------------------------------+\n");
#endif
}

void mocu_migrate(int devID){

#if DEBUG_MIG
  printf("+----------------------------------------------------------+\n");
  printf("|> Start Migration (to Device %2d)                          |\n",devID);
  printf("+==========================================================+\n");
#endif

  if(devID >= mocu.ndev){
    printf("[MOCU] Invalid device ID\n");
    exit(1);
  }
  
  context* cp;
  apilog* a;

  mocuID = devID;

  cp = mocu.c0->next;
  
  while(cp->next != NULL){

    recreate_context(cp);

    a = cp->alog->next;
    while(a != NULL){
      replay(a);
      a = a->next;
    }

#if DEBUG_MIG
    printf("+----------------------------------------------------------+\n");
#endif

    mocu_event_restore(cp);
    mocu_stream_restore(cp);
    mocu_device_restore(cp);

    cp = cp->next;
  }

  gettimeofday(&mocu.tv[2],NULL);

#if DEBUG_MIG
  printf("| Time Result                                              |\n");
  printf("| Backup phase    : %10f[sec].                       |\n",elapsed(0,1));
  printf("| Migration phase : %10f[sec].                       |\n",elapsed(1,2));
  printf("| Migrate done.                                            |\n");
  printf("+----------------------------------------------------------+\n");
#endif
}

/*
void mocu_migrate_to_optimum_pos(){
  size_t memSize;
  nvmlMemory_t mem_info;
  nvmlReturn_t _res;
  unsigned long long freeMem;

  memSize = check_memory_amount_used();

  int optimum_pos = _get_optimum_device_pos();
    
  int i;
  for(i = optimum_pos ; i < mocu.ndev + optimum_pos ; i ++){      

    if(i%mocu.ndev == mocuID)continue;

    lock_other_proc();

    _res = nvmlDeviceGetMemoryInfo(mocu.nvml_dev[i%mocu.ndev],&mem_info);

    if(_res != NVML_SUCCESS){
      printf("Failed to get memory information ... @device%d\n",i);
    }

    freeMem = mem_info.free;

#if DEBUG_MIG
    printf("+--------------------------------------------+\n");
    printf("|Device %2d                                   |\n",i%mocu.ndev);
    printf("+============================================+\n");
    printf("| Free  memory region %10lld[byte]       |\n",freeMem);
    printf("| Used  memory region %10lld[byte]       |\n",mem_info.used);
    printf("| Total memory region %10lld[byte]       |\n",mem_info.total);
    printf("+--------------------------------------------+\n");
#endif

    if(freeMem > memSize + 64*1024*1024){

      mocu_backup();

      mocu_migrate(i%mocu.ndev);


    }else if(i == mocu.ndev + optimum_pos - 1){
      printf("+---------------------------+\n");
      printf("|       *  Warning  *       |\n");
      printf("+---------------------------+\n");
      printf("| There is no enough region |\n");
      printf("|   This Process will exit  |\n");
      printf("+---------------------------+\n");
    }
    unlock_other_proc();
  }
}
*/
