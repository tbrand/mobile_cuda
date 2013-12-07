#include "cuda.h"
#include "nvml.h"

#define	TEXTURE_SET_ARRAY		(1<<0)
#define	TEXTURE_SET_ARRAY3D		(1<<1)
#define	TEXTURE_SET_ADDRESS		(1<<2)
#define	TEXTURE_SET_ADDRESS2D		(1<<3)
#define	TEXTURE_SET_FORMAT		(1<<4)
#define	TEXTURE_SET_ADDRESS_MODE	(1<<5)
#define	TEXTURE_SET_FILTER_MODE		(1<<6)
#define	TEXTURE_SET_FLAGS		(1<<7)

typedef struct _stream {
  struct _stream *prev, *next;
  CUstream s;
  int mode;
  unsigned int flags;
} stream;

typedef struct _event {
  struct _event *prev, *next;
  CUevent e;
  float charge;
  int mode;
  unsigned int flags;
  //	0: allocated, 1: recorded/ended, 2: charged.
} event;

typedef struct _region {
  struct _region *prev, *next;
  size_t size;
  CUdeviceptr base;
  char *addr;
  int mode;
  char *backup;
} region;

typedef struct _array {
  struct _array *prev, *next;
  int mode;
  int type;
  CUarray a;
  char *backup;
} array;

struct _module;

typedef struct _texref {
  struct _texref *prev, *next;
  struct _texref *master;
  int mode;
  CUtexref t;
  int type;
  int texunit;
  CUarray_format fmt;
  int NumPackedComponents;
  int dim;
  CUaddress_mode am;
  CUfilter_mode fm;
  unsigned int Flags;
  array *a;
  CUdeviceptr ptr;
  unsigned int ByteOffset;
  unsigned int bytes;
  unsigned int Pitch;
  CUDA_ARRAY_DESCRIPTOR desc;
  struct _module *m;
  char *name;
} texref;

typedef struct _function {
  struct _function *prev, *next;
  int mode;
  char *name;
  unsigned char param;
  int x, y, z;
  unsigned int param_size;
  unsigned int shared_size;
  CUfunc_cache cache;
  CUfunction f;
  texref *t0, *t1;
} function;

enum {
  SYMBOL_CONST = 1,
  SYMBOL_DEVICE,
  SYMBOL_TEXTURE
};

typedef struct _symbol {
  struct _symbol *prev, *next;
  int mode;
  int type;
  char *name;
  char *backup;
  CUdeviceptr addr;
  //  unsigned int size;
  size_t size;
} symbol;

typedef struct _module {
  struct _module *prev, *next;
  CUmodule m;
  int mode;
  char *source;
  int len;
  int njit_option;
  CUjit_option *jit_option;
  void **jit_option_value;
  function *f0, *f1;
  symbol *s0, *s1;
} module;

enum {
  MEM_ALLOC = 1,
  MEM_FREE,
  MEM_ALLOC_PITCH,
  ARRAY_CREATE,
  ARRAY_DESTROY,
  ARRAY3D_CREATE,
  MODULE_LOAD,
  MODULE_LOAD_DATA,
  MODULE_LOAD_DATA_EX,
  MODULE_LOAD_FAT_BINARY,
  MODULE_UNLOAD,
  MODULE_GET_FUNCTION,//TEST
  NOP
};

typedef struct {
  CUdeviceptr addr;
  unsigned int size;
} MemAlloc;

typedef struct {
  CUdeviceptr addr;
} MemFree;

typedef struct {
  CUdeviceptr ptr;
  //  unsigned int Pitch;
  size_t Pitch;
  //  unsigned int WidthInBytes;
  size_t WidthInBytes;
  //  unsigned int Height;
  size_t Height;
  unsigned int ElementSizeBytes;
} MemAllocPitch;

typedef struct {
  CUDA_ARRAY_DESCRIPTOR desc;
  array *a;
} ArrayCreate;

typedef struct {
  array *a;
} ArrayDestroy;

typedef struct {
  CUDA_ARRAY3D_DESCRIPTOR desc;
  array *a;
} Array3DCreate;

typedef struct {
  module *mod;
} ModuleLoad;

typedef struct {
  module *mod;
} ModuleLoadData;

typedef struct {
  module *mod;
} ModuleLoadDataEx;

typedef struct {
  module *mod;
} ModuleLoadFatBinary;

typedef struct {
  module *mod;
} ModuleUnload;

typedef struct {
  module *mod;
} ModuleGetFunction;

typedef struct _apilog {
  struct _apilog *prev, *next;
  unsigned int type;
  union {
    MemAlloc memAlloc;
    MemFree memFree;
    MemAllocPitch memAllocPitch;
    ArrayCreate arrayCreate;
    ArrayDestroy arrayDestroy;
    Array3DCreate array3DCreate;
    ModuleLoad moduleLoad;
    ModuleLoadData moduleLoadData;
    ModuleLoadDataEx moduleLoadDataEx;
    ModuleLoadFatBinary moduleLoadFatBinary;
    ModuleUnload moduleUnload;
    ModuleGetFunction moduleGetFucntion;
  } data;
} apilog;

typedef struct _context {
  struct _context *prev, *next;
  struct _context *stack;
  CUcontext user;
  CUcontext ctx;
  pthread_t tid;
  int ordinal;
  unsigned int flags;
  int mode;
  CUevent start, end;
  event *e0, *e1;
  stream *s0, *s1;
  region *d0, *d1;
  array *a0, *a1;
  module *m0, *m1;
  texref *t0, *t1;
  apilog *alog, *alast;
} context;

/*
typedef struct __context {
  struct __context *stack;
  CUcontext ctx;
  pthread_t tid;
  int ordinal;
  unsigned int flags;
  int mode;
  CUevent start, end;
  event *e0, *e1;
  stream *s0, *s1;
  region *d0, *d1;
  array *a0, *a1;
  module *m0, *m1;
  texref *t0, *t1;
  apilog *alog, *alast;
} context_for_t;
*/


typedef struct _thread {
  struct _thread *prev, *next;
  pthread_t tid;
  //  context *c0, *c1;
  context* ctx;
  int mode;
} thread;

typedef struct _mocu{
  CUcontext ctx;
  CUdevice *dev;
  int ndev;
  int *devid;
  void* libcuda;
  char version[64];
  thread *t0,*t1;
  context *c0,*c1;
  //  cr_spinlock_t lock;
  struct timeval tv[5];

  int use_pinned;
  int force_async;
  int forward2d;
  size_t pinned_align;
  size_t pinned_size;
  unsigned int pinned_mode;
  region *p0, *p1;

  nvmlDevice_t* nvml_dev;

} MOCU;

extern MOCU mocu;

extern void mocu_replay_mem_alloc(apilog*);
extern void mocu_replay_mem_free(apilog*);
extern void mocu_replay_mem_alloc_pitch(apilog*);
extern void mocu_device_restore(context*);
extern void mocu_event_restore(context*);
extern void mocu_replay_module_load(apilog*);
extern void mocu_replay_module_load_data(apilog*);
extern void mocu_replay_module_load_data_ex(apilog*);
extern void mocu_replay_module_load_fat_binary(apilog*);
extern void mocu_replay_module_unload(apilog*);
extern void mocu_apilog_free_array(apilog*);
extern void mocu_apilog_free_array3d(apilog*);
extern void mocu_apilog_free_load(context*,apilog*);
extern void mocu_apilog_free_unload(apilog*);
extern void replay(apilog*);
extern void mocu_stream_restore(context*);
extern void mocu_texref_restore(texref*,int);
extern void mocu_migrate(int);
extern void mocu_backup();
extern void recreate_context(context*);
extern void mocu_start_migration(int);

/*

//@ init.c
extern void _init_context();
extern void _init_region();
extern void _init_stream();
extern void _init_texref();
extern void _init_module();
extern void _init_thread();
extern void _init_apilog();
extern void _init_event();
extern void _init_env();
extern void _init_mocu();

//@ thread.c
extern thread* create_new_thread(pthread_t);
extern thread* get_now_thread(pthread_t);
extern void stack_context(context*);

//@ context.c
extern context* get_current_context();
extern void create_context(CUcontext*);

//@ api.c
extern void add_apilog(apilog*);

//@ module.c
extern int module_init(module*);
extern int module_collect_symbols(module*);
extern module* get_module_correspond_to_hmod(context*,CUmodule);

//@ region.c
extern void _unload_region(CUdeviceptr);

//@ util.c
extern void print_all_log();

*/
