
CC     = gcc
CFLAGS = -g -O2 -I../include
LIB    = -lnvidia-ml

OBJS = libcuda.o

SHARED  = libcuda.so.319.76 #depends on driver version
SHARED1 = libcuda.so.1
SHARED2 = libcuda.so


all : $(SHARED)

$(SHARED) : $(OBJS)
	$(CC) $(CFLAGS) $(LIB) -g -shared -o $@ $(OBJS) -L/usr/local/lib64 -fPIC -lpthread -ldl -lstdc++
	mv $(SHARED) ../lib64

.c.o:
	$(CC) $(CFLAGS) $(LIB) -g -c $*.c -fPIC

clean : 
	rm $(SHARED) $(SHARED1) $(SHARED2)

link:
	ln -s ../lib64/$(SHARED) ../lib64/$(SHARED1)
	ln -s ../lib64/$(SHARED) ../lib64/$(SHARED2)
