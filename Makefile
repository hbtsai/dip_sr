EXEC=sr
CC=gcc
CXX=g++
DBG_CFLAGS= -D_DEBUG -g
CFLAGS= $(DBG_CFLAGS)
CXXFLAGS=$(DBG_CFLAGS)
OMP_LIBS=#-fopenmp
LDFLAGS=-L/usr/local/lib
LIBS= -lm -lopencv_core -lopencv_highgui -lopencv_imgproc ${OMP_LIBS}
OBJS=main.o Convolute.o Backprojection.o

all: $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $(EXEC) $(LIBS) 

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LIBS) -c -o $@ $<

clean:
	rm -f *.o $(EXEC) MyTest*
