all: Train Test
Train: Train.cpp
	g++ Train.cpp -o Train -O2 -lpthread
Test: Test.cpp
	g++ Test.cpp -o Test -O2 -lpthread
clean:
	rm -rf Train Test