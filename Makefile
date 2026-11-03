CXX = g++
CPPFLAGS = -I.
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -Wpedantic -fopenmp -pthread

COMMON = ivf_io.cpp ivf_search.cpp ivf_student.cpp
HEADERS = ivf_io.h ivf_search.h

.PHONY: all test clean
all: ivf_lab

ivf_lab: ivf_lab.cpp $(COMMON) $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) ivf_lab.cpp $(COMMON) -o $@

ivf_tests: tests/ivf_tests.cpp $(COMMON) $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/ivf_tests.cpp $(COMMON) -o $@

test: ivf_tests
	./ivf_tests

clean:
	rm -f ivf_lab ivf_tests
