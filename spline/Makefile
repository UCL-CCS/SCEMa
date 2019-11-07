MPICC=/usr/bin/mpic++

mpi_comparison_test: mpi_comparison_test.cc ../headers/strain2spline.h spline.h
	${MPICC} -std=c++11 -Wall -lm mpi_comparison_test.cc -o mpi_comparison_test

compare_all_histories: compare_all_histories.cc ../headers/strain2spline.h
	g++ -std=c++11 -Wall -lm compare_all_histories.cc -o compare_all_histories

test_strain2spline: test_strain2spline.cc ../headers/strain2spline.h
	g++ -Wall -lm test_strain2spline.cc -o test_strain2spline
