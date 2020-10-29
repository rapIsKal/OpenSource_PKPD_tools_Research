#include "Benchmark.hpp"

int main(int argc, char *argv[]) {
	if( parseBenchmarkOptions(&argc, argv) )
		return EXIT_SUCCESS;

	runAllBenchmarks();
	printBenchmarkResults();
	return EXIT_SUCCESS;
}
