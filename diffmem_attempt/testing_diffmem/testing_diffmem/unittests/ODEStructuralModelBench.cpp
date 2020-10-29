/*
 * Copyright (c) 2016, Hasselt University
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the <organization> nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <Benchmark.hpp>
#include "models/Dummy.hpp"
#include "models/Fitzhugh.hpp"
#include "ODEStructuralModel.hpp"
#include "GaussianLikelihoodModel.hpp"

BENCHMARK(ODEStructuralModelBench, dummy, n) {
	BenchmarkSuspender braces;
	ODEStructuralModel<Dummy> sm;

	VecNd<Dummy::numberOfParameters()> parms;
	parms << 0.5, 2.3;

	Vecd t(8);
	t << 0.3, 9.9, 19.9, 39.9, 79.9, 99.9, 139.9, 179.9;

	Matrixd y(Dummy::numberOfObservations(), 8);
	y << 1.45403, 23.3428, 46.0651, 92.6142, 185.67, 229.462, 321.54, 413.857;

	Matrixd u(Dummy::numberOfObservations(), 8);

	Measurements m(t, y);
	GaussianLikelihoodModel ll{m};

	VecNd<1> sigma;
	sigma << 0.01;
	VecNd<1> tau;

	ll.transtau(sigma, tau);

	braces.dismiss();

	for(unsigned int i = 0; i < n; ++i) {
		double likelihood = sm.logpdf(parms, tau, t, ll);
		doNotOptimizeAway(likelihood);
	}
}

BENCHMARK(ODEStructuralModelBench, dummy_base, n) {
	BenchmarkSuspender braces;
	ODEStructuralModel<Dummy> sm;

	VecNd<Dummy::numberOfParameters()> parms;
	parms << 0.5, 2.3;

	Vecd t(8);
	t << 0.3, 9.9, 19.9, 39.9, 79.9, 99.9, 139.9, 179.9;

	Matrixd y(Dummy::numberOfObservations(), 8);
	y << 1.45403, 23.3428, 46.0651, 92.6142, 185.67, 229.462, 321.54, 413.857;

	Matrixd u(Dummy::numberOfObservations(), 8);

	Measurements m(t, y);
	GaussianLikelihoodModel ll{m};

	VecNd<1> sigma;
	sigma << 0.01;
	VecNd<1> tau;

	ll.transtau(sigma, tau);

	braces.dismiss();

	for(unsigned int i = 0; i < n; ++i) {
		double likelihood = sm.StructuralModel::logpdf(parms, tau, t, ll);
		doNotOptimizeAway(likelihood);
	}
}

BENCHMARK(ODEStructuralModelBench, fitzhugh, n) {
	BenchmarkSuspender braces;
	ODEStructuralModel<Fitzhugh> sm;
	VecNd<Fitzhugh::numberOfParameters()> parms;
	parms << .2, .2, 3, -1, 1;

	Vecd t(10);
	t << 0.000001, 0.222222, 0.444444, 0.666667, 0.888889, 1.11111, 1.33333, 1.55556, 1.77778, 2.0;

	Matrixd y_measured(Fitzhugh::numberOfObservations(), 10);
	y_measured << -1.00279, -0.726521, -0.355836, 0.372915, 1.47345, 1.99976, 2.03789, 2.00582, 1.94854, 1.90608,
			0.999867, 1.07677, 1.09734, 1.11611, 1.03624, 0.90122, 0.754829, 0.614646, 0.48111, 0.316174;

	Matrixd y(Fitzhugh::numberOfObservations(), 10);

	Measurements m(t, y_measured);
	GaussianLikelihoodModel llm{m};

	Vecd sigma(llm.numberOfSigmas());
	sigma << 1.0, 0.0, 0.0, 1.0;
	Vecd tau(llm.numberOfTaus());
	llm.transtau(sigma, tau);

	braces.dismiss();

	for(unsigned int i = 0; i < n; ++i) {
		double likelihood = sm.logpdf(parms, tau, t, llm);
		doNotOptimizeAway(likelihood);
	}
}

BENCHMARK(ODEStructuralModelBench, fitzhugh_base, n) {
	BenchmarkSuspender braces;
	ODEStructuralModel<Fitzhugh> sm;
	VecNd<Fitzhugh::numberOfParameters()> parms;
	parms << .2, .2, 3, -1, 1;

	Vecd t(10);
	t << 0.000001, 0.222222, 0.444444, 0.666667, 0.888889, 1.11111, 1.33333, 1.55556, 1.77778, 2.0;

	Matrixd y_measured(Fitzhugh::numberOfObservations(), 10);
	y_measured << -1.00279, -0.726521, -0.355836, 0.372915, 1.47345, 1.99976, 2.03789, 2.00582, 1.94854, 1.90608,
			0.999867, 1.07677, 1.09734, 1.11611, 1.03624, 0.90122, 0.754829, 0.614646, 0.48111, 0.316174;

	Matrixd y(Fitzhugh::numberOfObservations(), 10);

	Measurements m(t, y_measured);
	GaussianLikelihoodModel llm{m};

	Vecd sigma(llm.numberOfSigmas());
	sigma << 1.0, 0.0, 0.0, 1.0;
	Vecd tau(llm.numberOfTaus());
	llm.transtau(sigma, tau);

	braces.dismiss();

	for(unsigned int i = 0; i < n; ++i) {
		double likelihood = sm.StructuralModel::logpdf(parms, tau, t, llm);
		doNotOptimizeAway(likelihood);
	}
}
