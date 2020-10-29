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
#include "IntegrateODE.hpp"
#include "ODEStructuralModel.hpp"
#include "ode/RungeKutta45.hpp"

BENCHMARK(ODEBench, rk45, n) {
	BenchmarkSuspender braces;
	VecNd< Dummy::numberOfParameters() > parms;
	parms << 0.5, 2.3;

	Vecd t(8);
	t << 0.3, 9.9, 19.9, 39.9, 79.9, 99.9, 139.9, 179.9;

	Matrixd y(Dummy::numberOfObservations(), 8);

	ode::ODE<Dummy> ode(parms);
	braces.dismiss();

	for(unsigned int i = 0; i < n; ++i) {
		ode::RungeKutta45 integrator(ode);
		integrator.setIC();

		for(int i = 0; i < t.size(); ++i) {
			integrator.integrateTo(t[i]);
		}
	}
}

BENCHMARK(ODEBench, integrateProject, n) {
	BenchmarkSuspender braces;
	VecNd< Dummy::numberOfParameters() > parms;
	parms << 0.5, 2.3;

	Vecd t(8);
	t << 0.3, 9.9, 19.9, 39.9, 79.9, 99.9, 139.9, 179.9;

	Matrixd y(Dummy::numberOfObservations(), 8);

	ode::ODE<Dummy> ode(parms);
	braces.dismiss();

	for(unsigned int i = 0; i < n; ++i)
		integrateProject<false>(ode, t, parms, y);
}

BENCHMARK(ODEBench, integrateProjectFunctional, n) {
	BenchmarkSuspender braces;
	VecNd< Dummy::numberOfParameters() > parms;
	parms << 0.5, 2.3;

	Vecd t(8);
	t << 0.3, 9.9, 19.9, 39.9, 79.9, 99.9, 139.9, 179.9;

	Matrixd y(Dummy::numberOfObservations(), 8);

	ode::ODE<Dummy> ode(parms);
	braces.dismiss();

	for(unsigned int i = 0; i < n; ++i)
		integrateProject<true>(ode, t, parms, y);
}

BENCHMARK(ODEBench, ODEStructuralModel, n) {
	BenchmarkSuspender braces;
	ODEStructuralModel<Dummy> sm;

	VecNd< Dummy::numberOfParameters() > parms;
	parms << 0.5, 2.3;

	Vecd t(8);
	t << 0.3, 9.9, 19.9, 39.9, 79.9, 99.9, 139.9, 179.9;

	Matrixd y(Dummy::numberOfObservations(), 8);
	y << 1.45403, 23.3428, 46.0651, 92.6142, 185.67, 229.462, 321.54, 413.857;

	Matrixd u(Dummy::numberOfObservations(), 8);

	braces.dismiss();

	for(unsigned int i = 0; i < n; ++i)
		sm.evalU(parms, t, u);
}
