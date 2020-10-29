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

#include <gtest/gtest.h>
#include "models/Dummy.hpp"
#include "models/HIVLatent.hpp"
#include "models/Fitzhugh.hpp"
#include "ode/Adjoint.hpp"
#include "ode/Sensitivity.hpp"
#include "IntegrateODE.hpp"
#include "TestCommon.hpp"
#include "GaussianLikelihoodModel.hpp"

TEST(AdjointTest, testDummy) {
	VecNd< Dummy::numberOfParameters() > parms;
	parms << 0.5, 2.3;

	Vecd t(8);
	t << 0.3, 9.9, 19.9, 39.9, 79.9, 99.9, 139.9, 179.9;

	Matrixd y_measured(Dummy::numberOfObservations(), 8);
	y_measured << 1.17666, 23.2701, 46.2708, 92.2502, 184.267, 230.265, 322.277, 414.272;

	Vecd tau(1);
	tau << 1.0;

	Matrixd y(Dummy::numberOfObservations(), 8);

	ode::ODE<Dummy> ode(parms);
	integrateProject(ode, t, parms, y);

	double df_da = (y_measured - y).sum();
	double df_db = (y_measured - y).transpose().cwiseProduct(t).sum();

	Measurements m(t, y_measured);
	GaussianLikelihoodModel llm{m};

	Vecd grad;
	ode::Adjoint<Dummy> adjoint(t, llm, tau, 0.0, 1000);
	adjoint.solve(parms, grad);

	EXPECT_NEAR(df_da, grad[0], 1e-4);
	EXPECT_NEAR(df_db, grad[1], 1e-4);
}

TEST(AdjointTest, testFitzhugh) {
	VecNd< Fitzhugh::numberOfParameters() > parms;
	parms << .2, .2, 3, -1, 1;

	Vecd t(10);
	t << 0.000001, 0.222222, 0.444444, 0.666667, 0.888889, 1.11111, 1.33333, 1.55556, 1.77778, 2.0;

	Matrixd y_measured(HIVLatent::numberOfObservations(), 10);
	y_measured << -1.00279, -0.726521, -0.355836, 0.372915, 1.47345, 1.99976, 2.03789, 2.00582, 1.94854, 1.90608,
			  0.999867, 1.07677, 1.09734, 1.11611, 1.03624, 0.90122, 0.754829, 0.614646, 0.48111, 0.316174;

	Vecd tau(4);
	tau << 1.0, 0.0, 1.0, 0.0;

	Matrixd y(Fitzhugh::numberOfObservations(), 10);

	Measurements m(t, y_measured);
	GaussianLikelihoodModel llm{m};

	ode::ODE<Fitzhugh> ode(parms);
	integrateProject(ode, t, parms, y);

	Vecd grad_adj;
	ode::Adjoint<Fitzhugh> adjoint(t, llm, tau);
	adjoint.solve(parms, grad_adj);

	Vecd grad_sens;
	ode::Sensitivity<Fitzhugh>::computeGradient(t, llm, parms, tau, grad_sens);

	AssertEqualRel(grad_adj, grad_sens, 1e-4, 1e-3);
}

TEST(AdjointTest, testHIV) {
	VecNd< HIVLatent::numberOfParameters() > parms;
	parms << 2.61, 0.0021, 0.443, 1.6e-5, 641.0, 0.0085, 0.0092, 0.289, 30.0, 0.99, 0.9;

	Vecd t(8);
	t << 0.3, 9.9, 19.9, 39.9, 79.9, 99.9, 139.9, 179.9;

	Matrixd y_measured(HIVLatent::numberOfObservations(), 8);
	y_measured << 4.87889, 3.66814,    2.58389,    2.17221,    2.11244,    1.9438,    1.74483,    1.47871,
			 168.632,    175.51,     185.281,    202.68 ,    230.908,    241.99,    259.43,     272.34;

	Vecd tau(4);
	tau << 1.0, 0.0, 1.0, 0.0;

	Measurements m(t, y_measured);
	GaussianLikelihoodModel llm{m};

	Matrixd y(HIVLatent::numberOfObservations(), 8);

	ode::ODE<HIVLatent> ode(parms);
	integrateProject(ode, t, parms, y);

	Vecd grad_adj;
	ode::Adjoint<HIVLatent> adjoint(t, llm, tau, 0.0, 10000);
	adjoint.solve(parms, grad_adj);
	Vecd grad_sens;
	ode::Sensitivity<HIVLatent>::computeGradient(t, llm, parms, tau, grad_sens);
	AssertEqualRel(grad_adj, grad_sens, 1e-4, 1e-3);
}
