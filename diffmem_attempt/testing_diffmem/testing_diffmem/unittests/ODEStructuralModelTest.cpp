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
#include "ODEStructuralModel.hpp"
#include "GaussianLikelihoodModel.hpp"
#include "math/constants.hpp"
#include "TestCommon.hpp"

TEST(ODEStructuralModelTest, testDummy) {
	ODEStructuralModel<Dummy> sm;

	EXPECT_EQ(Dummy::numberOfParameters(), sm.numberOfParameters());
	EXPECT_EQ(Dummy::numberOfObservations(), sm.numberOfObservations());

	VecNd< Dummy::numberOfParameters() > parms;
	parms << 0.5, 2.3;

	Vecd t(8);
	t << 0.3, 9.9, 19.9, 39.9, 79.9, 99.9, 139.9, 179.9;

	Matrixd y(Dummy::numberOfObservations(), 8);
	y << 1.45403, 23.3428, 46.0651, 92.6142, 185.67, 229.462, 321.54, 413.857;

	Matrixd u(Dummy::numberOfObservations(), 8);

	sm.evalU(parms, t, u);

	for(int i = 0; i < 8; ++i) {
		const double expected = parms[0] + t[i] * parms[1];
		EXPECT_NEAR(expected, u(0,i), 1e-5);
	}

	Measurements m(t, y);
	GaussianLikelihoodModel ll{m};

	VecNd<1> sigma; sigma << 0.01;
	VecNd<1> tau;

	ll.transtau(sigma, tau);
	EXPECT_NEAR(std::sqrt(0.01), tau[0], 1e-5);

	ll.transsigma(tau, sigma);
	EXPECT_NEAR(0.01, sigma[0], 1e-5);

	double likelihood = ll.logpdf(parms, tau, sm);
	EXPECT_NEAR( 8*(-math::logSqrt2Pi() - 0.5 * std::log(sigma[0])) -0.5 * (y - u).squaredNorm() / sigma[0], likelihood, 1e-3);

	VecNd< Dummy::numberOfParameters() > grad;
	likelihood = ll.logpdf(parms, tau, sm, grad);

	double df_da = (y - u).sum() / sigma[0];
	double df_db = (y - u).transpose().cwiseProduct(t).sum() / sigma[0];

	EXPECT_NEAR(df_da, grad[0], 1e-3);
	EXPECT_NEAR(df_db, grad[1], 1e-3);
}

TEST(ODEStructuralModelTest, testDummySens) {
	ODEStructuralModel<Dummy> sm;

	EXPECT_EQ(Dummy::numberOfParameters(), sm.numberOfParameters());
	EXPECT_EQ(Dummy::numberOfObservations(), sm.numberOfObservations());

	VecNd< Dummy::numberOfParameters() > parms;
	parms << 0.5, 2.3;

	Vecd t(8);
	t << 0.3, 9.9, 19.9, 39.9, 79.9, 99.9, 139.9, 179.9;

	const int numberOfSensitivities = Dummy::numberOfObservations() * (parms.size() + 1);
	Matrixd sens_fd(numberOfSensitivities, 8);
	sm.StructuralModel::evalSens(parms, t, sens_fd);

	Matrixd sens_true(numberOfSensitivities, 8);
	sm.evalSens(parms, t, sens_true);

	AssertEqualRel(sens_true, sens_fd, 1e-4);
}

