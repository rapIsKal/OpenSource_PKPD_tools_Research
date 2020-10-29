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
#include "GaussianLikelihoodModel.hpp"
#include "LogGaussianLikelihoodModel.hpp"
#include "TestCommon.hpp"

TEST(GaussianLikelihoodModelTest, testSigmaTau) {
	Vecd t(8);
	t << 0.3, 9.9, 19.9, 39.9, 79.9, 99.9, 139.9, 179.9;

	Matrixd y(1, 8);
	y << 1.45403, 23.3428, 46.0651, 92.6142, 185.67, 229.462, 321.54, 413.857;

	Matrixd u(1, 8);
	u << 1, 23, 46, 92, 185, 229, 321, 413;

	Measurements m(t, y);
	GaussianLikelihoodModel ll{m};

	Vecd sigma(ll.numberOfSigmas());
	sigma << 0.01;
	Vecd tau(ll.numberOfTaus());
	ll.transtau(sigma, tau);

	ll.transsigma(tau, sigma);
	EXPECT_SIMILAR(sigma[0], 0.01, 1e-10);
}

TEST(GaussianLikelihoodModelTest, testOneByOne) {
	Vecd t(8);
	t << 0.3, 9.9, 19.9, 39.9, 79.9, 99.9, 139.9, 179.9;

	Matrixd y(1, 8);
	y << 1.45403, 23.3428, 46.0651, 92.6142, 185.67, 229.462, 321.54, 413.857;

	Matrixd u(1, 8);
	u << 1, 23, 46, 92, 185, 229, 321, 413;

	Measurements m(t, y);
	GaussianLikelihoodModel ll{m};

	Vecd sigma(ll.numberOfSigmas());
	sigma << 0.1;
	Vecd tau(ll.numberOfTaus());
	ll.transtau(sigma, tau);

	double sum = ll.logProportions(tau);
	for(int i = 0; i < m.size(); ++i) {
		const double proportional = ll.logProportionalPdf(tau, i, u.col(i));
		sum += proportional;
	}

	double full = ll.logpdf(tau, u);
	EXPECT_SIMILAR(full, sum, 1e-10);

	double base = ll.PointwiseLikelihoodModel::logpdf(tau, u);
	EXPECT_SIMILAR(full, base, 1e-10);
}

TEST(GaussianLikelihoodModelTest, testProportional) {
	Vecd t(8);
	t << 0.3, 9.9, 19.9, 39.9, 79.9, 99.9, 139.9, 179.9;

	Matrixd y(1, 8);
	y << 1.45403, 23.3428, 46.0651, 92.6142, 185.67, 229.462, 321.54, 413.857;

	Matrixd u(1, 8);
	u << 1, 23, 46, 92, 185, 229, 321, 413;

	Measurements m(t, y);
	GaussianLikelihoodModel ll{m};

	Vecd sigma(ll.numberOfSigmas());
	sigma << 0.1;
	Vecd tau(ll.numberOfTaus());
	ll.transtau(sigma, tau);

	double full = ll.logpdf(tau, u);
	double constants = ll.logProportions(tau);
	double proportional = 0.0;
	for(int i = 0; i < m.size(); ++i) {
		proportional += ll.logProportionalPdf(tau, i, u.col(i));
	}

	EXPECT_SIMILAR(full, constants + proportional, 1e-10);
	EXPECT_SIMILAR(full, -10.10880654816139, 1e-10);
}

TEST(GaussianLikelihoodModelTest, testGradHess) {
	Vecd t(8);
	t << 0.3, 9.9, 19.9, 39.9, 79.9, 99.9, 139.9, 179.9;

	Matrixd y(1, 8);
	y << 1.45403, 23.3428, 46.0651, 92.6142, 185.67, 229.462, 321.54, 413.857;

	Matrixd u(1, 8);
	u << 1, 23, 46, 92, 185, 229, 321, 413;

	Measurements m(t, y);
	GaussianLikelihoodModel ll{m};

	Vecd sigma(ll.numberOfSigmas());
	sigma << 0.1;
	Vecd tau(ll.numberOfTaus());
	ll.transtau(sigma, tau);

	Vecd g(1), g2(1);
	Matrixd H(1, 1);

	for(int i = 0; i < m.size(); ++i) {
		ll.logpdfGradU(tau, i, u.col(i), g);
		ll.logpdfHessU(tau, i, u.col(i), g2, H);
		AssertEqual(g, g2, 1e-15);
	}
}

template <typename Derived>
auto log(const MatrixBase<Derived> &m) {
	return m.unaryExpr([](auto x) { return std::log(x); });
}

template <typename Derived>
auto exp(const MatrixBase<Derived> &m) {
	return m.unaryExpr([](auto x) { return std::exp(x); });
}

TEST(GaussianLikelihoodModelTest, testLog) {
	Vecd t(8);
	t << 0.3, 9.9, 19.9, 39.9, 79.9, 99.9, 139.9, 179.9;

	Matrixd y(1, 8);
	y << 1.45403, 23.3428, 46.0651, 92.6142, 185.67, 229.462, 321.54, 413.857;

	Matrixd u(1, 8);
	u << 1, 23, 46, 92, 185, 229, 321, 413;

	Measurements m_log(t, log(y));
	GaussianLikelihoodModel ll{m_log};

	Measurements m(t, y);
	LogGaussianLikelihoodModel ll_log{m};

	Matrixd log_u = log(u);

	Vecd tau = Vecd::Ones(ll.numberOfTaus());

	EXPECT_NEAR(ll.logpdf(tau, log_u), ll_log.logpdf(tau, u), 1e-8);

	Vecd gradTau(tau.size());
	Vecd gradTau_log(tau.size());

	EXPECT_NEAR(ll.logpdfGradTau(tau, log_u, gradTau), ll_log.logpdfGradTau(tau, u, gradTau_log), 1e-8);
	AssertEqual(gradTau, gradTau_log);
}
