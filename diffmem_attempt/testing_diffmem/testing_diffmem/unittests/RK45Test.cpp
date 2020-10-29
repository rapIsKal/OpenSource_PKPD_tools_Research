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
#include "ode/RungeKutta45.hpp"
#include "TestCommon.hpp"

namespace {

	class SHO : public ode::ODEBase {
		public:
			SHO(double theta, bool stiff=false) : theta(theta), stiff(stiff) {}

		public:
			bool isStiff() const { return stiff; }
			int numberOfEquations() const { return 2; }
			void initial(RefVecd y0) {
				y0[0] = 1.0;
				y0[1] = 0.0;
			}

			void ddt(const double t, const RefVecd & y, RefVecd ydot) {
				ydot[0] = y[1];
				ydot[1] = -y[0] - theta * y[1];
			}

		private:
			double theta;
			bool stiff;
	};

	template <int N>
	class NCompartment : public ode::ODEBase {
		public:
			NCompartment(const Vecd & ka, const Vecd & ke) : ka(ka), ke(ke) {}

		public:
			int numberOfEquations() const { return N; }
			void initial(RefVecd y0) {
				y0[0] = 1.0;
				y0.segment(1, N-1) = Vecd::Zero(N-1);
			}

			void ddt(const double t, const RefVecd & y, RefVecd ydot) {
				ydot[0] = -ka[0] * y[0];
				ydot.segment(1, N-1) = ka.cwiseProduct(y.segment(0,N-1)) - ke.cwiseProduct(y.segment(1,N-1));
			}

		private:
			Vecd ka, ke;
	};

}

TEST(RK45Test, testDummy) {
	VecNd< Dummy::numberOfParameters() > parms;
	parms << 0.5, 2.3;

	Vecd t(8);
	t << 0.3, 9.9, 19.9, 39.9, 79.9, 99.9, 139.9, 179.9;

	ode::ODE<Dummy> ode(parms);
	ode::RungeKutta45 integrator(ode);
	integrator.setIC();

	for(int i = 0; i < 8; ++i) {
		integrator.integrateTo(t[i]);
		const double expected = parms[0] + t[i] * parms[1];
		EXPECT_NEAR(expected, integrator.currentState()[0], 1e-5);
	}
}

TEST(RK45Test, testIntegrateStiff) {
	try {
		double theta = 0.15;
		SHO ode(theta, true);
		ode::RungeKutta45 integrator(ode);
		FAIL();
	} catch( assertion_error &  ) {
		SUCCEED();
	} catch( ... ) {
		FAIL();
	}
}

TEST(RK45Test, testIntegrateNonStiff) {
	double theta = 0.15;
	SHO ode(theta, false);

	ode::RungeKutta45 integrator(ode);
	integrator.setIC();

	integrator.integrateTo(0.0);
	{
		const Vecd & s = integrator.currentState();
		EXPECT_NEAR(1.0, s[0], 1e-5);
		EXPECT_NEAR(0.0, s[1], 1e-5);
	}

	integrator.integrateTo(0.1);
	{
		const Vecd & s = integrator.currentState();
		EXPECT_NEAR(0.995029046910927, s[0], 1e-5);
		EXPECT_NEAR(-0.099088396397971, s[1], 1e-5);
	}

	integrator.integrateTo(10.0);
	{
		const Vecd & s = integrator.currentState();
		EXPECT_NEAR(-0.421909450282743, s[0], 1e-5);
		EXPECT_NEAR(0.246407890289464, s[1], 1e-5);
	}
}
