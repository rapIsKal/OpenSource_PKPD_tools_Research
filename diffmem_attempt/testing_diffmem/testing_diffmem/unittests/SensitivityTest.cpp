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

#include "ode/SensitivityIntegrator.hpp"
#include "models/HIVLatent.hpp"
#include "models/LinearModel.hpp"
#include "models/Locke.hpp"
#include "models/PKPD.hpp"
#include "IntegrateODE.hpp"
#include "TestCommon.hpp"

namespace {

	template <class Model>
	class SensitivityODE : public ode::ODEBase {
		public:
			typedef VecNd<Model::numberOfParameters()> Parameters;
			typedef VecNd<Model::numberOfEquations()> States;
			typedef MatrixMNd<Model::numberOfEquations(), Model::numberOfEquations()> JacSol;
			typedef MatrixMNd<Model::numberOfEquations(), Model::numberOfParameters()> JacPar;
			static constexpr int numberOfSensitivities = Model::numberOfEquations() * Model::numberOfParameters();

		public:
			template <typename PT>
			SensitivityODE(const MatrixBase<PT> & params) : parameters(params) {}

		public:
			int numberOfEquations() const { return Model::numberOfEquations() + numberOfSensitivities; }

			void ddt(const double t, const RefVecd & s, RefVecd sdot) {
				const States y = s.head(Model::numberOfEquations());

				States ydot;
				Model::ODE::ddt(t, y, parameters, ydot);

				JacSol Jy;
				Model::ODE::jacobianState(t, y, parameters, Jy);

				JacPar Jp;
				Model::ODE::jacobianParameters(t, y, parameters, Jp);

				sdot.head(Model::numberOfEquations()) = ydot;
				for(int i = 0; i < Model::numberOfParameters(); ++i) {
					const int ip = (i + 1) * Model::numberOfEquations();
					sdot.segment(ip, Model::numberOfEquations()) =
							Jp.col(i) + Jy * s.segment(ip, Model::numberOfEquations());
				}
			}

			bool hasJacobian() const { return false; }

			virtual void initial(RefVecd s0) {
				States y0;
				Model::ODE::initial(parameters, y0);

				JacPar Jp;
				Model::ODE::jacobianParametersInitial(parameters, Jp);

				s0.head(Model::numberOfEquations()) = y0;
				s0.tail(numberOfSensitivities) =
						MapVecNd<numberOfSensitivities>(Jp.data(), Jp.cols()*Jp.rows());
			}

		public:
			template <typename PT>
			void setParameters(const MatrixBase<PT> & params) { parameters = params; }

		private:
			Parameters parameters;
	};

	class SHO : public ode::SensitivityODEBase {
		public:
			SHO(double theta, bool stiff=false) : theta(theta), stiff(stiff) {}

		public:
			bool isStiff() const { return stiff; }
			int numberOfEquations() const { return 2; }
			int numberOfParameters() const { return 3; }

			void initial(RefVecd y0) {
				y0[0] = 1.0;
				y0[1] = 0.0;
			}

			void ddt(const double t, const RefVecd & y, RefVecd ydot) {
				ydot[0] = y[1];
				ydot[1] = -y[0] - theta * y[1];
			}

			void jacobian(const double t, const RefVecd & y, const RefVecd & ydot, RefMatrixd J) {
				J(0,0) = 0.0;
				J(0,1) = 1.0;
				J(1,0) = -1.0;
				J(1,1) = -theta;
			}

			void jacobianParametersInitial(RefMatrixd yS0) {
				yS0(0,0) = 0;
				yS0(1,0) = 0;
				yS0(0,1) = 1;
				yS0(1,1) = 0;
				yS0(0,2) = 0;
				yS0(1,2) = 1;
			}

			void jacobianParameters(const double t, const RefVecd & y, const RefVecd & ydot, RefMatrixd J) {
				J(0,0) = 0.0;
				J(0,1) = 0.0;
				J(0,2) = 0.0;
				J(1,0) = -y[1];
				J(1,1) = 0.0;
				J(1,2) = 0.0;
			}

		private:
			double theta;
			bool stiff;
	};
}

static void testIntegrate(bool stiff) {
	double theta = 0.15;
	SHO ode(theta, stiff);

	ode::SensitivityIntegrator integrator(ode);
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
		EXPECT_NEAR(-0.09908839639797, s[1], 1e-5);

		const Matrixd & sens = integrator.currentSensitivity();
		EXPECT_NEAR(0.000165256909940, sens(0,0), 1e-5);
		EXPECT_NEAR(0.004942025551712, sens(1,0), 1e-5);
		EXPECT_NEAR(0.995029046910927, sens(0,1), 1e-5);
		EXPECT_NEAR(-0.09908839639797, sens(1,1), 1e-5);
		EXPECT_NEAR(0.099088396397971, sens(0,2), 1e-5);
		EXPECT_NEAR(0.980165787451232, sens(1,2), 1e-5);
	}

	integrator.integrateTo(10.0);
	{
		const Vecd & s = integrator.currentState();
		EXPECT_NEAR(-0.421909450282743, s[0], 1e-5);
		EXPECT_NEAR(0.246407890289464, s[1], 1e-5);

		const Matrixd & sens = integrator.currentSensitivity();
		EXPECT_NEAR(1.904654025941569, sens(0,0), 1e-5);
		EXPECT_NEAR(-1.374888503725091, sens(1,0), 1e-5);
		EXPECT_NEAR(-0.421909450282743, sens(0,1), 1e-5);
		EXPECT_NEAR(0.246407890289464, sens(1,1), 1e-5);
		EXPECT_NEAR(-0.246407890289465, sens(0,2), 1e-5);
		EXPECT_NEAR(-0.384948266739324, sens(1,2), 1e-5);
	}
}

TEST(SensitivityTest, testIntegrateStiff) {
	testIntegrate(true);
}

TEST(SensitivityTest, testIntegrateNonStiff) {
	testIntegrate(false);
}

TEST(SensitivityTest, testHIV) {
	VecNd< HIVLatent::numberOfParameters() > parms;
	parms << 2.61, 0.0021, 0.443, 1.6e-5, 641.0, 0.0085, 0.0092, 0.289, 30.0, 0.99, 0.9;

	Vecd t(8);
	t << 0.3, 9.9, 19.9, 39.9, 79.9, 99.9, 139.9, 179.9;

	const int numSensitivities = HIVLatent::numberOfEquations() * HIVLatent::numberOfParameters();
	const int N = HIVLatent::numberOfEquations() + numSensitivities;

	Matrixd res(N, t.size());
	{
		SensitivityODE<HIVLatent> sens(parms);
		integrate(sens, t, res);
	}

	Matrixd res2(N, t.size());
	{
		ode::ODE<HIVLatent> ode(parms);
		ode::SensitivityIntegrator integrator(ode);
		integrator.setIC();

		for(int i = 0; i < t.size(); ++i) {
			integrator.integrateTo(t[i]);
			auto s = res2.col(i);
			s.head(HIVLatent::numberOfEquations()) = integrator.currentState();
			const Matrixd & sens = integrator.currentSensitivity();
			s.tail(numSensitivities) = MapVecd(const_cast<double *>(sens.data()), sens.rows() * sens.cols());
		}
	}

	AssertEqualRel(res, res2, 1e-6, 1e-5);
}

TEST(SensitivityTest, testLocke) {
	VecNd< Locke::numberOfParameters() > parms;
	parms << 3.7051,  9.7142,  7.8618,  3.2829,  6.3907,  1.0631,  0.9271,
			5.0376,  7.3892,  0.4716,  4.1307,  5.7775,  4.4555,  7.6121,
			0.6187,  7.7768,  9.0002,  3.6414,  5.6429,  8.2453,  1.2789,
			5.3527,  0.1290, 13.6937,  9.1584,  1.9919,  5.9266,  1.1007;

	Vecd t = Vecd::LinSpaced(100, 0.0, 48.0);

	const int numSensitivities = Locke::numberOfEquations() * Locke::numberOfParameters();
	const int N = Locke::numberOfEquations() + numSensitivities;

	Matrixd res(N, t.size());
	{
		SensitivityODE<Locke> sens(parms);
		integrate(sens, t, res);
	}

	Matrixd res2(N, t.size());
	{
		ode::ODE<Locke> ode(parms);
		ode::SensitivityIntegrator integrator(ode);
		integrator.setIC();

		for(int i = 0; i < t.size(); ++i) {
			integrator.integrateTo(t[i]);
			auto s = res2.col(i);
			s.head(Locke::numberOfEquations()) = integrator.currentState();
			const Matrixd & sens = integrator.currentSensitivity();
			s.tail(numSensitivities) = MapVecd(const_cast<double *>(sens.data()), sens.rows() * sens.cols());
		}
	}

	AssertEqualRel(res, res2, 1e-6, 1e-5);
}

TEST(SensitivityTest, testLinearModel) {
	VecNd< LinearModel::numberOfParameters() > parms;
	parms.setOnes();

	Vecd t(10);
	t << 0.000001, 0.222222, 0.444444, 0.666667, 0.888889, 1.11111, 1.33333, 1.55556, 1.77778, 2.0;

	const int numSensitivities = LinearModel::numberOfEquations() * LinearModel::numberOfParameters();
	const int N = LinearModel::numberOfEquations() + numSensitivities;

	Matrixd res(N, t.size());
	{
		SensitivityODE<LinearModel> sens(parms);
		integrate(sens, t, res);
	}

	Matrixd res2(N, t.size());
	{
		ode::ODE<LinearModel> ode(parms);
		ode::SensitivityIntegrator integrator(ode);
		integrator.setIC();

		for(int i = 0; i < t.size(); ++i) {
			integrator.integrateTo(t[i]);
			auto s = res2.col(i);
			s.head(LinearModel::numberOfEquations()) = integrator.currentState();
			const Matrixd & sens = integrator.currentSensitivity();
			s.tail(numSensitivities) = MapVecd(const_cast<double *>(sens.data()), sens.rows() * sens.cols());
		}
	}

	AssertEqualRel(res, res2, 1e-6, 1e-5);
}

TEST(SensitivityTest, testPKPD) {
	VecNd< PKPD::numberOfParameters() > parms;
	parms << 1.00, 0.500,10.00, 54.5981500, 148.4131591, 0.3678794, 1250.00, 0.00, 300.00;

    Vecd t(7);
    t << 0, 0.5000, 1.0000, 2.0000, 4.0000, 8.0000, 25.0000;

	const int numSensitivities = PKPD::numberOfEquations() * PKPD::numberOfParameters();
	const int N = PKPD::numberOfEquations() + numSensitivities;

	Matrixd res(N, t.size());
	{
		SensitivityODE<PKPD> sens(parms);
		integrate(sens, t, res);
	}

	Matrixd res2(N, t.size());
	{
		ode::ODE<PKPD> ode(parms);
		ode::SensitivityIntegrator integrator(ode);
		integrator.setIC();

		for(int i = 0; i < t.size(); ++i) {
			integrator.integrateTo(t[i]);
			auto s = res2.col(i);
			s.head(PKPD::numberOfEquations()) = integrator.currentState();
			const Matrixd & sens = integrator.currentSensitivity();
			s.tail(numSensitivities) = MapVecd(const_cast<double *>(sens.data()), sens.rows() * sens.cols());
		}
	}

	AssertEqualRel(res, res2, 1e-6, 1e-5);
}
