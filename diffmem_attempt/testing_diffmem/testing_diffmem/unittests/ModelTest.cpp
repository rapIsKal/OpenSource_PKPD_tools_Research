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
#include "FiniteDifference.hpp"
#include "models/Fitzhugh.hpp"
#include "models/HIVLatent.hpp"
#include "TestCommon.hpp"
#include "models/Locke.hpp"
#include "models/PKPD.hpp"
#include "models/PK.hpp"

using testing::Types;

// The list of types we want to test.
typedef Types<HIVLatent, Fitzhugh, Locke, PKPD, PK> Implementations;

TYPED_TEST_CASE(ModelTest, Implementations);

template <typename Model>
class ModelTest : public testing::Test {
	public:
		typedef VecNd<Model::numberOfParameters()> Parameters;
		typedef VecNd<Model::numberOfEquations()> States;
		typedef VecNd<Model::numberOfObservations()> Observations;
		typedef MatrixMNd<Model::numberOfEquations(), Model::numberOfEquations()> Jac_u;
		typedef MatrixMNd<Model::numberOfEquations(), Model::numberOfParameters()> Jac_p;

	public:
		void setupParameters(Parameters & x) const;
		void setupODE(Parameters & x, double & t, States & u) const;
		void setupProjection(double & t, Parameters & x, States & u, Observations & yu) const;
};

TYPED_TEST(ModelTest, testODEInitial) {
	typedef TypeParam Model;
	typedef VecNd<Model::numberOfParameters()> Parameters;
	typedef VecNd<Model::numberOfEquations()> States;
	typedef MatrixMNd<Model::numberOfEquations(), Model::numberOfParameters()> Jac;

	Parameters x;
	double t;
	States u;
	this->setupODE(x, t, u);

	auto f = [](ConstRefVecd &x_, RefVecd u_) {
		Parameters x = x_;
		States u;
		Model::ODE::initial(x, u);
		u_ = u;
	};

	Jac J_fd;
	FiniteDifference::computeJacobian(f, x, u, J_fd);

	Jac J;
	Model::ODE::jacobianParametersInitial(x, J);
	AssertEqualRel(J_fd, J, 1e-4, 1e-4);
}

TYPED_TEST(ModelTest, testODESol) {
	typedef TypeParam Model;
	typedef VecNd<Model::numberOfParameters()> Parameters;
	typedef VecNd<Model::numberOfEquations()> States;
	typedef MatrixMNd<Model::numberOfEquations(), Model::numberOfEquations()> Jac;

	Parameters x;
	double t;
	States u;
	this->setupODE(x, t, u);

	auto f = [t, &x](ConstRefVecd & u_, RefVecd du_) {
		States u = u_;
		States du;
		Model::ODE::ddt(t, u, x, du);
		du_ = du;
	};

	States du;
	Jac J_fd;
	FiniteDifference::computeJacobian(f, u, du, J_fd, 1e-6, 1e-8);

	Jac J;
	Model::ODE::jacobianState(t, u, x, J);
	AssertEqualRel(J_fd, J, 1e-3);
}

TYPED_TEST(ModelTest, testODEPar) {
	typedef TypeParam Model;
	typedef VecNd<Model::numberOfParameters()> Parameters;
	typedef VecNd<Model::numberOfEquations()> States;
	typedef MatrixMNd<Model::numberOfEquations(), Model::numberOfParameters()> Jac;

	Parameters x;
	double t;
	States u;
	this->setupODE(x, t, u);

	auto f = [t, &u](ConstRefVecd & x_, RefVecd du_) {
		Parameters x = x_;
		States du;
		Model::ODE::ddt(t, u, x, du);
		du_ = du;
	};

	States du;
	Jac J_fd;
	FiniteDifference::computeJacobian(f, x, du, J_fd, 1e-6, 1e-8);

	Jac J;
	Model::ODE::jacobianParameters(t, u, x, J);
	AssertEqualRel(J_fd, J, 1e-5);
}

TYPED_TEST(ModelTest, testProjection) {
	typedef TypeParam Model;
	typedef VecNd<Model::numberOfParameters()> Parameters;
	typedef VecNd<Model::numberOfEquations()> States;
	typedef VecNd<Model::numberOfObservations()> Observations;
	typedef MatrixMNd<Model::numberOfObservations(), Model::numberOfEquations()> Jac;

	double t;
	Parameters x;
	States u;
	Observations yu_cor;
	this->setupProjection(t, x, u, yu_cor);

	auto f = [t, &x](ConstRefVecd & u_, RefVecd yu_) {
		States u = u_;
		Observations yu;
		Model::Projection::project(t, u, x, yu);
		yu_ = yu;
	};

	Observations yu;
	Jac J_fd; J_fd.setOnes();
	FiniteDifference::computeJacobian(f, u, yu, J_fd);
	AssertEqualRel(yu, yu_cor, 1e-4);

	Jac J; J.setOnes();
	Model::Projection::jacobian(t, u, x, J);
	AssertEqualRel(J_fd, J, 1e-4, 1e-4);
}

TYPED_TEST(ModelTest, testTransform) {
	typedef TypeParam Model;
	typedef VecNd<Model::numberOfParameters()> Parameters;

	Parameters psi;
	this->setupParameters(psi);

	Parameters phi;
	Model::Transform::transphi(psi, phi);

	auto f = [](ConstRefVecd & phi_, RefVecd psi_) {
		Parameters phi = phi_;
		Parameters psi;
		Model::Transform::transpsi(phi, psi);
		psi_ = psi;
	};

	typedef MatrixMNd<Model::numberOfParameters(), Model::numberOfParameters()> Jac;
	Jac J_fd; J_fd.setOnes();
	FiniteDifference::computeJacobian(f, phi, psi, J_fd);

	Parameters grad; grad.setOnes();
	Model::Transform::dtranspsi(phi, grad);
	Jac J = grad.asDiagonal();

	AssertEqualRel(J, J_fd, 1e-4);
}

template <>
void ModelTest<HIVLatent>::setupParameters(Parameters & x) const {
	x << 2.61, 0.0021, 0.443, 1.6e-5, 641, .0085, 0.0092, .289, 30, .99, .90;
}

template <>
void ModelTest<HIVLatent>::setupODE(Parameters & x, double & t, States & u) const {
	x << 2.61, 0.0021, 0.443, 1.6e-5, 641, .0085, 0.0092, .289, 30, .99, .90;
	t = 0.0;
	u << 41.9104, 44.4992, 0.963129, 6.66067, 0.0;
}

template <>
void ModelTest<HIVLatent>::setupProjection(double & t, Parameters & x, States & u, Observations & yu) const {
	t = 0.0;
	x << 2.61, 0.0021, 0.443, 1.6e-5, 641, .0085, 0.0092, .289, 30, .99, .90;
	u << 41.9104, 44.4992, 0.963129, 6.66067, 0.0;
	yu << 3.82352, 87.3728;
}

template <>
void ModelTest<Fitzhugh>::setupParameters(Parameters & x) const {
	x << .2, .2, 3.0, -1, 1;
}

template <>
void ModelTest<Fitzhugh>::setupODE(Parameters & x, double & t, States & u) const {
	x << .2, .2, 3.0, -1, 1;
	t = 0.0;
	u << -1, 1;
}

template <>
void ModelTest<Fitzhugh>::setupProjection(double & t, Parameters & x, States & u, Observations & yu) const {
	t = 0.0;
	x << .2, .2, 3.0, -1, 1;
	u << -1, 1;
	yu << -1, 1;
}

template <>
void ModelTest<Locke>::setupParameters(Parameters & x) const {
	x << 3.7051,  9.7142,  7.8618,  3.2829,  6.3907,  1.0631,  0.9271,
            5.0376,  7.3892,  0.4716,  4.1307,  5.7775,  4.4555,  7.6121,
            0.6187,  7.7768,  9.0002,  3.6414,  5.6429,  8.2453,  1.2789,
            5.3527,  0.1290, 13.6937,  9.1584,  1.9919,  5.9266,  1.1007;
}

template <>
void ModelTest<Locke>::setupODE(Parameters & x, double & t, States & u) const {
	x << 3.7051,  9.7142,  7.8618,  3.2829,  6.3907,  1.0631,  0.9271,
            5.0376,  7.3892,  0.4716,  4.1307,  5.7775,  4.4555,  7.6121,
            0.6187,  7.7768,  9.0002,  3.6414,  5.6429,  8.2453,  1.2789,
            5.3527,  0.1290, 13.6937,  9.1584,  1.9919,  5.9266,  1.1007;
	t = 0.0;
	u << 0.129, 13.6937, 9.1584, 1.9919,   5.9266,   1.1007;
}

template <>
void ModelTest<Locke>::setupProjection(double & t, Parameters & x, States & u, Observations & yu) const {
	t = 0.0;
	x << 3.7051,  9.7142,  7.8618,  3.2829,  6.3907,  1.0631,  0.9271,
            5.0376,  7.3892,  0.4716,  4.1307,  5.7775,  4.4555,  7.6121,
            0.6187,  7.7768,  9.0002,  3.6414,  5.6429,  8.2453,  1.2789,
            5.3527,  0.1290, 13.6937,  9.1584,  1.9919,  5.9266,  1.1007;
	u << 0.129, 13.6937, 9.1584, 1.9919,   5.9266,   1.1007;
	yu << 0.129, 13.6937, 9.1584, 1.9919,   5.9266,   1.1007;
}

template <>
void ModelTest<PKPD>::setupParameters(Parameters & x) const {
	x << 1.00, 0.500,10.00, 54.5981500, 148.4131591, 0.3678794, 1250.00, 0.00, 300.00;
}

template <>
void ModelTest<PKPD>::setupODE(Parameters & x, double & t, States & u) const {
	x << 1.00, 0.500,10.00, 54.5981500, 148.4131591, 0.3678794, 1250.00, 0.00, 300.00;
	t = 0.0;
	u << 1250.00, 0.00, 300.00;
}

template <>
void ModelTest<PKPD>::setupProjection(double & t, Parameters & x, States & u, Observations & yu) const {
	t = 0.0;
	x << 1.00, 0.500,10.00, 54.5981500, 148.4131591, 0.3678794, 1250.00, 0.00, 300.00;
	u << 1250.00, 0.00, 300.00;
	yu << 0.00, 300.00;
}

template <>
void ModelTest<PK>::setupParameters(Parameters & x) const {
	x << 1.00, 0.500,10.00;
}

template <>
void ModelTest<PK>::setupODE(Parameters & x, double & t, States & u) const {
	x << 1.00, 0.500,10.00;
	t = 0.0;
	u << 1250.00, 0.00;
}

template <>
void ModelTest<PK>::setupProjection(double & t, Parameters & x, States & u, Observations & yu) const {
	t = 0.0;
	x << 1.00, 0.500,10.00;
	u << 1250.00, 0.00;
	yu << 0.00;
}

