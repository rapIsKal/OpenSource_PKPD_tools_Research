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
#include "TestCommon.hpp"
#include "simple.hpp"
#include <MEModel.hpp>
#include <ModelFactory.hpp>
#include <DictCpp.hpp>
#include <Random.hpp>
#include <SAEM.hpp>

TEST(MEModel, testSimpleSAEM) {
	auto A = Matrixd::Identity(3,3);
	auto B = make_matrix<double>({ {0, 0}, {1, 0}, {0, 1} });

	Population population;
	for(size_t i = 0; i < 1000; ++i) {
		auto t = MapVecd(simple_individuals[i].t, 10);
		auto y = MapMatrixd(simple_individuals[i].y, 1, 10);
		auto sm = ModelFactory::instance().createStructuralModel("Linear", DictCpp{});
		auto lm = ModelFactory::instance().createLikelihoodModel("Gaussian", DictCpp{"dim", 1, "t", Vecd(t), "y", Matrixd(y)});
		population.add( Individual(i, std::move(sm), std::move(lm), A, B));
	}

	auto beta = make_vector<double>({.7, .5, 1.});
	auto estimated = make_vector<bool>({true, true, true});

	auto omega = make_matrix<double>({{1.0, 0.0}, {0.0, 1.0}});
	auto cov_model = make_matrix<bool>({{true, true}, {true, true}});

	auto sigma = make_vector<double>({1.0});
	auto sigma_model = make_vector<bool>({true});

	MEModel model(population, beta, estimated, omega, cov_model, sigma, sigma_model);

	Random rng;
	SAEM::Options opts;
	opts.K1 = 120;
	opts.K2 = 20;
	SAEM algo(model, opts);
	algo.optimize(rng);

	auto beta_opt = make_vector<double>({0.862853, 0.360712, -1.27187});
	auto omega_opt = make_matrix<double>({{0.9, 0.5}, {0.5, 1.0}});
	auto sigma_opt = make_vector<double>({.1});

	AssertEqual(model.beta(), beta_opt, 1e-1);
	AssertEqual(model.omega(), omega_opt, 1e-1);
	AssertEqual(model.sigma(), sigma_opt, 1e-1);
}
