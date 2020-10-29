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

#ifndef TESTCOMMON_H
#define TESTCOMMON_H

#include <gtest/gtest.h>
#include <MatrixVector.hpp>
#include <math/number.hpp>
#include <math/gradients.hpp>
#include <apply.hpp>

testing::AssertionResult DoubleSimilar(
		const char* expr1, const char* expr2, const char* abs_error_expr,
		double val1, double val2, double abs_error);
testing::AssertionResult DoubleSimilarNaN(
		const char* expr1, const char* expr2, const char* abs_error_expr,
		double val1, double val2, double abs_error);

#define EXPECT_SIMILAR(val1, val2, abs_error)\
  EXPECT_PRED_FORMAT3(DoubleSimilar, \
                      val1, val2, abs_error)

#define EXPECT_SIMILAR_OR_NAN(val1, val2, abs_error)\
  EXPECT_PRED_FORMAT3(DoubleSimilarNaN, \
                      val1, val2, abs_error)

inline static double epsilon(double x, double rel_eps, double cutoff) {
	return std::max( rel_eps * std::abs(x), cutoff );
}

template <typename XT, typename YT>
void AssertEqualRel(const MatrixBase<XT> & x, const MatrixBase<YT> & y,
					const double rel_eps=1e-6, const double cutoff=1e-5) {
	EXPECT_EQ( x.rows(), y.rows() );
	EXPECT_EQ( x.cols(), y.cols() );
	for(int i = 0; i < x.size(); ++i)
		EXPECT_SIMILAR( x(i), y(i), epsilon(x(i), rel_eps, cutoff) );
}


template <typename XT, typename YT>
void AssertEqual(const MatrixBase<XT> & x, const MatrixBase<YT> & y, const double eps=1e-8) {
	EXPECT_EQ( x.rows(), y.rows() );
	EXPECT_EQ( x.cols(), y.cols() );
	for(int i = 0; i < x.size(); ++i)
		EXPECT_SIMILAR( x(i), y(i), eps );
}

template <typename XT, unsigned int Mode, typename YT>
void AssertEqual(const Eigen::TriangularView<XT, Mode> & x, const MatrixBase<YT> & y, const double eps=1e-8) {
	EXPECT_EQ( x.rows(), y.rows() );
	EXPECT_EQ( x.cols(), y.cols() );

	if( Mode == Eigen::Upper ) {
		for(int j = 0; j < x.cols(); ++j)
			for(int i = 0; i <= j; ++i)
				EXPECT_SIMILAR( x(i,j), y(i,j), eps );
	} else {
		for(int j = 0; j < x.cols(); ++j)
			for(int i = j; i < x.rows(); ++i)
				EXPECT_SIMILAR( x(i,j), y(i,j), eps );
	}
}

template <typename T, typename... Args>
struct TestValues {
	std::tuple<Args...> params;
	T value;
};

template <typename T>
struct TestEqual {
	template <typename FTOL>
	static void check(const T &expected, const T &actual, const FTOL &ftol) {
		EXPECT_SIMILAR_OR_NAN(expected, actual, ftol(expected));
	}
};

template <std::size_t N>
struct TestEqual<math::Gradients<N>> {
	template <typename FTOL>
	static void check(const math::Gradients<N> &expected, const math::Gradients<N> &actual, const FTOL &ftol) {
		for(std::size_t i = 0; i < N; ++i)
			EXPECT_SIMILAR_OR_NAN(expected[i], actual[i], ftol(expected[i]));
	}
};

template <typename T, typename... Args, std::size_t N, typename FTOL>
void testFunction(const TestValues<T, Args...> (&v)[N], T (&f)(Args...), const FTOL &ftol) {
	for(std::size_t i = 0; i < N; ++i) {
		try {
			const T val = apply(f, v[i].params);
			TestEqual<T>::check(v[i].value, val, ftol);
		} catch( std::domain_error &  ) {
			TestEqual<T>::check(v[i].value, math::NaN(), ftol);
		}
	}
}

template <typename T, typename... Args, std::size_t N>
void testFunction(const TestValues<T, Args...>(&v)[N], T (&f)(Args...)) {
	auto ftol = [](double expected) { return epsilon(expected, 1e-14, 1e-14); };
	testFunction(v, f, ftol);
}

#define EXPECT_DOMAIN_ERROR(expr) \
	do {\
		try {\
			auto val = expr;\
			EXPECT_TRUE( val ); \
		} catch( std::domain_error & ) {\
		}\
	} while(0)

template <typename T, typename... Args, std::size_t N>
void testDistribution(const TestValues<T, Args...>(&v)[N], T (&f)(Args...), T (&fProp)(Args...)) {
	for(std::size_t i = 0; i < N; ++i) {
		try {
			const double val = apply(f, v[i].params);
			EXPECT_SIMILAR_OR_NAN( v[i].value, val, 5e-14 );

			if( math::isFinite(val) ) {
				const double valProp = apply(fProp, v[i].params);

				std::tuple<Args...> p = v[i].params;
				std::get<0>(p) += epsilon( std::get<0>(p), 1e-6, 1e-14 );

				const double valEps = apply(f, p);
				const double valPropEps = apply(fProp, p);

				EXPECT_SIMILAR( valEps - val, valPropEps - valProp, 1e-14 );
			}
		} catch( std::domain_error &  ) {
			EXPECT_TRUE( math::isNaN(v[i].value) );
		}
	}
}

template <typename T, typename... Args, std::size_t N>
void testDistributionDx(const TestValues<T, Args...>(&v)[N], T (&f)(Args...), T (&fdx)(Args...)) {
	for(std::size_t i = 0; i < N; ++i) {
		if( ! math::isFinite(v[i].value) )
			continue;

		try {
			std::tuple<Args...> p = v[i].params;
			const double eps = epsilon( std::get<0>(p), 1e-8, 1e-14 );
			std::get<0>(p) += eps;

			const double val = apply(f, v[i].params);
			const double valEps = apply(f, p);

			const double dx = apply(fdx, v[i].params);
			EXPECT_SIMILAR( (valEps - val) / eps, dx, 1e-6 );
		} catch( std::domain_error & e ) {
			EXPECT_TRUE( math::isNaN(v[i].value) );
		}
	}
}
#endif
