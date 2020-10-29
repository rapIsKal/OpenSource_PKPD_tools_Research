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

#ifndef CHISQ_TEST_H
#define CHISQ_TEST_H

#include <MatrixVector.hpp>
#include <math/sqr.hpp>
#include <stats/chisquared.hpp>
#include <Random.hpp>
#include <vector>

template <typename T>
class histogram {
public:
	template <typename It,
			typename = typename std::enable_if<std::is_convertible<
					typename std::iterator_traits<It>::iterator_category, std::input_iterator_tag>::value>::type>
	histogram(It first, It last, size_t nbins)
			: counts(nbins, size_t{0}) {
		find_range(first, last, nbins, std::is_floating_point<T>{});

		std::for_each(first, last, [this](T x) {
			if( x >= lb && x <= ub ) {
				size_t i = bin(x);
				counts[i]++;
			}
		});
	}

public:
	size_t nbins() const { return counts.size(); }
	size_t count(size_t idx) const { return counts[idx]; }
	T lower(size_t idx) const {
		return lb + idx * step;
	}
	T upper(size_t idx) const {
		return lower(idx+1);
	}
	size_t bin(T x) const {
		return (size_t)((x - lb) / step);
	}

private:
	template <typename It>
	void find_range(It first, It last, size_t n, std::true_type) {
		auto minmax = std::minmax_element(first, last);
		lb = *minmax.first;
		ub = *minmax.second + 1e-8;
		Require(lb != ub, "range should not be empty");

		step = (ub - lb) / n;
	}

	template <typename It>
	void find_range(It first, It last, size_t n, std::false_type) {
		auto minmax = std::minmax_element(first, last);
		lb = *minmax.first;
		ub = *minmax.second;
		Require(lb != ub, "range should not be empty");

		step = ((ub - lb) / n) + (((ub - lb) % n == 0) ? 0 : 1);
		ub = lb + step * n;
	}

private:
	T lb, step, ub;
	std::vector<size_t> counts;
};

template <typename Distribution, typename... Args>
double chisq_test(Random & rng, Args && ...args) {
	auto gen = Distribution::generator(rng, args...);
	const size_t n = 10000;

	using result_type = typename Distribution::result_type;
	using vec_type = Eigen::Matrix<result_type, Eigen::Dynamic, 1>;
	vec_type x = generate_random<vec_type>(gen, n);

	histogram<result_type> hist(begin(x), end(x), 50);

	double statistic = 0.0;
	double prev_p = Distribution::cdf(hist.lower(0), args...);
	for(size_t i = 0; i < hist.nbins() && prev_p != 1.0; ++i) {
		double p = Distribution::cdf(hist.upper(i), args...);
		const double expected = n * (p - prev_p);
		statistic += math::sqr(hist.count(i) - expected) / expected;
		prev_p = p;
	}

#if 0
	statistic = reduce_sum(0, hist.nbins(),
		[](size_t i) {
			const double p = Distribution::cdf(hist.upper(i), args...) - Distribution::cdf(hist.lower(i), args...);
			const double expected = n * p;
			return math::sqr(hist.count(i) - expected) / expected;
		}
	);
#endif

	return stats::Chisquared::cdf(statistic, n - 1);
}
#endif