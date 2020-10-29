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

#include "Expected.hpp"

#include <gtest/gtest.h>
#include <cmath>

namespace {
	class CopyMoveCounter {
		public:
			CopyMoveCounter() : ncopies(0), nmoves(0) {}
			CopyMoveCounter(const CopyMoveCounter & other) : ncopies(other.ncopies + 1), nmoves(other.nmoves) {}
			CopyMoveCounter(CopyMoveCounter && other) : ncopies(other.ncopies), nmoves(other.nmoves + 1) {}

			CopyMoveCounter & operator =(const CopyMoveCounter & other) {
				ncopies = other.ncopies + 1;
				nmoves = other.nmoves;
				return *this;
			}

			CopyMoveCounter & operator =(CopyMoveCounter && other) {
				ncopies = other.ncopies;
				nmoves = other.nmoves + 1;
				return *this;
			}

		public:
			int copyCount() const { return ncopies; }
			int moveCount() const { return nmoves; }

		private:
			int ncopies, nmoves;
	};

	struct MoveFlag {
		MoveFlag() = default;
		MoveFlag& operator=(const MoveFlag&) = delete;
		MoveFlag(const MoveFlag&) = delete;
		MoveFlag(MoveFlag&& other) noexcept {
			other.moved = true;
		}
		bool moved { false };
	};
}

TEST(ExpectedTest, testPiecewiseConstruct) {
	auto t = Expected<std::pair<double, int>>::success(0.5, 2);
	EXPECT_TRUE( t.hasValue() );

	auto val = t.value();
	EXPECT_EQ(val.first, 0.5);
	EXPECT_EQ(val.second, 2);
}

TEST(ExpectedTest, testMapExpected) {
	{
		Expected<double> t;
		t = tryTo([]{ return 5.2; });
		Expected<double> s = t.map([](Expected<double> & x) { return std::floor(x.value()); });
		EXPECT_TRUE( s.hasValue() );
		EXPECT_EQ(s.value(), 5.0);
	}

	{
		Expected<double> t = tryTo([]{ return 5.2; })
					.map([](Expected<double> & x) { return std::floor(x.value()); });
		EXPECT_TRUE( t.hasValue() );
		EXPECT_EQ(t.value(), 5.0);
	}

	{
		Expected<double> t = tryTo([]{ return 5.2; })
					.map([](Expected<double> & x) { return Expected<double>::success(std::floor(x.value())); });
		EXPECT_TRUE( t.hasValue() );
		EXPECT_EQ(t.value(), 5.0);
	}

	{
		Expected<MoveFlag> t = tryTo([]{return MoveFlag();}).map([](MoveFlag && v) { return std::move(v); });
		EXPECT_FALSE(t.value().moved);
	}
}

TEST(ExpectedTest, testMapSuccess) {
	Expected<double> t;
	t = tryTo([]{ return 5.2; });
	Expected<double> s = t.map([](double x) { return std::floor(x); });
	EXPECT_TRUE( s.hasValue() );
	EXPECT_EQ(s.value(), 5.0);
}

TEST(ExpectedTest, testMapException) {
	Expected<double> t;
	t = tryTo([]{ throw std::logic_error("blaa"); return 5.0; });
	Expected<double> s = t.map([](double x) { return std::floor(x); });

	try {
		s.value();
		FAIL();
	} catch( std::logic_error & e) {
		EXPECT_EQ(e.what(),std::string("blaa"));
	} catch( ... ) {
		FAIL();
	}
}

TEST(ExpectedTest, testMapFailure) {
	Expected<double> t;
	t = tryTo([]{ return 5.2; });
	Expected<double> s = t.map([](double x){ throw std::logic_error("blaa"); return std::floor(x); });

	try {
		s.value();
		FAIL();
	} catch( std::logic_error & ) {
		SUCCEED();
	} catch( ... ) {
		FAIL();
	}
}

TEST(ExpectedTest, testRecover) {
	{
		Expected<void> t;
		t = tryTo([]{ throw std::logic_error("blaa"); }).recover([](std::exception_ptr) {});
		EXPECT_TRUE( t.hasValue() );
	}

	{
		Expected<double> t;
		t = tryTo([]{ throw std::logic_error("blaa"); return 5.0; }).recover([](std::exception_ptr) -> int { return 0; });
		EXPECT_TRUE( t.hasValue() );
		EXPECT_EQ(t.value(), 0);
	}
}

TEST(ExpectedTest, testRecoverFailure) {
	Expected<double> t;
	t = tryTo([]{ throw std::logic_error("blaa"); return 5.0; }).recover(
			[](std::exception_ptr){ throw std::logic_error("blaa"); return 0; }
	);

	try {
		t.value();
		FAIL();
	} catch( std::logic_error & ) {
		SUCCEED();
	} catch( ... ) {
		FAIL();
	}
}

TEST(ExpectedTest, testValue) {
	Expected<double> t;
	t = tryTo([]{ return 5.0; });
	EXPECT_TRUE( t.hasValue() );
	EXPECT_EQ(t.value(), 5.0);
}

TEST(ExpectedTest, testException) {
	Expected<double> t;
	t = tryTo([]{ throw std::logic_error("blaa"); return 5.0; });
	EXPECT_TRUE( t.hasException() );

	try {
		t.value();
		FAIL();
	} catch( std::logic_error & ) {
		SUCCEED();
	} catch( ... ) {
		FAIL();
	}
}

TEST(ExpectedTest, testSimpleValue) {
	Expected<double> t(5.0);
	EXPECT_TRUE( t.hasValue() );
	EXPECT_EQ(t.value(), 5.0);
}

TEST(ExpectedTest, testValueNoCopies) {
	Expected<CopyMoveCounter> t;
	t = tryTo([]{ return CopyMoveCounter(); });
	EXPECT_TRUE( t.hasValue() );
	EXPECT_EQ(t.value().copyCount(), 0);

	Expected<CopyMoveCounter> s;
	s = t.map([](const CopyMoveCounter & cmc) {
		EXPECT_EQ(cmc.copyCount(), 0);
		return CopyMoveCounter();
	});
	EXPECT_TRUE( s.hasValue() );
	EXPECT_EQ(s.value().copyCount(), 0);
}
