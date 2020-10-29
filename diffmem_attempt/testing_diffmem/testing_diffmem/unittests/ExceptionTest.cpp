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

#include "Exception.hpp"
#include "Error.hpp"
#include <gtest/gtest.h>

TEST(Exception, testFormat) {
	try {
		throwSystemErrorExplicit(5, "hello world {}", 10);
		FAIL() << "throwSystemErrorExplicit() should throw an error\n";
	} catch(std::system_error &exception) {
		EXPECT_EQ(std::string(exception.what()), ("hello world 10: Input/output error"));
	}
}

class Date {
	int year_, month_, day_;

public:
	Date(int year, int month, int day)
			: year_(year), month_(month), day_(day) {}

	friend std::ostream &operator<<(std::ostream &os, const Date &d) {
		return os << d.year_ << '-' << d.month_ << '-' << d.day_;
	}
};

TEST(Exception, testOStream) {
	try {
		notYetImplemented("nyi {}", Date(2012, 12, 9));
		FAIL() << "notYetImplemented() should throw an error\n";
	} catch(notyetimplemented_error &exception) {
		EXPECT_EQ(std::string(exception.what()), ("nyi 2012-12-9"));
	}
}
