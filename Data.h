/*
 * Data.h
 *
 *  Created on: Mar 18, 2019
 *      Author: ryan
 */

#pragma once
#ifndef DATA_H_
#define DATA_H_

#include <ReadWriteNet.h>
#include <unordered_set>
#include <limits>
#include <numeric>
#include <typeinfo>
#include <thread>
#include <cxxabi.h>


namespace MyData {


	struct classcomp {
	  bool operator() (const double& lhs, const double& rhs) const
	  {return lhs>rhs;}
	};

	// class generator:
	struct c_unique {
	  int current;
	  c_unique() {current=0;}
	  int operator()() {return ++current;}
	};// UniqueNumber;



	std::vector<float> VectorVectorStringToVectorFloat(std::vector<std::vector<string> > &data_in, int column);


	template< typename T, typename T1, typename T2, typename T3 >
	void GroupBy(std::set<T> &set_2_convert,std::vector<std::vector<T1> > &data, T2 &grouped, int group_by_column, int set_column, T3 &var_columns);

	template<typename T, typename T1 >
	void GroupBy(T &data, T1 &grouped, int group_by_column, int value_column);

	void EmbedVect(std::multimap<double, std::vector<float>, classcomp>::iterator cBegin, std::multimap<double, std::vector<float>,classcomp>::iterator cEnd,
	std::vector<std::vector<double> > &vect, int vect_data_it, int m, int d, bool skip);
	std::vector<std::vector<double> > EmbedThreading(std::multimap<double, std::vector<float>, classcomp >  &data ,int m, int d, bool skip,
	std::multimap<double, std::vector<float>, classcomp>::iterator cBegin, std::multimap<double, std::vector<float>,classcomp>::iterator cEnd, bool pnt_data = false);
	std::vector<std::vector<double> > Embed(std::multimap<double, std::vector<float>, classcomp >  &data ,int m, int d, bool skip);
	std::vector<double> AlignedYvectWithEmbedX(std::multimap<double, std::vector<float>, classcomp >  &data,
	std::multimap<double, std::vector<float>, classcomp>::iterator cBegin, std::multimap<double, std::vector<float>,classcomp>::iterator cEnd);


};

#endif /* DATA_H_ */
