/*
 * Stats.h
 *
 *  Created on: Mar 18, 2019
 *      Author: ryan
 */

#ifndef STATS_H_
#define STATS_H_

#include <ReadWriteNet.h>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/statistics/skewness.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/density.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/plot.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cvplot.h"

#include <typeinfo>

namespace bacc = boost::accumulators;
typedef boost::accumulators::accumulator_set<double, boost::accumulators::features<boost::accumulators::tag::density> > acc;
typedef boost::iterator_range<std::vector<std::pair<double, double> >::iterator > histogram_type;

//typedef boost::accumulators::accumulator_set<double, bacc::features<bacc::tag::density> > facc;
typedef bacc::accumulator_set< double, bacc::features<
			bacc::tag::min,
			bacc::tag::max,
			bacc::tag::mean,
			bacc::tag::median,
			bacc::tag::variance,
			bacc::tag::skewness,
			bacc::tag::density > > stat_acc;//w_acc;





namespace Stats{
struct Histogram {

	int count = 0;//number of observations in the data set-> or ncases
	std::vector<short int> bin_index;//hold the bin the values(cases) in data ex: data[0] = -5 whic in at the histogram
					//is bin 3 then bin_index[0] = 3
	//std::vector<double> marginals;
	std::vector<float> marginals;
	std::vector<double> bins;
	std::vector<float > hist_vect;//marginal-> or percent of data that falls in that bin
	std::vector<float > hist_vect_bin;//bin values so (-inf,1],(1,2],....(n,inf) were 1,2,...,n represent the range of the bins
	std::vector<std::vector<int > > contingency_table;


};


	template<typename T>
	void CalcSimpleStats(vector<T> &data,Histogram &hist_st, int n_bins = 10);

	void PlotHist(Histogram &hist_st, string file_path, string file_name);
	template< typename T >
	void PlotLine(std::vector<T> &data, string name);
	template< typename T >
	void ComputeHistograms(std::vector<T> &data,Histogram &hist_st, int n_bins = 10);
	template<typename T, typename T1>
	void LaggedMI(std::vector<T> pred_vars, std::vector<T1 > target, int n_bins, int min_lag, int max_lag, int lag_step);
	void LaggedMI(std::vector<std::vector<double> > embedVect, std::vector<double > vect_y_alligned, int n_bins=10);
	double DiscreteMI (Histogram &pred_hist_st, Histogram &target_hist_st);
	double Entropy (Histogram &hist_st);
	void ACF(std::vector<std::vector<double> > &embedVect, int lag = 1);
	template<typename T, typename T1>
	double correlation(int var_num, std::vector<std::vector<T> > &data, std::vector<T1> &target);
	double ComputeV (Histogram &pred_hist_st, Histogram &target_hist_st);



}


#endif /* STATS_H_ */
