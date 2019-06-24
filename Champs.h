/*
 * Champs.h
 *
 *  Created on: Jun 20, 2019
 *      Author: ryan
 */

#ifndef CHAMPS_CHAMPS_H_
#define CHAMPS_CHAMPS_H_

#include "Data.h"
#include "Stats.h"

class Champs {
public:
	Champs();
	virtual ~Champs();

	void ChampsMain();
	vector<vector<string> > Parse(string filename, std::set<string> &my_set, int set_column =-1);
	void SetFileNames(string path);
	void SetDataIter(int end);


	std::string file_path_;
	std::string structures_filename_;
	std::string dipole_filename_;
	std::string magnet_shielding_filename_;
	std::string mulliken_filename_;
	std::string potential_filename_;
	std::string scalar_couple_contrib_filename_;
	std::string test_filename_;
	std::string train_filename_;



	std::set<string> atom_set_;
	std::set<string> type_set_;
	std::vector<std::vector<string > > stuctures_vect;

	std::vector<float > hist_vect_;//marginal-> or percent of data that falls in that bin
	std::vector<float > hist_vect_bin_;//bin values so (-inf,1],(1,2],....(n,inf) were 1,2,...,n represent the range of the bins
	std::multimap<double, std::vector<float>,MyData::classcomp > data_;
	std::multimap<double, std::vector<float>,MyData::classcomp >::iterator data_it_start_;
	std::multimap<double, std::vector<float>,MyData::classcomp >::iterator data_it_end_;

};

#endif /* CHAMPS_CHAMPS_H_ */
