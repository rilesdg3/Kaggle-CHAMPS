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

#include <caffe2/core/init.h>
#include <caffe2/core/tensor.h>
#include "blob.h"
#include "model.h"
#include "net.h"

#include "ChampsNet.h"

#include <limits>
#include <numeric>
#include <sys/stat.h>

class Champs {
public:
	Champs();
	virtual ~Champs();

	void ChampsMain();
	void ChampsTrain();
	void ScalarCouplingContributions();
	void MergeTrainAndStructures();
	void BuildFeatureVector(std::map<string, std::vector<std::vector<double> > > &data_map, std::vector<float> &not_a_feature);
	void BuildNet();

	void PredictAll();
	void BuildNet2();
	void print(const caffe2::Blob *blob, const std::string &name);
	vector<vector<string> > Parse(string filename, std::set<string> &my_set, int set_column =-1);
	vector<vector<string> > Parse(string filename);
	void SetFileNames(string path);
	void SetDataIter(int end);
	template<typename T, typename T1>
		void SaveDataFirstSavedAsInt(T &path, std::vector<std::vector<T1> > &data);



	std::string file_path_;
	std::string model_name = "test_name";
	std::string init_model_name = "init"+model_name;

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
	std::vector<std::vector<float > > features_;
	std::vector<float> scalar_couplings_;
	std::vector<float> test_ids_;
	std::vector<std::vector<string > > stuctures_vect;
	std::map<string, std::vector<std::vector<double> > > stuctures_map_;
	std::map<string, std::vector<std::vector<double> > > train_map_;
	std::map<string, std::vector<std::vector<double> > > test_map_;

	std::vector<float > hist_vect_;//marginal-> or percent of data that falls in that bin
	std::vector<float > hist_vect_bin_;//bin values so (-inf,1],(1,2],....(n,inf) were 1,2,...,n represent the range of the bins
	std::multimap<double, std::vector<float>,MyData::classcomp > data_;
	std::multimap<double, std::vector<float>,MyData::classcomp >::iterator data_it_start_;
	std::multimap<double, std::vector<float>,MyData::classcomp >::iterator data_it_end_;

};

#endif /* CHAMPS_CHAMPS_H_ */
