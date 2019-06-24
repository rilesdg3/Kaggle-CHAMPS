/*
 * Champs.cpp
 *
 *  Created on: Jun 20, 2019
 *      Author: ryan
 */

#include <Champs.h>

Champs::Champs() {
	// TODO Auto-generated constructor stub

}

Champs::~Champs() {
	// TODO Auto-generated destructor stub
}

void Champs::ChampsMain(){


	//Parse(this->filename_,600000);
	//Parse(this->structures_filename_,this->atom_set_,2);
	auto train = Parse(this->train_filename_,this->type_set_,4);
	std::vector<double> type_grouped;

	//MyData::GroupBy(this->type_set_,train,type_grouped, 4,5);

	auto tmp = MyData::VectorVectorStringToVectorFloat(train, 5);

	//simple stats and distribution of scalar coupling
	Stats::Histogram hist_st;
	Stats::CalcSimpleStats(tmp,hist_st,14);
	Stats::PlotHist(hist_st,this->file_path_, "scalar_coupling_hist");



	//simple stats distribution of scalar coupling by type
	std::map<string, std::vector<double> > type_to_scalar_constants;
	MyData::GroupBy(train, type_to_scalar_constants,4,5);
	std::map<string, std::vector<double> >::iterator it;
	for(it = type_to_scalar_constants.begin(); it!= type_to_scalar_constants.end(); ++it){

		cout<<it->first<<endl;//" "<<it->second[0]<<endl;
		Stats::Histogram hist_st;
		Stats::CalcSimpleStats(it->second,hist_st,14);

		Stats::PlotHist(hist_st,this->file_path_, it->first+"_hist");

	}

}

void Champs::SetFileNames(string path){

	this->file_path_ = path;

	this->structures_filename_ = path+"structures.csv";

	this->dipole_filename_ = path+"dipole_moments.csv";
	this->magnet_shielding_filename_ = path+"magnetic_shielding_tensors.csv";
	this->mulliken_filename_ = path+"mulliken_charges.csv";
	this->potential_filename_ = path+"potential_energy.csv";
	this->scalar_couple_contrib_filename_ = path+"scalar_coupling_contributions.csv";
	this->test_filename_ = path+"test.csv";
	this->train_filename_ = path+"train.csv";

}

/*
 * @param: string filename of the file to open
 * @param: set<string> my_set a set of the types and or atoms
 * @param: int set_column column number of variable you wan to put into the set
 *
 */
vector<vector<string> > Champs::Parse(string filename, std::set<string> &my_set, int set_column){

	int count = 0;

	vector<string> tmp_vect;
	vector<vector<string> > fnl_data;


	std::ifstream  data(filename);

	std::string line;


	//to skip header
	std::getline(data,line);
	while(std::getline(data,line))
	{
		std::stringstream  lineStream(line);
		std::string        cell;

		//cout<<"line "<<line<<endl;

			while(std::getline(lineStream,cell,','))
			{

				if(cell.size()>=1 && cell.compare("NaN")!=0){

					tmp_vect.push_back(cell);
					if(count == set_column)
						my_set.insert(cell);

				}
				else{
					cout<<"Parse line "<<line<<" cell "<<cell<<endl;
					tmp_vect.clear();
					break;
				}
				count++;

			}
			fnl_data.push_back(tmp_vect);
			tmp_vect.clear();
			count = 0;



	}

	return fnl_data;

}















