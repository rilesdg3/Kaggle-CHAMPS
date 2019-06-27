/*
 * Data.cpp
 *
 *  Created on: Mar 18, 2019
 *      Author: ryan
 */

#include "Data.h"

/* @brief
 * A collection of Functions that converts data
 * into format/container type needed
 */

namespace MyData{



void EmbedVect(std::multimap<double, std::vector<float>, classcomp>::iterator cBegin, std::multimap<double, std::vector<float>,classcomp>::iterator cEnd,
		std::vector<std::vector<double> > &vect, int vect_data_it, int m, int d, bool skip){


	std::multimap<double, std::vector<float>,classcomp>::iterator iter;

	cout<<"EmbedVect cBegin "<<cBegin->first<<" cEnd "<<cEnd->first<<endl;
	int end = std::abs(m*d)-d+1;

		//int vect_size = data.size() - end;

	int vect_data_it_start = vect_data_it;

		//int vect_data_it =0;
		//boost::timer::auto_cpu_timer t;

		if(skip == false){
			for( ; cEnd!=cBegin; cEnd--){
				iter = cEnd;

				//cout<<std::distance(cBegin, cEnd)<<endl;

				for(int i =1; i<m; i++){
					std::advance(iter, -1*std::abs(d));

					//cEnd->second.insert(cEnd->second.begin(),iter->second[0]);
					//vect[m-i-1][vect_data_it] = iter->second[0];
					vect[vect_data_it][m-i-1] = iter->second[0];

					//vect[m-i-1].push_back(iter->second[0]);
					//cout<<"cEnd "<<cEnd->first<<" ";
					//cout<<"cEnd "<<cEnd->second<<" iter "<<iter->second<<endl;
					//cout<<"cEnd "<<cEnd->second<<" iter->second[0] "<<iter->second[0]<<endl;
					//cout<<"cEnd "<<cEnd->second<<" iter->second[1] "<<iter->second[1]<<endl;

					//cout<<"vect[m-i-1] "<<m-i-1<<" values "<<vect[m-i-1]<<endl;
				}
				if(std::distance(cBegin, cEnd)<end){
				//	data.erase(cBegin, cEnd);
					break;
				}
				vect_data_it++;
				//if(vect_data_it>=m+vect_data_it_start){
				//	break;
				//}
			}
		}
		else{
			for( ; cEnd!=cBegin; cEnd--){
				iter = cEnd;
				//cout<<std::distance(cBegin, cEnd)<<endl;

				for(int i =1; i<m; i++){
					std::advance(iter, -1*std::abs(d));
					//cout<<"cEnd b4 "<<cEnd->second<<" iter "<<iter->second<<endl;
					cEnd->second.insert(cEnd->second.begin(),iter->second[0]);
				}
				cEnd->second.pop_back();

				if(std::distance(cBegin, cEnd)<end){
				//	data.erase(cBegin, cEnd);
					break;
				}
			}
		}
		//Print(data);
		cout<<"vect_data_it_start "<<vect_data_it_start<<" vect_data_it "<<vect_data_it<<endl;

}

/*
 * @brief: performs time delay embedding using multiple threads
 *
 * @param data:
 * @param int m: number of embedding dimesnions
 * @param int d: time delay
 * @param bool skip: whether to skip the first value, this is used when when using same data but with different time delays
 *
 */
std::vector<std::vector<double> > EmbedThreading(std::multimap<double, std::vector<float>, classcomp >  &data ,int m, int d, bool skip,
		std::multimap<double, std::vector<float>, classcomp>::iterator cBegin, std::multimap<double, std::vector<float>,classcomp>::iterator cEnd,bool pnt_data ){


	/*std::multimap<double, std::vector<float>,classcomp>::iterator test;
	if(cBegin != data.begin()){
		cBegin = data.begin();
		cEnd = data.end();
	}*/

	std::multimap<double, std::vector<float>,classcomp>::iterator iter;
	//cEnd--;

	int end;

	if(pnt_data == true)
		end = std::abs(m*d)-d+1;
	else
		end = std::abs(m*d)-d;

	int distance = std::distance(cBegin,cEnd);

	int vect_size = end;//distance - end;//data.size() - end;
	std::vector<double > vect_data (vect_size);
	std::vector<std::vector<double> > vect(distance - end);//m-1);


	for(uint i = 0; i< vect.size(); ++i)
		vect[i] = vect_data;

	int n_cores = std::thread::hardware_concurrency();


	//cout<<"modulus "<<std::modulus<double>((double)vect_size/(double)n_cores)<<endl;
	cout<<"modulus "<<vect_size/n_cores<<endl;
	cout<<"modulus "<<vect_size%n_cores<<endl;

	std::div_t map_it_per_core= std::div(distance,n_cores);//std::div(data.size(),n_cores);
	//map_it_per_core();
	//data.size()/n_cores;
	//int iter_per_core = vect_size/n_cores;
	int iter_per_core = (distance - end)/n_cores;

	std::vector<std::thread> threads;

	int what_to_name = 0;//vect_size;

	iter = cEnd;
	cBegin = cEnd;

	for(int i = 0; i< n_cores; ++i){

		std::advance(cBegin, -map_it_per_core.quot);


		cout<<"i "<<i<<" cBegin "<<cBegin->first<<" iter "<<iter->first;
		cout<<" what_to_name "<<what_to_name<<endl;
		//if(cBegin == data.end()){
		//	cBegin++;
		//	cout<<"i "<<i<<" cBegin "<<cBegin->first<<" iter "<<iter->first;
		//}
		if(iter == data.end())
			iter--;
		threads.push_back(std::thread(EmbedVect, cBegin, iter, std::ref(vect), what_to_name, m, d,skip));
		what_to_name = what_to_name + iter_per_core;
		std::advance(iter, -map_it_per_core.quot);
		cout<<"iter after advance "<<iter->first<<endl;
		//if(iter == data.end())
		//	break;

	}

	for(uint i = 0; i<threads.size(); ++i)
		{
		threads[i].join();
		cout<<"i "<<i<<endl;
		}




	return vect;
}

/*
 * @brief: performs time delay embedding
 *
 * @param data:
 * @param int m: number of embedding dimesnions
 * @param int d: time delay
 * @param bool skip: whether to skip the first value, this is used when when using same data but with different time delays
 *
 *
 *
 */
std::vector<std::vector<double> > Embed(std::multimap<double, std::vector<float>, classcomp >  &data ,int m, int d, bool skip){


	std::multimap<double, std::vector<float>,classcomp>::iterator cBegin = data.begin();
	std::multimap<double, std::vector<float>,classcomp>::iterator cEnd = data.end();
	std::multimap<double, std::vector<float>,classcomp>::iterator iter;
	cEnd--;

	int end = std::abs(m*d)-d+1;

	int vect_size = data.size() - end;
	std::vector<double > vect_data (vect_size);
	std::vector<std::vector<double> > vect(m-1);

	for(int i = 0; i< vect.size(); ++i)
		vect[i] = vect_data;

	int n_cores = std::thread::hardware_concurrency();

	//cout<<"modulus "<<std::modulus<double>((double)vect_size/(double)n_cores)<<endl;
	cout<<"modulus "<<vect_size/n_cores<<endl;
	cout<<"modulus "<<vect_size%n_cores<<endl;

	int iter_per_core = vect_size/n_cores;


	int vect_data_it =0;
	//boost::timer::auto_cpu_timer t;
	if(skip == false){
		for( ; cEnd!=cBegin; cEnd--){
			iter = cEnd;
			//cout<<std::distance(cBegin, cEnd)<<endl;

			for(int i =1; i<m; i++){
				std::advance(iter, -1*std::abs(d));

				//cEnd->second.insert(cEnd->second.begin(),iter->second[0]);
				vect[m-i-1][vect_data_it] = iter->second[0];
				//vect[m-i-1].insert(vect[m-i-1].begin(),iter->second[0]);



				//vect[m-i-1].push_back(iter->second[0]);
				//cout<<"cEnd "<<cEnd->first<<" ";
				//cout<<"cEnd "<<cEnd->second<<" iter "<<iter->second<<endl;
				//cout<<"cEnd "<<cEnd->second<<" iter->second[0] "<<iter->second[0]<<endl;
				//cout<<"cEnd "<<cEnd->second<<" iter->second[1] "<<iter->second[1]<<endl;

				//cout<<"vect[m-i-1] "<<m-i-1<<" values "<<vect[m-i-1]<<endl;
			}
			if(std::distance(cBegin, cEnd)<end){
				data.erase(cBegin, cEnd);
				break;
			}
			vect_data_it++;
			//if(vect_data_it>750)
			//	break;
		}
	}
	else{
		for( ; cEnd!=cBegin; cEnd--){
			iter = cEnd;
			//cout<<std::distance(cBegin, cEnd)<<endl;

			for(int i =1; i<m; i++){
				std::advance(iter, -1*std::abs(d));
				//cout<<"cEnd b4 "<<cEnd->second<<" iter "<<iter->second<<endl;
				cEnd->second.insert(cEnd->second.begin(),iter->second[0]);
			}
			cEnd->second.pop_back();

			if(std::distance(cBegin, cEnd)<end){
				data.erase(cBegin, cEnd);
				break;
			}
		}
	}
	//Print(data);

	for(int i = 0; i<vect.size(); ++i){
		vect[i].resize(vect_data_it);
		vect[i].shrink_to_fit();
	}


	return vect;
}

/*
 * @brief: Aligns Y with the corresponding embeded x values
 *
 * @param std::multimap<double, std::vector<float>, classcomp >  &data: map of time delayed x values
 *
 * @return std::vector<double> : a vector were the value is lined up with the corresponding time delayed x values
 *
 *
 */
std::vector<double> AlignedYvectWithEmbedX(std::multimap<double, std::vector<float>, classcomp >  &data,
		std::multimap<double, std::vector<float>, classcomp>::iterator cBegin, std::multimap<double, std::vector<float>,classcomp>::iterator cEnd){


	//std::multimap<double, std::vector<float>,classcomp>::iterator cBegin = data.begin();
	//std::multimap<double, std::vector<float>,classcomp>::iterator cEnd = data.end();
	std::multimap<double, std::vector<float>,classcomp>::iterator iter;

	int distance = std::distance(cBegin,cEnd);
	std::vector<double > vect_y;//(distance);//corresponds with embeded data were the y's should correspond with the last vector in vect and the last index
	//in data->second.size()-1

	//if(cEnd == data.end())
		cEnd--;
	int i = vect_y.size()-1;
	for(; cEnd!=cBegin; --cEnd,--i){
		//vect_y[i]=cEnd->first;//
		vect_y.push_back(cEnd->first);
		//--i;
	}

	return vect_y;

}


/*
 * @brief: uses a map to put data into groups, were each group can have multiple sets of variables associated with it. were one variable
 * represents an index to a set of strings, in short this creates a one hot encoding for that variable
 *
 * @tparam set<T> &set_2_convert: A set of variables that need to be converted
 * @tparam vector<vector<T1> &data: The input data that contains the variables
 * @tparam T2 &grouped: The container that we are putting the grouped by and converted into
 * @param int group_by_column: The column number of variable that we are grouping by
 * @param int set_column: The column number in the data vector for the data found in groups that we are converting
 * @param set<int> var_columns: The columns to be used as variables
 *
 */
template< typename T, typename T1, typename T2 >
void GroupBy(std::set<T> &set_2_conert,std::vector<std::vector<T1> > &data, T2 &grouped, int group_by_column, int set_column, set<int> var_columns){

	auto set_it = set_2_conert.begin();

	string group_by_string;

	decltype(grouped.begin()->second) tmp_vect;

	int     status;
	char   *realname;
	const std::type_info  &ti = typeid(tmp_vect.data());
	realname = abi::__cxa_demangle(ti.name(), 0, 0, &status);
	cout<<realname<<endl;
	string type = realname;

	int count=0;

	if(string(realname).find("double")!= std::string::npos){

		std::vector<double> tmp_vect(var_columns.size());

		//uint data_it = 0; data_it <data.size(); ++data_it
		for(auto data_it : data ){
			for(auto it = var_columns.begin(); it!= var_columns.end(); ++it){
				if(*it != group_by_column){
					if(*it == set_column)
						tmp_vect[count]=(double)std::distance(set_2_conert.begin(),set_2_conert.find(data_it[*it]));
					else
						tmp_vect[count] = std::stod(data_it[*it]);
				}
				count++;
			}
			grouped[data_it[group_by_column]].push_back(tmp_vect);
			count=0;
		}
	}




}
template void GroupBy<string, string, map<string, vector<vector<double> > > >(std::set<string> &, std::vector<std::vector<string > > &, map<string, vector<vector< double> > > &, int,int,set<int>);


/*
 * @brief: converts everything into a map<type, vector<double> >
 *
 */
template< typename T, typename T1 >
void GroupBy(T &data, T1 &grouped, int group_by_column, int value_column){


	string type;
	double value;

	for(uint data_it = 0; data_it <data.size(); ++data_it){
		type = data[data_it][group_by_column];
		value = std::stod(data[data_it][value_column]);
		grouped[data[data_it][group_by_column]].push_back(std::stod(data[data_it][value_column]));


	}



}
template void GroupBy<std::vector<std::vector<string > >, std::map<string, std::vector<double> > >( std::vector<std::vector<string > > &, std::map<string, std::vector<double> >  &, int,int);


/*
 * @brief: Converts column in a vector<vector<string> > to a single vector<float>
 *
 * @param vector<std::vector<string> >: &data_the data
 * @param int colum: column number of data to be converted to float
 *
 */
std::vector<float> VectorVectorStringToVectorFloat(std::vector<std::vector<string> > &data_in, int column){

	std::vector<float> tmp(data_in.size());

	for(int i = 0; i<data_in.size(); ++i)
		tmp[i]= std::stof(data_in[i][column]);

	return tmp;

}



}




















