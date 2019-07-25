/*
 * Champs.cpp
 *
 *  Created on: Jun 20, 2019
 *      Author: ryan
 */

#include <Champs.h>
//#include "cmd.h"


C10_DEFINE_int(iters, 4000, "The of training runs.");


Champs::Champs() {
	// TODO Auto-generated constructor stub

}

Champs::~Champs() {
	// TODO Auto-generated destructor stub
}

void Champs::ChampsMain(){
	//this->ChampsTrain();
	//this->ScalarCouplingContributions();
	this->MergeTrainAndStructures();
	this->BuildFeatureVector(this->train_map_,this->scalar_couplings_);
	this->BuildNet();
	this->BuildFeatureVector(this->test_map_,this->test_ids_);
	this->PredictAll();
}
void Champs::ChampsTrain(){


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

		cout<<it->first<<" ";//endl;//" "<<it->second[0]<<endl;
		Stats::Histogram hist_st;
		Stats::CalcSimpleStats(it->second,hist_st,14);

		Stats::PlotHist(hist_st,this->file_path_, it->first+"_hist");

	}

}

void Champs::ScalarCouplingContributions(){


	//Parse(this->filename_,600000);
	//Parse(this->structures_filename_,this->atom_set_,2);
	auto scalar_couple_contrib = Parse(this->scalar_couple_contrib_filename_);
	std::vector<double> type_grouped;

	//MyData::GroupBy(this->type_set_,train,type_grouped, 4,5);

	auto tmp = MyData::VectorVectorStringToVectorFloat(scalar_couple_contrib, 5);

	//simple stats and distribution of scalar coupling
	Stats::Histogram hist_st;
	Stats::CalcSimpleStats(tmp,hist_st,14);
	Stats::PlotHist(hist_st,this->file_path_, "scalar_couple_contrib_hist");

	std::map<string, int> var_map;

	var_map["fc"] = 4;
	var_map["sd"] = 5;
	var_map["pso"] = 6;
	var_map["dso"] = 7;

	for(auto var_map_it = var_map.begin(); var_map_it!=var_map.end(); ++var_map_it){
		//simple stats distribution of scalar coupling by type
		std::map<string, std::vector<double> > type_to_scalar_couple_contrib;
		MyData::GroupBy(scalar_couple_contrib, type_to_scalar_couple_contrib,3,var_map_it->second);
		std::map<string, std::vector<double> >::iterator it;
		for(it = type_to_scalar_couple_contrib.begin(); it!= type_to_scalar_couple_contrib.end(); ++it){

			cout<<it->first<<" "<<var_map_it->first<<" ";//endl;//" "<<it->second[0]<<endl;
			Stats::Histogram hist_st;
			Stats::CalcSimpleStats(it->second,hist_st,14);

			Stats::PlotHist(hist_st,this->file_path_, it->first+"_"+var_map_it->first+"_scalar_couple_contrib_hist");

		}
	}

}

void Champs::MergeTrainAndStructures(){


	auto structures = Parse(this->structures_filename_, this->atom_set_,2);
	auto train = Parse(this->train_filename_,this->type_set_,4);
	auto test = Parse(this->test_filename_,this->type_set_,4);
	std::vector<double> type_grouped;

	set<int> structures_var_columns({2,3,4,5});
	MyData::GroupBy(this->atom_set_, structures,this->stuctures_map_, 0,2,structures_var_columns);
	auto tmp = MyData::VectorVectorStringToVectorFloat(train, 5);

	//simple stats distribution of scalar coupling by type
	set<int> train_var_columns({2,3,4,5});
	//std::map<string, std::vector<double> > type_to_scalar_constants;
	MyData::GroupBy(this->type_set_,train,this->train_map_,1,4,train_var_columns);

	//simple stats distribution of scalar coupling by type
	std::vector<int> test_var_columns({2,3,4,0});
	//std::map<string, std::vector<double> > type_to_scalar_constants;
	MyData::GroupBy(this->type_set_,test,this->test_map_,1,4,test_var_columns);


}

/*
 *
 * not_a_feature is used for Scalar coupling and ID of test set or what ever the target is
 */
void Champs::BuildFeatureVector(std::map<string, std::vector<std::vector<double> > > &data_map, std::vector<float> &not_a_feature ){


	std::vector<double> atom_hot(this->atom_set_.size());

	//2*atom_set_ Because there are 2 atoms in the bond
	//type_set type of bonding
	//3 x,y,z coordinates
	int atom_set_size = this->atom_set_.size();
	int feature_vector_length = 2*atom_set_size+this->type_set_.size()+6;
	vector<float> features(feature_vector_length+1);
	float distance = 0.00000;
	float distance_tmp = 0.00000;
	float atom_0=0.00000;
	float atom_1=0.00000;

	std::map<string, std::vector<std::vector<double> > >::iterator data_map_it =  data_map.begin();//this->train_map_.begin();
	this->features_.clear();
	this->features_.shrink_to_fit();


	//std::vector<std::vector<double> >::iterator vector_it;

	int atom_idx_0 = 0;
	int atom_idx_1 = 1;
	int type_idx = 2;

	for(; data_map_it != data_map.end(); ++data_map_it){
		for(auto vector_it : data_map_it->second)/*for(vector_it = data_map_it->second.begin(); vector_it != data_map_it->second.end(); ++vector_it)*/{

			features[this->stuctures_map_[data_map_it->first][vector_it[atom_idx_0]][0]]=1;
			features[this->stuctures_map_[data_map_it->first][vector_it[atom_idx_1]][0]+atom_set_size]=1;
			features[vector_it[type_idx]+2*atom_set_size]=1;

			/*//positions of atom
			for(uint i = 1; i<this->stuctures_map_[data_map_it->first][vector_it[atom_idx_0]].size(); ++i)
				features[features.size()-6+i-1]=this->stuctures_map_[data_map_it->first][vector_it[atom_idx_0]][i];

			//position of atom
			for(uint i = 1; i<this->stuctures_map_[data_map_it->first][vector_it[atom_idx_1]].size(); ++i)
				features[features.size()-3+i-1]=this->stuctures_map_[data_map_it->first][vector_it[atom_idx_1]][i];*/

			//positions of atom
			for(uint i = 1; i<this->stuctures_map_[data_map_it->first][vector_it[atom_idx_0]].size(); ++i){
				features[features.size()-6+i-1]=this->stuctures_map_[data_map_it->first][vector_it[atom_idx_0]][i];
				features[features.size()-3+i-1]=this->stuctures_map_[data_map_it->first][vector_it[atom_idx_1]][i];
				atom_0=this->stuctures_map_[data_map_it->first][vector_it[atom_idx_0]][i];
				atom_1=this->stuctures_map_[data_map_it->first][vector_it[atom_idx_1]][i];

				//calculate distance
				//cout<<"distance "<<distance<<" "<<distance_tmp<<endl;
				//distance += (features[features.size()-6+i-1] - features[features.size()-3+i-1])*(features[features.size()-6+i-1] - features[features.size()-3+i-1]);
				distance += (atom_0-atom_1)*(atom_0-atom_1);

			}


			distance = ::sqrtf(distance);

			if(features[0]>0)
				cout<<"stop"<<endl;

			features[0] = distance;

			not_a_feature.push_back(vector_it[vector_it.size()-1]);//this->scalar_couplings_.push_back(vector_it[vector_it.size()-1]);
			this->features_.push_back(features);

			features[0] = 0;

			//reset atom and type variables
			features[this->stuctures_map_[data_map_it->first][vector_it[atom_idx_0]][0]]=0;
			features[this->stuctures_map_[data_map_it->first][vector_it[atom_idx_1]][0]+atom_set_size]=0;
			features[vector_it[type_idx]+2*atom_set_size]=0;
			features[0] = 0;
			distance = 0;
			//distance_tmp = 0;
		}

	}


}

void Champs::BuildNet() {

	int n_features = this->features_[0].size();
	int n_hide = 5;
	float base_learning_rate = -.0002998;
	int batch_size = 875;
	int classes = 1;

	std::cout << "Start training" << std::endl;
	string model_name = this->model_name;
	string init_model_name = this->init_model_name;

	string model_path = this->file_path_;

	// >>> model = model_helper.ModelHelper(name="char_rnn")
	caffe2::NetDef init_model, predict_model;
	//caffe2::ModelUtil model(init_model, predict_model, model_name);
	//ChampsNet ChampsNet(init_model, predict_model, model_name);
	ChampsNet model(init_model, predict_model, model_name);







/*
flow is
1. input
2. FC
3. activation-> RELU, tanh, sigmoid, softmax
4. repeat 2 and 3 for total number of layers
5. cost funtion(measure of error rate, or the total loss over all the examples) loss function-> MSE RMSE,
*/




	//set activation       model.predict.AddTanhOp("fourth_layer", "tanh3");
	//std::vector<string > layer({"1"});
	vector<int > n_nodes_per_layer({8,16,8,2,3,2,3,5,1,2,4});
	string layer_name = " ";
	string activation = "LeakyRelu";//"Tanh";//
	string layer_in_name = " ";
	string layer_out_name = " ";

	model.predict.AddInput("input_blob");
		model.predict.AddInput(activation);
		model.predict.AddInput("target");
		//model.predict.AddInput("accuracy");
		model.predict.AddInput("loss");

		//Add layer, inputs are model to add, name of layer coming in, name of layer going out(i.e. name of this layer??)
		//number of neurons in this layer,  number of neurons in layer is connection to
		//think FC does add(matmul(inputs*w,b))

		model.predict.AddStopGradientOp("input_blob");

	for(int i =0; i< n_nodes_per_layer.size(); ++i){
		layer_name = std::to_string((i));
		if(i == 0)
			model.AddFcOps("input_blob", layer_name, n_features, n_nodes_per_layer[i]);
		else
			model.AddFcOps(activation+std::to_string(i),layer_name,n_nodes_per_layer[i-1], n_nodes_per_layer[i]);
		if(activation == "LeakyRelu")
			model.predict.AddLeakyReluOp(layer_name,activation+std::to_string(i+1),.3);//model.predict.AddSumOp(what, "sum");
		else if(activation == "Tanh")
			model.predict.AddTanhOp(layer_name,activation+std::to_string(i+1));

		//cout<<"layer_name "<<layer_name<<" activation+std::to_string(i+1) "<<activation+std::to_string(i+1)<<endl;

	}
	//layer_name = activation+std::to_string(n_nodes_per_layer.size());
	layer_name = "last_layer";//"last_layer";//std::to_string(n_nodes_per_layer.size());
	//cout<<activation+std::to_string(n_nodes_per_layer.size())<<endl;
	model.AddFcOps(activation+std::to_string(n_nodes_per_layer.size()),layer_name,n_nodes_per_layer[n_nodes_per_layer.size()-1], classes);








	//this adds the learning rate opp
	//model.predict.AddConstantFillOp({1},base_learning_rate, "lr");


	//model.predict.AddConstantFillWithOp(1,"sum","loss");
	model.init.AddConstantFillOp({1},0.f,"loss");//model.predict.AddConstantFillOp({1},0.f,"loss");

	//had to add this so I could usev train.AddSgdOps();
	model.init.AddConstantFillOp({1},0.f,"one");//model.predict.AddConstantFillOp({1},0.f,"loss");

	//model.init.AddConstantFillWithOp(1.f, "loss", "loss_grad");
	//set loss
	//model.predict.AddSquaredL2Op(layer_name,"target","sql2");
	model.predict.AddL1DistanceOp(layer_name,"target","sql2");

	//model.predict.net.A
	model.predict.AddAveragedLossOp("sql2", "loss");


	model.AddIterOps();

	caffe2::NetDef f_int = model.init.net;
	caffe2::NetDef pred = model.predict.net;
	caffe2::ModelUtil save_model(f_int, pred, model_name);


	//cout<<model.predict.net.DebugString()<<endl;
/*	caffe2::NetDef train_model(model.predict.net);
	caffe2::NetUtil train(train_model, "train");*/
	caffe2::NetDef train_init, train_predict;
	caffe2::ModelUtil train(train_init, train_predict,"train");
	string su = "relu";
	model.CopyTrain(layer_name, 1,train);

	//train.predict.AddInput("iter");//this should some how get done in void ModelUtil::CopyTrain(const std::string &layer, int out_size,ModelUtil &train)
	//train.predict.AddInput("one");//this should some how get done in void ModelUtil::CopyTrain(const std::string &layer, int out_size,ModelUtil &train)
	train.predict.AddConstantFillWithOp(1.f, "loss", "loss_grad");

	//set optimizer
	//model.AddAdamOps();
	//model.AddRmsPropOps();
	train.predict.AddGradientOps();
	base_learning_rate = -1*base_learning_rate;
	train.predict.AddLearningRateOp("iter","lr",base_learning_rate,.9);
	train.AddSgdOps();
	//train.AddRmsPropOps();






	auto predictions = "softmax";

	//I think prepare just copies the values
	caffe2::NetDef prepare_model;
	caffe2::NetUtil prepare(prepare_model, "prepare_state");
	prepare.AddInput("input_blob");
	prepare.AddInput("first_layer");
	//prepare.AddInput("second_layer");
	//prepare.AddInput("last_layer");
	//prepare.AddInput("loss");
/*	prepare.AddInput("argmax");
	prepare.AddInput("accuracy");
	prepare.AddInput("softmax");
	prepare.AddInput("target");*/



	/*cout<<model.init.Proto()<<endl;
	cout<<endl;
	cout<<model.predict.Proto()<<endl;
	cout<<endl;
	cout<<train.init.Proto()<<endl;
	cout<<endl;
	cout<<train.predict.Proto()<<endl;*/
	//Start training
	caffe2::Workspace workspace("tmp");

	// >>> log.debug("Training model")
	std::cout << "Train model" << std::endl;

	// >>> workspace.RunNetOnce(self.model.param_init_net)
	CAFFE_ENFORCE(workspace.RunNetOnce(model.init.net));

	auto epoch = 0;

	// >>> workspace.CreateNet(self.prepare_state)
	//CAFFE_ENFORCE(workspace.CreateNet(prepare.net));

	//std::cout<<"train.net.de "<<train.net.DebugString()<<std::endl;

	// >>> CreateNetOnce(self.model.net)
	workspace.CreateBlob("input_blob");
	//workspace.CreateBlob("accuracy");
	workspace.CreateBlob("loss");
	workspace.CreateBlob("target");
	//workspace.CreateBlob("one");

	workspace.CreateBlob("lr");

	CAFFE_ENFORCE(workspace.RunNetOnce(train.init.net));
	CAFFE_ENFORCE(workspace.CreateNet(train.predict.net));//CAFFE_ENFORCE(workspace.CreateNet(train.net));

	cout<<train.init.net.name()<<" "<<model.init.net.name()<<" "<<prepare.net.name()<<endl;
	// >>> CreateNetOnce(self.forward_net)
	CAFFE_ENFORCE(workspace.CreateNet(model.predict.net));//CAFFE_ENFORCE(workspace.CreateNet(model.predict.net));

	float wFbefore = caffe2::BlobUtil(*workspace.GetBlob("2_w")).Get().data<float>()[0];
	float wFafter = 0.00000000;
//	float wSefore = caffe2::BlobUtil(*workspace.GetBlob("1_w")).Get().data<float>()[0];
	//float wLefore = BlobUtil(*workspace.GetBlob("last_layer_w")).Get().data<float>()[0];

	int nTrainBatches = batch_size;//FLAGS_batch;//TrainData.Features.size()/FLAGS_batch;

	//CAFFE_ENFORCE(workspace.RunNet(prepare.net.name()));
	std::vector<float> tmp;

	//compute number of minibatches for training, validation and testing
	int n_train_batches = 100;//TrainData.Features.size() / batch_size;//.get_value(borrow=True).shape[0] // batch_size
	int n_valid_batches = 100; // ValidateData.Features.size() / batch_size;//#get_value(borrow=True).shape[0] // batch_size
	int n_test_batches = 100;//TestData.Features.size() / batch_size;//#get_value(borrow=True).shape[0] // batch_size

	//early-stopping parameters
	int patience = 4000;//  # look as this many examples regardless
	int patience_increase = 4;//  # wait this much longer when a new best is found
	float improvement_threshold = 0.595; //a relative improvement of this much is
	//considered significant
	int validation_frequency = std::min(n_train_batches, patience / 2);

	int iter= 0;

	bool done_looping = false;
	int server = 0;

	std::vector<float> validation_losses;
	float this_validation_loss =0.0;
	float best_validation_loss = std::numeric_limits<float>::max();
	int best_iter =0;
	float test_score =0.0;
	float train_score = 0.0;

	//do I need this here or not??
	//CAFFE_ENFORCE(workspace.RunNet(prepare.net.name()));

	vector<float>tmp_w(n_features);
	vector<vector<float> > check_weights_w;

	//while (epoch < n_epochs) and (not done_looping):
	while (epoch < FLAGS_iters && !done_looping) {

		epoch++;//this for like total number of iterations

		// >>> workspace.RunNet(self.prepare_state.Name())
		//CAFFE_ENFORCE(workspace.RunNet(prepare.net.name()));

		//Train
		for(auto minibatch_index = 0; minibatch_index<(n_train_batches); ++minibatch_index ){

			//CAFFE_ENFORCE(workspace.RunNet(prepare.net.name()));

			{

				//std::vector<int> dim({nTrainBatches,TrainData.Features[minibatch_index].size()});
				std::vector<int> dim({nTrainBatches,this->features_[minibatch_index].size()});
				//BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, TrainData.Features, minibatch_index, false);
				caffe2::BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, this->features_, minibatch_index, false);
			}

			{
				std::vector<int> dim({nTrainBatches,1});
				//BlobUtil(*workspace.CreateBlob("target")).Set(dim, TrainData.Labels, minibatch_index, false);
				caffe2::BlobUtil(*workspace.CreateBlob("target")).Set(dim, this->scalar_couplings_, minibatch_index, false);
				//std::cout<<"Train Label "<<BlobUtil(*workspace.GetBlob("target")).Get().data<int>()[0]<<std::endl;

				//std::cout<<"Train Label "<<caffe2::BlobUtil(*workspace.GetBlob("target")).Get().DebugString()<<std::endl;

			}


			CAFFE_ENFORCE(workspace.RunNet(train.predict.net.name()));//CAFFE_ENFORCE(workspace.RunNet(train.net.name()));

/*			cout<<caffe2::BlobUtil(*workspace.GetBlob(layer_name)).Get().DebugString()<<endl;//data<float>()[0];
			for(int i = 0; i<nTrainBatches; ++i)
				cout<<"layr_name data "<<caffe2::BlobUtil(*workspace.GetBlob(layer_name)).Get().data<float>()[i]<<endl;

			for(int i =0; i<n_features; ++i)
					tmp_w[i] = caffe2::BlobUtil(*workspace.GetBlob("0_w")).Get().data<float>()[i];//cout<<caffe2::BlobUtil(*workspace.GetBlob("0_w")).Get().data<float>()[i]<<" ";
				check_weights_w.push_back(tmp_w);
				cout<<endl;*/
			//cout<<"wFbefore "<<wFbefore <<" After "<< caffe2::BlobUtil(*workspace.GetBlob("0_w")).Get().data<float>()[0]<<endl;
//			cout<<"wSefore "<<wSefore<<" After "<< caffe2::BlobUtil(*workspace.GetBlob("1_w")).Get().data<float>()[0]<<endl;
			//	cout<<"wLefore "<<wLefore<<" After "<< BlobUtil(*workspace.GetBlob("last_layer_w")).Get().data<float>()[0]<<endl;



			//train_score = 1-caffe2::BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];// cout<<"Train Score "<<train_score<<endl;
			train_score = caffe2::BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];// cout<<"Train Score "<<train_score<<endl;
			//cout<<"Train accuracy "<<BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0]<<endl;

			//cout<<"train_score "<<train_score<<endl;
			//#print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))

			iter = (epoch - 1) * n_train_batches + minibatch_index;

			if((iter + 1) % validation_frequency == 0){
				//for(auto validate_index = 0; validate_index<(n_valid_batches+e); ++validate_index ){

				//cout<<"optimizer_iteration "<<BlobUtil(*workspace.GetBlob("optimizer_iteration")).Get() <<endl;

				//validating and tessting here
				//Does prepeare build the prediction net ???????
				//CAFFE_ENFORCE(workspace.RunNet(prepare.net.name()));
				// >>> workspace.FeedBlob("input_blob", input)

				{
					//std::vector<int> dim({nTrainBatches,ValidateData.Features[minibatch_index].size()});
					std::vector<int> dim({nTrainBatches,this->features_[minibatch_index].size()});
					//BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, ValidateData.Features, minibatch_index,false);
					caffe2::BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, this->features_, minibatch_index,false);
				}

				{
					std::vector<int> dim({nTrainBatches,1});
					//BlobUtil(*workspace.CreateBlob("target")).Set(dim, ValidateData.Labels,minibatch_index, false);
					caffe2::BlobUtil(*workspace.CreateBlob("target")).Set(dim, this->scalar_couplings_,minibatch_index, false);
					//std::cout<<"Validate Label "<<BlobUtil(*workspace.GetBlob("target")).Get().data<int>()[0]<<std::endl;
				}


				// >>> workspace.RunNet(self.forward_net.Name())
				CAFFE_ENFORCE(workspace.RunNet(model.predict.net.name()));

				//cout<<"Accuracy "<<BlobUtil(*workspace.GetBlob("accuracy")).Get() <<endl;


				//cout<<"Validate accuracy "<<1.0-BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0]<<endl;

				//this_validation_loss =1.0-caffe2::BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];
				this_validation_loss =caffe2::BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];
				//cout<<"Valied Score "<<this_validation_loss<<endl;// 1-(float)(nCorrect)/(float)(countValue);//sess.run([accuracy], feed_dict={X: batch_x, Y: batch_y})[0]#numpy.mean(validation_losses)
				//cout<<"Validate Percent Correct "<<(float)(nCorrect)/(float)(countValue)<<endl;


				//#print('epoch %i, minibatch %i/%i, validation error %f %%' %
				//#     (epoch, minibatch_index + 1, n_train_batches,
				//#      this_validation_loss * 100.))

				validation_losses.push_back(this_validation_loss);
				// if we got the best validation score until now
				if(this_validation_loss < best_validation_loss){
					//improve patience if loss improvement is good enough
					if( this_validation_loss < best_validation_loss * improvement_threshold)
						patience = max(patience, iter * patience_increase);

					best_validation_loss = this_validation_loss;
					best_iter = iter;

					//cout<<"optimizer_iteration "<<BlobUtil(*workspace.GetBlob("optimizer_iteration")).Get() <<endl;

					//validating and tessting here
					//Does prepeare build the prediction net ???????
					//CAFFE_ENFORCE(workspace.RunNet(prepare.net.name()));
					// >>> workspace.FeedBlob("input_blob", input)

					{
						//std::vector<int> dim({nTrainBatches,TestData.Features[minibatch_index].size()});
						std::vector<int> dim({nTrainBatches,this->features_[minibatch_index].size()});
						//BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, TestData.Features,minibatch_index, false);
						caffe2::BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, this->features_,minibatch_index, false);
					}

					{
						std::vector<int> dim({nTrainBatches,1});
						//BlobUtil(*workspace.CreateBlob("target")).Set(dim, TestData.Labels, minibatch_index, false);
						caffe2::BlobUtil(*workspace.CreateBlob("target")).Set(dim, this->scalar_couplings_, minibatch_index, false);
						//std::cout<<"Test Label "<<BlobUtil(*workspace.GetBlob("target")).Get().data<int>()[0]<<std::endl;
					}

					// >>> workspace.RunNet(self.forward_net.Name())
					CAFFE_ENFORCE(workspace.RunNet(model.predict.net.name()));

					//cout<<"Accuracy "<<BlobUtil(*workspace.GetBlob("accuracy")).Get() <<endl;

					test_score = caffe2::BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];
					//cout<<"Test Score "<<test_score<<endl;//(float)(nCorrect)/(float)(countValue);

					cout<<"Train "<<train_score<< " Validate "<<this_validation_loss<<" Test "<<test_score<<endl;

					cout<<"wFbefore "<<wFbefore*10000 <<" After "<< caffe2::BlobUtil(*workspace.GetBlob("2_w")).Get().data<float>()[0]*10000<<endl;
					//cout<<"Test accuracy "<<1.0-BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0]<<endl;

					caffe2::NetDef deploy_init_model;  // the final initialization model
					caffe2::ModelUtil deploy(deploy_init_model, save_model.predict.net,model.init.net.name());
					//caffe2::ModelUtil deploy(deploy_init_model, model.predict.net,model.init.net.name());


					save_model.CopyDeploy(deploy, workspace);
					//CAFFE_ENFORCE(workspace.RunNet(model.predict.net.name()));
					if(server == 0){
						//cout<<"Save Model"<<endl;//mdir=saver.save(sess, save_path='/home/ryan/workspace/MultiLayerTen/'+modelName, global_step=tf.train.global_step(sess,global_step_tensor))
						caffe2::WriteProtoToTextFile(deploy.init.net, model_path+init_model_name+".pbtxt");//caffe2::WriteProtoToBinaryFile(deploy.init.net, model_path+init_model_name);//
						caffe2::WriteProtoToTextFile(model.predict.net, model_path+ model_name+".pbtxt");//caffe2::WriteProtoToBinaryFile(model.predict.net, model_path+ model_name);//

					}
					else{
						//cout<<"Save Model"<<endl;//mdir=saver.save(sess, save_path='/home/riley/data/HE/HE1-2Models/'+modelName, global_step=tf.train.global_step(sess,global_step_tensor))
						caffe2::WriteProtoToTextFile(model.init.net, "/home/ryan/workspace/adsf/initModel");
						caffe2::WriteProtoToTextFile(model.predict.net, "/home/ryan/workspace/adsf/model1");
					}
					//save the best model

				}

			}


			if(patience <= iter){
				done_looping = true;
				break;
			}

		}


		//std::cout << "Smooth loss: " << smooth_loss << std::endl;
		//std::cout<<"last_layer_w After Training "<<BlobUtil(*workspace.GetBlob("last_layer_w")).Get().data<float>()[0]<<std::endl;
	}

	//cout<<"Optimization complete. Best validation score of "<<best_validation_loss<< " obtained at iteration "<<best_iter + 1<<
	//		" with test performance "<<test_score<<endl;


	//cout<<"wFbefore "<<wFbefore <<" After "<< BlobUtil(*workspace.GetBlob("first_layer_w")).Get().data<float>()[0]<<endl;
	//cout<<"wSefore "<<wSefore<<" After "<< BlobUtil(*workspace.GetBlob("second_layer_w")).Get().data<float>()[0]<<endl;
	//cout<<"wLefore "<<wLefore<<" After "<< BlobUtil(*workspace.GetBlob("last_layer_w")).Get().data<float>()[0]<<endl;
}

void Champs::PredictAll(){

	string model_path = this->file_path_;
		string model_name = this->model_name+".pbtxt";
			string init_model_name = this->init_model_name+".pbtxt";



		caffe2::NetDef init_model, predict_model;

		CAFFE_ENFORCE(caffe2::ReadProtoFromTextFile(model_path+init_model_name, &init_model));
		CAFFE_ENFORCE(caffe2::ReadProtoFromTextFile(model_path+model_name, &predict_model));



	caffe2::Workspace workspace("tmp");

	std::multimap<boost::posix_time::ptime, long > data;
	float pred_value = 0;
	//CAFFE_ENFORCE(workspace.RunNet(initModel.n));
//	int nTrainBatches = AllData.Features.size();//233;
	int minibatch_index = 0;
	int nTrainBatches = this->features_.size();
	vector<vector<float> > final_test(nTrainBatches);
	vector<float> final_test_tmp(2);

/*	{
		std::vector<int> dim({nTrainBatches,AllData.Features[minibatch_index].size()});
		BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, AllData.Features, minibatch_index,false);
	}

	{
		std::vector<int> dim({nTrainBatches});
		BlobUtil(*workspace.CreateBlob("target")).Set(dim, AllData.Labels,minibatch_index, false);
		//std::cout<<"Validate Label "<<BlobUtil(*workspace.GetBlob("target")).Get().data<int>()[0]<<std::endl;
	}*/

	{
						//std::vector<int> dim({nTrainBatches,ValidateData.Features[minibatch_index].size()});
						std::vector<int> dim({nTrainBatches,this->features_[minibatch_index].size()});
						//BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, ValidateData.Features, minibatch_index,false);
						caffe2::BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, this->features_, minibatch_index,false);
					}

					{
						std::vector<int> dim({nTrainBatches,1});
						//BlobUtil(*workspace.CreateBlob("target")).Set(dim, ValidateData.Labels,minibatch_index, false);
						caffe2::BlobUtil(*workspace.CreateBlob("target")).Set(dim, this->scalar_couplings_,minibatch_index, false);
						//std::cout<<"Validate Label "<<BlobUtil(*workspace.GetBlob("target")).Get().data<int>()[0]<<std::endl;
					}

					//workspace.CreateBlob("target");
	caffe2::ModelUtil model(init_model, predict_model, model_name);
	CAFFE_ENFORCE(workspace.CreateNet(model.init.net));
	CAFFE_ENFORCE(workspace.CreateNet(model.predict.net));


	CAFFE_ENFORCE(workspace.RunNet(model.init.net.name()));
	CAFFE_ENFORCE(workspace.RunNet(model.predict.net.name()));

	//cout<<"Accuracy "<<BlobUtil(*workspace.GetBlob("accuracy")).Get() <<endl;

//	Results.allResults = BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];

	boost::posix_time::ptime date;

	float max = -std::numeric_limits<float>::max();


	for(int a = 0; a<nTrainBatches; a++){

		pred_value = caffe2::BlobUtil(*workspace.GetBlob("last_layer")).Get().data<float>()[a];
		final_test_tmp[0] = this->test_ids_[a];
		final_test_tmp[1] = pred_value;
		final_test[a] = final_test_tmp;
		if(pred_value>max){
			max=pred_value;
			cout<<"Max "<<max<<endl;
		}

	}


	string test_file = this->file_path_+"test_submission.txt";
	this->SaveDataFirstSavedAsInt(test_file, final_test);



}

void Champs::BuildNet2(){


	std::cout << std::endl;
	std::cout << "## Caffe2 Toy Regression Tutorial ##" << std::endl;
	std::cout << "https://caffe2.ai/docs/tutorial-toy-regression.html"
			<< std::endl;
	std::cout << std::endl;

	using namespace caffe2;

	int n_features = this->features_[0].size();
	int n_hide = 5;
	float base_learning_rate = -.001;
	int batch_size = 175;
	int classes = 1;

	std::cout << "Start training" << std::endl;
	string model_name = "test_name";
	string init_model_name = "init"+model_name;

	string model_path = this->file_path_;

	// >>> from caffe2.python import core, cnn, net_drawer, workspace, visualize
	Workspace workspace;

	// >>> init_net = core.Net("init")
	NetDef initModel;
	initModel.set_name("init");

	// >>> W_gt = init_net.GivenTensorFill([], "W_gt", shape=[1, 2],
	/*	  // values=[2.0, 1.5])
	  {
	    auto op = initModel.add_op();
	    op->set_type("GivenTensorFill");
	    auto arg1 = op->add_arg();
	    arg1->set_name("shape");
	    arg1->add_ints(1);
	    arg1->add_ints(2);
	    auto arg2 = op->add_arg();
	    arg2->set_name("values");
	    arg2->add_floats(2.0);
	    arg2->add_floats(1.5);
	    op->add_output("W_gt");
	  }

	  // >>> B_gt = init_net.GivenTensorFill([], "B_gt", shape=[1], values=[0.5])
	  {
	    auto op = initModel.add_op();
	    op->set_type("GivenTensorFill");
	    auto arg1 = op->add_arg();
	    arg1->set_name("shape");
	    arg1->add_ints(1);
	    auto arg2 = op->add_arg();
	    arg2->set_name("values");
	    arg2->add_floats(0.5);
	    op->add_output("B_gt");
	  }*/

	// >>> ONE = init_net.ConstantFill([], "ONE", shape=[1], value=1.)
	{
		auto op = initModel.add_op();
		op->set_type("ConstantFill");
		auto arg1 = op->add_arg();
		arg1->set_name("shape");
		arg1->add_ints(1);
		auto arg2 = op->add_arg();
		arg2->set_name("value");
		arg2->set_f(1.0);
		op->add_output("ONE");
	}

	// >>> ITER = init_net.ConstantFill([], "ITER", shape=[1], value=0,
	// dtype=core.DataType.INT32)
	{
		auto op = initModel.add_op();
		op->set_type("ConstantFill");
		auto arg1 = op->add_arg();
		arg1->set_name("shape");
		arg1->add_ints(1);
		auto arg2 = op->add_arg();
		arg2->set_name("value");
		arg2->set_i(0);
		auto arg3 = op->add_arg();
		arg3->set_name("dtype");
		arg3->set_i(TensorProto_DataType_INT32);
		op->add_output("ITER");
	}

	// >>> W = init_net.UniformFill([], "W", shape=[1, 2], min=-1., max=1.)
	{
		auto op = initModel.add_op();
		op->set_type("UniformFill");
		auto arg1 = op->add_arg();
		arg1->set_name("shape");
		arg1->add_ints(1);
		arg1->add_ints(24);
		auto arg2 = op->add_arg();
		arg2->set_name("min");
		arg2->set_f(-1);
		auto arg3 = op->add_arg();
		arg3->set_name("max");
		arg3->set_f(1);
		op->add_output("W");
	}

	// >>> B = init_net.ConstantFill([], "B", shape=[1], value=0.0)
	{
		auto op = initModel.add_op();
		op->set_type("ConstantFill");
		auto arg1 = op->add_arg();
		arg1->set_name("shape");
		arg1->add_ints(1);
		auto arg2 = op->add_arg();
		arg2->set_name("value");
		arg2->set_f(0);
		op->add_output("B");
	}

	// print(initModel);


	// >>> train_net = core.Net("train")
	NetDef trainModel;
	trainModel.set_name("train");

	trainModel.add_external_input("ONE");
	trainModel.add_external_input("ITER");
	trainModel.add_external_input("X");
	trainModel.add_external_input("Y_noise");
	trainModel.add_external_input("W");
	trainModel.add_external_input("B");
	//generates ground truth X
	/*
	  // >>> X = train_net.GaussianFill([], "X", shape=[64, 2], mean=0.0, std=1.0,
	  // run_once=0)
	  {
	    auto op = trainModel.add_op();
	    op->set_type("GaussianFill");
	    auto arg1 = op->add_arg();
	    arg1->set_name("shape");
	    arg1->add_ints(64);
	    arg1->add_ints(2);
	    auto arg2 = op->add_arg();
	    arg2->set_name("mean");
	    arg2->set_f(0);
	    auto arg3 = op->add_arg();
	    arg3->set_name("std");
	    arg3->set_f(1);
	    auto arg4 = op->add_arg();
	    arg4->set_name("run_once");
	    arg4->set_i(0);
	    op->add_output("X");
	  }*/

	//generates ground truth Y
	/*
	  // >>> Y_gt = X.FC([W_gt, B_gt], "Y_gt")
	  {
	    auto op = trainModel.add_op();
	    op->set_type("FC");
	    op->add_input("X");
	    op->add_input("W_gt");
	    op->add_input("B_gt");
	    op->add_output("Y_gt");
	  }
	 */

	//Just creates noise to add to Y_gt
	/*	  // >>> noise = train_net.GaussianFill([], "noise", shape=[64, 1], mean=0.0,
	  // std=1.0, run_once=0)
	  {
	    auto op = trainModel.add_op();
	    op->set_type("GaussianFill");
	    auto arg1 = op->add_arg();
	    arg1->set_name("shape");
	    arg1->add_ints(64);
	    arg1->add_ints(1);
	    auto arg2 = op->add_arg();
	    arg2->set_name("mean");
	    arg2->set_f(0);
	    auto arg3 = op->add_arg();
	    arg3->set_name("std");
	    arg3->set_f(1);
	    auto arg4 = op->add_arg();
	    arg4->set_name("run_once");
	    arg4->set_i(0);
	    op->add_output("noise");
	  }*/

	//adds the nois to Y_gt
	/*	  // >>> Y_noise = Y_gt.Add(noise, "Y_noise")
	  {
	    auto op = trainModel.add_op();
	    op->set_type("Add");
	    op->add_input("Y_gt");
	    op->add_input("noise");
	    op->add_output("Y_noise");
	  }*/

	/*	  // >>> Y_noise = Y_noise.StopGradient([], "Y_noise")
	  {
	    auto op = trainModel.add_op();
	    op->set_type("StopGradient");
	    op->add_input("Y_noise");
	    op->add_output("Y_noise");
	  }*/

	std::vector<OperatorDef *> gradient_ops;




	// >>> Y_pred = X.FC([W, B], "Y_pred")
	{
		auto op = trainModel.add_op();
		op->set_type("FC");
		op->add_input("X");
		op->add_input("W");
		op->add_input("B");
		op->add_output("Y_pred");
		gradient_ops.push_back(op);
	}

	// >>> dist = train_net.SquaredL2Distance([Y_noise, Y_pred], "dist")
	{
		auto op = trainModel.add_op();
		op->set_type("SquaredL2Distance");
		op->add_input("Y_noise");
		op->add_input("Y_pred");
		op->add_output("dist");
		gradient_ops.push_back(op);
	}

	// >>> loss = dist.AveragedLoss([], ["loss"])
	{
		auto op = trainModel.add_op();
		op->set_type("AveragedLoss");
		op->add_input("dist");
		op->add_output("loss");
		gradient_ops.push_back(op);
	}

	// >>> gradient_map = train_net.AddGradientOperators([loss])
	{
		auto op = trainModel.add_op();
		op->set_type("ConstantFill");
		auto arg = op->add_arg();
		arg->set_name("value");
		arg->set_f(1.0);
		op->add_input("loss");
		op->add_output("loss_grad");
		op->set_is_gradient_op(true);
	}
	std::reverse(gradient_ops.begin(), gradient_ops.end());
	for (auto op : gradient_ops) {
		vector<GradientWrapper> output(op->output_size());
		for (auto i = 0; i < output.size(); i++) {
			output[i].dense_ = op->output(i) + "_grad";
		}
		GradientOpsMeta meta = GetGradientForOp(*op, output);
		auto grad = trainModel.add_op();
		grad->CopyFrom(meta.ops_[0]);
		grad->set_is_gradient_op(true);
	}

	// >>> train_net.Iter(ITER, ITER)
	{
		auto op = trainModel.add_op();
		op->set_type("Iter");
		op->add_input("ITER");
		op->add_output("ITER");
	}

	// >>> LR = train_net.LearningRate(ITER, "LR", base_lr=-0.1, policy="step",
	// stepsize=20, gamma=0.9)
	{
		auto op = trainModel.add_op();
		op->set_type("LearningRate");
		auto arg1 = op->add_arg();
		arg1->set_name("base_lr");
		arg1->set_f(-0.1);
		auto arg2 = op->add_arg();
		arg2->set_name("policy");
		arg2->set_s("step");
		auto arg3 = op->add_arg();
		arg3->set_name("stepsize");
		arg3->set_i(20);
		auto arg4 = op->add_arg();
		arg4->set_name("gamma");
		arg4->set_f(0.9);
		op->add_input("ITER");
		op->add_output("LR");
	}

	// >>> train_net.WeightedSum([W, ONE, gradient_map[W], LR], W)
	{
		auto op = trainModel.add_op();
		op->set_type("WeightedSum");
		op->add_input("W");
		op->add_input("ONE");
		op->add_input("W_grad");
		op->add_input("LR");
		op->add_output("W");
	}

	// >>> train_net.WeightedSum([B, ONE, gradient_map[B], LR], B)
	{
		auto op = trainModel.add_op();
		op->set_type("WeightedSum");
		op->add_input("B");
		op->add_input("ONE");
		op->add_input("B_grad");
		op->add_input("LR");
		op->add_output("B");
	}

	// print(trainModel);

	// >>> workspace.RunNetOnce(init_net)
	CAFFE_ENFORCE(workspace.RunNetOnce(initModel));

	workspace.CreateBlob("X");
	workspace.CreateBlob("Y_noise");
	//workspace.CreateBlob("W");
	//workspace.CreateBlob("B");
	//generates ground truth X
	// >>> workspace.CreateNet(train_net)
	CAFFE_ENFORCE(workspace.CreateNet(trainModel));

	// >>> print("Before training, W is: {}".format(workspace.FetchBlob("W")))
	print(workspace.GetBlob("W"), "W before");

	// >>> print("Before training, B is: {}".format(workspace.FetchBlob("B")))
	print(workspace.GetBlob("B"), "B before");

	initModel.PrintDebugString();

	trainModel.PrintDebugString();


	// >>> for i in range(100):
	for (auto i = 1; i <= 100; i++) {

		{

			//std::vector<int> dim({nTrainBatches,TrainData.Features[minibatch_index].size()});
			std::vector<int> dim({64,this->features_[i].size()});
			//BlobUtil(*workspace.CreateBlob("input_blob")).Set(dim, TrainData.Features, minibatch_index, false);
			caffe2::BlobUtil(*workspace.CreateBlob("X")).Set(dim, this->features_, i, false);
		}

		{
			std::vector<int> dim({64,1});
			//BlobUtil(*workspace.CreateBlob("target")).Set(dim, TrainData.Labels, minibatch_index, false);
			caffe2::BlobUtil(*workspace.CreateBlob("Y_noise")).Set(dim, this->scalar_couplings_, i, false);
			//std::cout<<"Train Label "<<BlobUtil(*workspace.GetBlob("target")).Get().data<int>()[0]<<std::endl;

			//std::cout<<"Train Label "<<caffe2::BlobUtil(*workspace.GetBlob("target")).Get().DebugString()<<std::endl;

		}


		// >>> workspace.RunNet(train_net.Proto().name)
		CAFFE_ENFORCE(workspace.RunNet(trainModel.name()));

		//cout<<workspace.GetBlob("Y_pred")->Get<caffe2::TensorCPU>().DebugString()<<endl;//data<float>()[0];
		//cout<<workspace.GetBlob("Y_noise")->Get<caffe2::TensorCPU>().DebugString()<<endl;//data<float>()[0];

		if (i % 10 == 0) {
			float w = workspace.GetBlob("W")->Get<TensorCPU>().data<float>()[0];
			float b = workspace.GetBlob("B")->Get<TensorCPU>().data<float>()[0];
			float loss = workspace.GetBlob("loss")->Get<TensorCPU>().data<float>()[0];
			std::cout << "step: " << i << " W: " << w << " B: " << b
					<< " loss: " << loss << std::endl;
		}
	}

	// >>> print("After training, W is: {}".format(workspace.FetchBlob("W")))
	print(workspace.GetBlob("W"), "W after");

	// >>> print("After training, B is: {}".format(workspace.FetchBlob("B")))
	print(workspace.GetBlob("B"), "B after");

	// >>> print("Ground truth W is: {}".format(workspace.FetchBlob("W_gt")))
	// print(workspace.GetBlob("W_gt"), "W ground truth");

	// >>> print("Ground truth B is: {}".format(workspace.FetchBlob("B_gt")))
	//print(workspace.GetBlob("B_gt"), "B ground truth");
}


void Champs::print(const caffe2::Blob *blob, const std::string &name) {
	auto tensor = blob->Get<caffe2::TensorCPU>();
	const auto &data = tensor.data<float>();
	std::cout << name << "(" << tensor.dims()
	            		<< "): " << std::vector<float>(data, data + tensor.size())
						<< std::endl;
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
	int row_count = 0;

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
			row_count++;

			//if(row_count == 300000)
			//	break;

	}

	return fnl_data;

}

/*
 * @param: string filename of the file to open
 * @param: set<string> my_set a set of the types and or atoms
 * @param: int set_column column number of variable you wan to put into the set
 *
 */
vector<vector<string> > Champs::Parse(string filename){

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


template<typename T, typename T1>
void Champs::SaveDataFirstSavedAsInt(T &path, std::vector<std::vector<T1> > &data){
	ofstream myFile;
	string datetime;
	std::stringstream poo;

	myFile.open(path, ios_base::app);

	int count = 0;

	string id = "id";
	string scc = "scalar_coupling_constant";
	myFile<<id<<","<<scc<<endl;
	myFile.flush();
		for(auto contractIter=data.begin(); contractIter!=data.end(); contractIter++){
			for(auto dataIter=contractIter->begin();dataIter!=contractIter->end(); dataIter++){
				if(count == 0)
					myFile<<(int)*dataIter<<",";
				else
					myFile<<*dataIter;
				++count;

			}
			myFile<<endl;
					myFile.flush();
					//cout<<endl;
					count = 0;
		}



	myFile.close();
}

template void Champs::SaveDataFirstSavedAsInt<string, float>(string &, std::vector<std::vector<float> > &);













