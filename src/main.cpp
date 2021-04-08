#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <fmt/format.h>
#include <ranges>
#include <ctime>

#include "../include/KmeansClasifier.h"
#include "../include/NetUtils.h"


using std::cout;
using std::endl;
using std::copy;
using fmt::format;
using std::ostream_iterator;
using namespace mlpack;
using namespace arma;

void cluster() {

	using type = double;
	// generating 1000 randomly distributed doubles 
	vector<type> data(1000);
	vector<type> centroids(4);
	std::generate(data.begin(), data.end(), []() { return rand() % 1001; });

	// generating initial centroid locations
	std::generate(centroids.begin(), centroids.end(), [](){ return rand()%1001; });
	const vector<string>& categories = vector<string>{{"Apples"}, {"Oranges"}, {"Bananas"}, {"Pineapples"}};
    
	KmeansClassifier<type> kms(data, categories, centroids);
	
	cout<<"After convergence:"<<endl<<kms.to_string()<<endl;
	
	std::default_random_engine generator(time(0));
	std::uniform_real_distribution<type> distribution(0.0, 1001.0);
	vector<type> samples(4);

	std::generate(samples.begin(), samples.end(), [&](){ return distribution(generator);});
	cout<<"Verification run (data): [" + NetUtils::formatDoubles(samples) + "]\n";
	for(type i : samples) { 
		Cluster<type> c = kms.classify(i);
		cout<<endl<<c.getLabel(); 
	}
	cout<<endl;
}


int main(int argc, char* args[]) {
	cluster();	
	return 0;
}
