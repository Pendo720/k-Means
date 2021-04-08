#ifndef __ML_KMEANS_CLASSIFIER__
#define __ML_KMEANS_CLASSIFIER__
#include <ranges>
#include <vector>
#include <map>
#include <string>
#include "../include/Cluster.h"
using std::vector;
using std::string;
using std::map;

namespace rns = std::ranges;
namespace vws = std::ranges::views;

enum DistanceMeasure { Eucledean, Manhattan };

template<typename T>
class KmeansClassifier {

public:

    KmeansClassifier<T>(vector<T> data, const vector<string>& categories, vector<T> centroids) {
        mData = data;
        mClusters.clear();
        for (size_t i : vws::iota(0)|vws::take(centroids.size())) {
            mClusters.push_back(Cluster(categories[i], centroids[i]));
        }
        initCentroids();
    } 
    
    void initCentroids(){
        for(auto d: mData){
            vector<T> distances;
            for( auto c: mClusters ) { distances.push_back(c.distanceTo(d)); }
            double smallest = *std::min_element(distances.begin(), distances.end());
            auto index = std::find(distances.begin(), distances.end(), smallest);
            mClusters[std::distance(distances.begin(), index)].getElements().push_back(d);
        }
		std::cout<<"Initial centroids(randomly assigned)"<<std::endl<<to_string()<<std::endl;

        for(Cluster<T> c: mClusters) { mCentroids.push_back(c.getCentroid()); }
        converge();
    }
   
	Cluster<T>& classify(T value){
        auto distances = mClusters | vws::transform([&](auto& c) { return c.distanceTo(value); });
        auto smallest = *std::min_element(distances.begin(), distances.end());
        auto index = std::find(distances.begin(), distances.end(), smallest);
        return mClusters.at(std::distance(distances.begin(), index));
    }
    
    Cluster<T> classifyOnline(T value) {
        Cluster<T>& target = identifyCluster(value);
        target.add(value);
        return target;	
    }

    std::string to_string() {
        string toReturn = {};
        std::for_each(mClusters.begin(), mClusters.end(), [&](auto c){ toReturn += c.to_string() + "\n"; });
        return toReturn;
    }

private:

    vector<T> mData, mCentroids;
    vector<Cluster<T>> mClusters;
	int mIteration = 0;
    void converge() {
	    bool repeat = true;
        auto newCentroids = mClusters | vws::transform([&](auto& c) { return c.getElementsAverage(); });

        int i = 0;
        for(auto c: newCentroids) {
            repeat &= mCentroids.at(i) != c;
            mClusters.at(i).updateCentroid(c);
            i++;
        }

        if(repeat) {
			mIteration++;

            mCentroids.clear();
            for(Cluster<T>& c: mClusters) {
                mCentroids.push_back(c.getCentroid());
                c.clear();
            }

            for(auto d: mData) {
                auto distances = mClusters | vws::transform([&](auto& c) { return c.distanceTo(d); });
                double smallest = *std::min_element(distances.begin(), distances.end());
                auto index = std::find(distances.begin(), distances.end(), smallest);
                mClusters.at(std::distance(distances.begin(), index)).getElements().push_back(d);
            }

			//std::cout<<"Iteration: "<<mIteration<<std::endl<<to_string()<<std::endl;            
            converge();
        }
        
    }
};

#endif
