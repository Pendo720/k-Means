#ifndef __ML_CLUSTER__
#define __ML_CLUSTER__
#include <string>
#include <vector>
#include <numeric>
#include <fmt/format.h>
using std::string;
using std::vector;

template<typename Feature>
using Scalar = std::decay<decltype(Feature()[0])>::type;

template<typename Feature>
concept Nearness = std::floating_point<Feature>;

template<typename T>
class Cluster {
public:
    Cluster<T>(string label, T t){ 
        mLabel = label;
        mCentroid = t; 
    }

    T getCentroid() { return mCentroid; }

    void updateCentroid(T centroid){ mCentroid = centroid; }
    
    vector<T>& getElements(){ return mElements; }
    
    void add(T t){ 
        int count = mElements.size();
        mElements.push_back(t); 
        mCentroid = (mCentroid * count + t)/(count + 1);
    }

    T distanceTo(T i){ return std::abs(std::pow((mCentroid - i), 2.0f)); }
    
    T getElementsAverage(){
        return (std::accumulate(mElements.begin(), mElements.end(), 0.0)/mElements.size());
    }
    
    std::string getLabel(){ return mLabel;}
    
    std::string to_string(){  

		std::string toReturn="Empty";

		if(mElements.size() > 0){        
        	T minv = *std::min_element(mElements.begin(), mElements.end());
        	T maxv = *std::max_element(mElements.begin(), mElements.end());
			toReturn = mLabel + fmt::format(": \t({:^4.1f} - {:^4.1f})", minv, maxv);
		}

	    return toReturn;
    }
    
    void clear(){ mElements.clear();}

    const T min() { return *std::min_element(mElements.begin(), mElements.end()); }

    const T max() { return *std::max_element(mElements.begin(), mElements.end()); }

private:
    T mCentroid;
    vector<T> mElements;
    string mLabel;
};
#endif
