#ifndef __ML_NETUTILS__
#define __ML_NETUTILS__

#include <string>
#include <vector>
#include <ranges>
#include <random>
#include <fmt/format.h>

using std::vector;
using std::string;
namespace vws = std::ranges::views;

class NetUtils {
public:
    static string formatDoubles(vector<double> doubles){
        string toReturn{""};
        for( auto d: doubles ){ toReturn += fmt::format("{:^ 8.7f},", d); }
        toReturn = toReturn.substr(1, toReturn.length()-2);
        return toReturn;
    }   

    static vector<double> explodeToDoubles(int target, int bits){
        vector<double> toReturn{};
        string s = fmt::format("{:^0{}b}", target, bits);
        for( auto c: s ) { toReturn.push_back(c=='0'?0.0:1.0); }
        return toReturn;
    }

    static vector<int> trainTestSplit(int dataSize, double trainingPercentage, vector<int>& testSamples){   
        vector<int> items{};
        long train = (long) (dataSize*trainingPercentage), test = (long)(dataSize*(1.0 - trainingPercentage));
        auto all = vws::iota(0)|vws::take(dataSize);

        for( int x : all | vws::take(train)) { items.push_back(x); }
        std::shuffle(items.begin(), items.end(), std::default_random_engine());
        testSamples.clear();
        for(int t: all | vws::drop(train)|vws::take(test+1)) { testSamples.push_back(t); }
        std::shuffle(testSamples.begin(), testSamples.end(), std::default_random_engine());
        return items;
    }

    // squashes each of the values into the range between 0 and 1.
    static vector<double> normalize(vector<double> values){
        vector<double> toReturn;
        double min = *std::min_element(values.begin(), values.end());
        double rng = *std::max_element(values.begin(), values.end()) - min;
        for(double v: values){ toReturn.push_back(v-min/rng); }
        return toReturn;
    }

    static vector<int> duplicate(vector<int> source, int howManyTimes){
        vector<int> toReturn{};
        for (int i = 0; i < howManyTimes; i++){
            std::shuffle(source.begin(), source.end(), std::default_random_engine());
            for( auto v : source ) { toReturn.push_back(v); }
        }
        
        return toReturn;
    }

    static vector<int> createInputRange(int bits){
        vector<int> toReturn{};
        for( auto i: vws::iota(0) | vws::take(pow(2, bits))){ toReturn.push_back(i); }
        return toReturn;
    }

    static double GetRandom() { return std::rand(); }    
};
#endif
