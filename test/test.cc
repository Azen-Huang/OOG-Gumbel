#include <iostream>
#include <algorithm>
#include <ctime>
#include <cmath>
#include <cfloat>
#include <numeric>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <utility>
#include <map>
#include <unordered_map>
#include <assert.h>
#include <iterator>

#include "bitboard.h"
#include "state.h"
#include "Tboard.h"
#include "Cboard.h"

#include <cppflow/cppflow.h>
#include <cppflow/model.h>
#include <cppflow/tensor.h>

#include <omp.h>
#include <thread>
#include <atomic>
#include <shared_mutex>
using namespace std;

int main() {
    #pragma omp for
    for (int i = 0; i < 100; ++i) {
        cout << i << endl;
    }
    return 0;
}