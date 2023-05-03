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
#define BOARD_SIZE 15
#define RESET "\033[0m"
#define RED  "\033[31m" 
//g++ -march=native KataGo.cc -g -ltensorflow -fopenmp -O3 -o KataGo && "/home/azon/Documents/OOG/OOG-KataGo/"KataGo --self_play | tee output.log
////////////////////////////////
//#define DEBUG
bool bolzman_noise = true, dirichlet_noise = true;
int EVALUATION_COUNT = 1000;
int REPLAY_BUFFER_SIZE = 1536;//2048
int noise_limit_step = 7;
int THREAD_NUM = 10;
int EVALUATION_PLAY_COUNT = 40;
double VIRTUAL_LOSS = 5.0;
float C_PUCT {3.0}; // 4.0
double EPS {1e-8};
float frac = 0.25;
float temp = 1.0;
bool HUMAN_PLAY = false;
int ITER;
////////////////////////////////
vector<int> common_pos_to_bitboard_pos = {
    1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15, 
    17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31, 
    33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47, 
    49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63, 
    65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79, 
    81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95, 
    97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 
    113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 
    129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 
    145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 
    161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 
    177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 
    193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 
    209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 
    225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239
};

vector<int> bitboard_pos_to_common_pos = {
    -1,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14, 
    -1,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29, 
    -1,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44, 
    -1,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59, 
    -1,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74, 
    -1,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89, 
    -1,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 
    -1, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 
    -1, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 
    -1, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 
    -1, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 
    -1, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 
    -1, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 
    -1, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 
    -1, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224,
};

void init(){
    ifstream fin;
    fin.open("data/Tboard.dat", ios::binary | ios::in);
    fin.read((char *)Tboard_h, sizeof(Tboard_h));
    fin.read((char *)Tboard_b, sizeof(Tboard_b));
    fin.read((char *)Tboard_v, sizeof(Tboard_v));
    fin.read((char *)Tboard_s, sizeof(Tboard_s));
    fin.close();
    fin.open("data/Cboard.dat", ios::binary | ios::in);
    fin.read((char *)Cboard_h, sizeof(Cboard_h));
    fin.read((char *)Cboard_b, sizeof(Cboard_b));
    fin.read((char *)Cboard_v, sizeof(Cboard_v));
    fin.read((char *)Cboard_s, sizeof(Cboard_s));
    fin.close();
}

uint64_t Input(char c, int n){
    return (15-n)*16 + c - 'A' + 1;
}

string Output(uint64_t pos){
    int i = pos / 16;
    int j = pos % 16;
    string alphabet(1, char('A'+j-1));
    string number = to_string(15 - i);
    return alphabet + number;
}

template <typename T>
void print_1d_vec(T& vec){
    for(auto& v: vec){
        cout << v << ", ";
    }
    cout << endl;
    return;
}

template <typename T>
void print_board(T& vec, char c){
    cout << "   ";
    for(char i = 'A'; i <= 'O'; ++i){
        cout << i << " ";
    }
    cout << endl;
    cout << "   ";
    for(int i = 'A'; i <= 'O'; ++i){
        cout << "--";
    }
    cout << endl;
    cout << 15 << "|";
    for(int i = 1; i <= 225; ++i){
        if(vec[i - 1] > 0){
            cout << c << " ";
        }
        else{
            cout << "-" << " ";
        }
        if(i % 15 == 0){
            cout << endl;
            if(i != 225)
                cout << setw(2) << (225 - i) / 15 << "|";
        }
    }
    return;
}

template <typename T>
void write_2D_vector_to_file(vector<vector<T>>& myVector, string filename){
    std::ofstream ofs(filename, std::ios::out | std::ofstream::binary);
    std::ostream_iterator<T> osi{ofs,", "};
    for(int i = 0; i < myVector.size(); ++i){
      std::copy(myVector.at(i).begin(), myVector.at(i).end(), osi);
      ofs << '\n';
    }
}

template <typename T>
void write_1D_vector_to_file(const vector<T>& myVector, string filename){
    std::ofstream ofs(filename, std::ios::out | std::ofstream::binary);
    std::ostream_iterator<T> osi{ofs,", "};
    std::copy(myVector.begin(), myVector.end(), osi);
}

struct Network{
    vector<float> policies;
    float value;
};

Network predict(BitBoard my, BitBoard opp, cppflow::model& model){   
    vector<int> board(15 * 15 * 2, 1);
    uint64_t pos;
    while(my){
        pos = my.ls1b();
        board[bitboard_pos_to_common_pos[pos] * 2] = 0;
    }
    while(opp){
        pos = opp.ls1b();
        board[bitboard_pos_to_common_pos[pos] * 2 + 1] = 0;
    }
    auto TF_vec = cppflow::tensor(board, {15,15,2});
    auto TF_input = cppflow::cast(TF_vec, TF_UINT8, TF_FLOAT);
    TF_input = cppflow::expand_dims(TF_input, 0);

    //katago
    auto output = model({{"serving_default_input_1:0", TF_input}},{"StatefulPartitionedCall:0", "StatefulPartitionedCall:1","StatefulPartitionedCall:2"});
    
    return {output[0].get_data<float>(), output[2].get_data<float>()[0]};
}

class Node{
public:
    atomic<float> visit_count;
    atomic<int> to_play;
    atomic<float> prior;
    atomic<float> reward;
    atomic<float> raw_value;
    atomic<int> level;
    BitBoard my, opp, move, myTzone, oppTzone;
    map<int, Node*> children;
    //shared_mutex children_lock;
    Node(float _prior,int level, BitBoard _my, BitBoard _opp, BitBoard _move, BitBoard _myTzone, BitBoard _oppTzeon);
    bool expanded();
    float value();
    void PrintTree(uint64_t pos, string pre_output, bool isLast);
};

Node::Node(float _prior, int _level, BitBoard _my, BitBoard _opp, BitBoard _move, BitBoard _myTzone, BitBoard _oppTzone): 
my(_my), opp(_opp), move(_move), myTzone(_myTzone), oppTzone(_oppTzone),
visit_count(0.0), to_play(-1), prior(_prior), level(_level), reward(0.0), raw_value(0.0){}

bool Node::expanded(){
    return children.size() > 0;
}

float Node::value(){
    if(visit_count == 0.0){
        return 0.0;
    }
    return (-1 * raw_value + reward) / (visit_count + 1);
}

void Node::PrintTree(uint64_t pos, string pre_output, bool isLast){
    auto node = this;
    // Condition when node is None
    if (node == NULL)
        return;
     
    cout << pre_output;
     int count = 2;
    if (isLast) {
        cout << "└── " << "action:" << Output(pos); 
        cout << ", level:" << node->level << ", to_play:" << node->to_play << ", n:" << node->visit_count << ", q_value:" << node->value() << ", policy:" << node->prior <<  ", child size:" << node->children.size();
        /*
        cout << ", child: [";
        for(auto& child: node->children){
            cout << child.first << ", ";
            if(count-- == 0){
                break;
            }
        }
        cout << "]";
        */
        cout << endl;
    }
    else {
        cout << "├── " << "action:" << Output(pos); 
        cout << ", level:" << node->level << ", to_play:" << node->to_play << ", n:" << node->visit_count << ", q_value:" << node->value() << ", policy:" << node->prior <<  ", child size:" << node->children.size();
        /*
        cout << ", child: [";
        for(auto& child: node->children){
            cout << child.first << ", ";
            if(count-- == 0){
                break;
            }
        }
        cout << "]";
        */
        cout << endl;
    }
 
    int it = 0;
    int cnt = node->children.size();
    for(auto child: node->children){
        if(child.second->visit_count >= 20.0 || child.second->level == 1){
            child.second->PrintTree(child.first, pre_output+"   ", it == (cnt - 1));
            it++;
        }
    }
}

class Game{
public:
    int history[225];
    vector<vector<float>> child_visits;
    vector<float> child_values;
    int num_actions;
    int SIZE;

    bool terminal(BitBoard black, BitBoard white, uint64_t pos);
    float terminal_value(int to_play);
    vector<uint64_t> legal_actions(BitBoard my, BitBoard opp, BitBoard myTzone, BitBoard oppTzone, BitBoard move);
    Game* clone();
    void apply(uint64_t pos);
    void store_search_statistics(Node* root);
    vector<BitBoard> make_bitboard(int state_index);
    vector<vector<int>> make_image(int state_index);
    pair<float,vector<float>> make_target(int state_index);
    int to_play();
    int to_size();

    Game(int* _history, int SIZE);
    Game();

    void print();
};

Game::Game(): num_actions(225), SIZE(0){
    for(int i = 0; i < 225; ++i){
        history[i] = -1; 
    }
}

Game::Game(int* _history, int _SIZE): num_actions(225), SIZE(_SIZE){
    for(int i = 0; i < SIZE; ++i){
        history[i] = _history[i]; 
    }
}

bool Game::terminal(BitBoard my, BitBoard opp, uint64_t pos){
    if(SIZE == 225){
        return true;
    }
    
    uint64_t feat = feature(my, opp, pos);
    uint64_t s_h = state[_pext_u64(feat, pext_h)];
    uint64_t s_b = state[_pext_u64(feat, pext_b)];
    uint64_t s_v = state[_pext_u64(feat, pext_v)];
    uint64_t s_s = state[_pext_u64(feat, pext_s)];
    if((s_h | s_b | s_v | s_s) & 0x80000){ //L5 情況 1 - 遊戲結束時
        return true;
    }
    
    return false;
}

float Game::terminal_value(int to_play){
    if(SIZE == 225) return 0;
    int who_win = SIZE % 2;
    if(to_play == who_win){
        return -1.0;
    }
    return 1.0;
}

vector<uint64_t> Game::legal_actions(BitBoard my, BitBoard opp, BitBoard moves, BitBoard myTzone, BitBoard oppTzone){
    vector<uint64_t> actions;
    if(SIZE == 0){
        for(int i = 18; i <= 24; ++i){
        //for(int i = 1; i <= 24; ++i){
        //    if(i > 8 && i < 17) continue;
            actions.push_back(i);
        }
        return actions;
    }

    BitBoard cTzone = myTzone & my & opp;  //迫著
    bool threat_flag = false;
    uint64_t pos, feat, s_h, s_b, s_v, s_s;
    while(cTzone){//我方可連五
        pos = cTzone.ls1b();
        feat = feature(my, opp, pos);
        s_h = state[_pext_u64(feat, pext_h)];
        s_b = state[_pext_u64(feat, pext_b)];
        s_v = state[_pext_u64(feat, pext_v)];
        s_s = state[_pext_u64(feat, pext_s)];
        if((s_h | s_b | s_v | s_s) & 0x80000){ //L5
            actions.push_back(pos);
            threat_flag = true;
            break;
        }
    }

    if(!threat_flag){//對方連5就檔
        cTzone = oppTzone &  my & opp;
        while(cTzone){
            pos = cTzone.ls1b();
            feat = feature(opp, my, pos);
            s_h = state[_pext_u64(feat, pext_h)];
            s_b = state[_pext_u64(feat, pext_b)];
            s_v = state[_pext_u64(feat, pext_v)];
            s_s = state[_pext_u64(feat, pext_s)];
            if((s_h | s_b | s_v | s_s) & 0x80000){ //L5
                actions.push_back(pos);
                threat_flag = true;
                break;
            }
        }
    }

    if(!threat_flag){//我方可連四
        cTzone = myTzone & my & opp;;
        while(cTzone){
            pos = cTzone.ls1b();
            feat = feature(my, opp, pos);
            s_h = state[_pext_u64(feat, pext_h)];
            s_b = state[_pext_u64(feat, pext_b)];
            s_v = state[_pext_u64(feat, pext_v)];
            s_s = state[_pext_u64(feat, pext_s)];
            if((s_h | s_b | s_v | s_s) & 0xC0000){ //L5, L4
                actions.push_back(pos);
                threat_flag = true;
                break;
            }
        }
    }

    if(!threat_flag){
        BitBoard mov = moves & my & opp;
        while(mov){
            pos = mov.ls1b();
            actions.push_back(pos);
        }
    }

    if(actions.size() == 0){
        moves.print();
        cout << endl;
        my.print();
        cout << endl;
        opp.print();
        cout << endl;
        cout << "legal_actions(): Action size is Zero!!!" << endl;
        exit(1);
    }
    return actions;
}

Game* Game::clone(){
    return new Game(history, SIZE);
}

void Game::apply(uint64_t pos){
    history[(SIZE)++] = pos;
    return;
}

void Game::store_search_statistics(Node* root){
    float sum_visits = 0;
    float max_q_value = -100000.0;
    for(auto& child: root->children){
        //katago
        float n_forced = sqrt(2.0 * child.second->prior * (float)root->visit_count);
        if((float)child.second->visit_count < n_forced) continue;
        max_q_value = max(max_q_value, child.second->value());
        sum_visits += child.second->visit_count;
    }
    vector<float> child_visit(225, 0.0);
    //int n = 0;
    for(auto& child: root->children){
        //katago
        float n_forced = sqrt(2.0 * child.second->prior * (float)root->visit_count);
        if((float)child.second->visit_count < n_forced) {
            //n++;
            continue;
        }
        child_visit[bitboard_pos_to_common_pos[child.first]] = child.second->visit_count / sum_visits;
    } 
    //cout << "should be 1. :" << accumulate(child_visit.begin(), child_visit.end(), 0.0) << endl;
    //cout << "prunning: " << n << endl;
    child_values.push_back(-1 * max_q_value);
    child_visits.push_back(child_visit);
    return;
}

vector<BitBoard> Game::make_bitboard(int state_index){
    BitBoard black; black.init();
    BitBoard white; white.init();
    BitBoard moves; moves.initZero();
    BitBoard blackTzone; blackTzone.initZero();
    BitBoard whiteTzone; whiteTzone.initZero();
    state_index = state_index == -1 ? (SIZE - 1) : state_index;
    for(int i = 0; i <= state_index; ++i){
        uint64_t feat;
        uint64_t pos = history[i];
        if(i % 2 == 0)
        {
            feat = feature(black, white, pos);
            black.append(pos);
            uint64_t s_h = state[_pext_u64(feat, pext_h)];
            uint64_t s_b = state[_pext_u64(feat, pext_b)];
            uint64_t s_v = state[_pext_u64(feat, pext_v)];
            uint64_t s_s = state[_pext_u64(feat, pext_s)];
            blackTzone = blackTzone
                        |Tboard_h[(s_h & 0x3F00) | pos]
                        |Tboard_b[(s_b & 0x3F00) | pos]
                        |Tboard_v[(s_v & 0x3F00) | pos]
                        |Tboard_s[(s_s & 0x3F00) | pos];
        }
        else
        {
            feat = feature(white, black, pos);
            white.append(pos);
            uint64_t s_h = state[_pext_u64(feat, pext_h)];
            uint64_t s_b = state[_pext_u64(feat, pext_b)];
            uint64_t s_v = state[_pext_u64(feat, pext_v)];
            uint64_t s_s = state[_pext_u64(feat, pext_s)];
            whiteTzone = whiteTzone
                        |Tboard_h[(s_h & 0x3F00) | pos]
                        |Tboard_b[(s_b & 0x3F00) | pos]
                        |Tboard_v[(s_v & 0x3F00) | pos]
                        |Tboard_s[(s_s & 0x3F00) | pos];
        }
        moves = moves.mind(pos);
    }

    vector<BitBoard> board_info(5);
    if((state_index + 1) % 2){
        board_info[0] = white;
        board_info[1] = black;
        board_info[2] = moves;
        board_info[3] = whiteTzone;
        board_info[4] = blackTzone;
    }
    else{
        board_info[0] = black;
        board_info[1] = white;
        board_info[2] = moves;
        board_info[3] = blackTzone;
        board_info[4] = whiteTzone;
        
    }
    return board_info;
}

vector<vector<int>> Game::make_image(int state_index){
    vector<int> black(225, 0);
    vector<int> white(225, 0);
    for(int i = 0; i <= state_index; ++i){
        int pos = history[i];
        if(i % 2 == 0)
        {
            black[bitboard_pos_to_common_pos[pos]] = 1;
        }
        else
        {
            white[bitboard_pos_to_common_pos[pos]] = 1;
        }
    }
    return {black, white};
}

pair<float,vector<float>> Game::make_target(int state_index){
    return {terminal_value(state_index), child_visits[state_index]};
}

int Game::to_play(){
    return SIZE % 2;
}

int Game::to_size(){
    return SIZE;
}

void Game::print(){
    BitBoard black; black.init();
    BitBoard white; white.init();
    uint64_t last_move = bitboard_pos_to_common_pos[history[SIZE - 1]];
    for(int i = 0; i < SIZE; ++i){
        uint64_t feat, pos = history[i];
        if(i % 2 == 0){
            black.append(pos);
        }
        else{
            white.append(pos);
        }
    }

    vector<int> Black(225,1);
    vector<int> White(225,1);
    while(black){
        uint64_t pos = black.ls1b();
        Black[bitboard_pos_to_common_pos[pos]] = 0;
    }
    while(white){
        uint64_t pos = white.ls1b();
        White[bitboard_pos_to_common_pos[pos]] = 0;
    }

    cout << "   ";
    for(char i = 'A'; i <= 'O'; ++i){
        cout << i << " ";
    }
    cout << endl;
    cout << "   ";
    for(int i = 'A'; i <= 'O'; ++i){
        cout << "--";
    }
    cout << endl;
    cout << 15 << "|";
    for(int i = 1; i <= 225; ++i){
        
        if(HUMAN_PLAY && i - 1 == last_move){
            cout << RED;
        }
        
        if(Black[i - 1] == 1){
            cout << "o" << " ";
        }
        else if(White[i - 1] == 1){
            cout << "x" << " ";
        }
        else{
            cout << "-" << " ";
        }
        
        if(HUMAN_PLAY && i - 1 == last_move){
            cout << RESET;
        }
        
        if(i % 15 == 0){
            cout << endl;
            if(i != 225)
                cout << setw(2) << (225 - i) / 15 << "|";
        }
    }

}
vector<float> softmax(const std::vector<float>& input) {
	auto MAX = *max_element(input.begin(), input.end());
    std::vector<float> output(input.size());
    float sum = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - MAX);
        sum += output[i];
    }
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] /= sum;
    }
    return output;
}

float expand_node(Node* node, Game* game, cppflow::model& model){
    Network network_output = predict(node->my, node->opp, model);
    node->raw_value = network_output.value;
    node->to_play = game->to_play();
    vector<uint64_t> legal_actions = game->legal_actions(node->my, node->opp, node->move, node->myTzone, node->oppTzone);
    float policy_sum = 0.0;
    
    //////////////////soft max///////////////////
    vector<float> soft_max_policy = softmax(network_output.policies);
    //////////////////////////////////////////
    
    vector<float> policies(225, 0.0);
    policy_sum = 0.0;
    for(auto& pos: legal_actions){
        uint64_t legal_move = bitboard_pos_to_common_pos[pos];
        policies[legal_move] = soft_max_policy[legal_move];
        policy_sum += policies[legal_move];
    }
    if(policy_sum > 0){
        for(auto& p:policies){
            p /= policy_sum;
        }
    }
    else{
        cout << "policy have some problem." << endl;
        if(legal_actions.size() > 2)
            cout << "WTF" << endl;
        policies = soft_max_policy;
    }
    BitBoard new_my, new_moves, newTzone;
    uint64_t pos, feat, s_h, s_b, s_v, s_s;
    for(auto& pos: legal_actions){
        new_my = node->my;
        new_moves = node->move;
        feat = feature(new_my, node->opp, pos);
        s_h = state[_pext_u64(feat, pext_h)];
        s_b = state[_pext_u64(feat, pext_b)];
        s_v = state[_pext_u64(feat, pext_v)];
        s_s = state[_pext_u64(feat, pext_s)];
        newTzone = node->myTzone
                            |Tboard_h[(s_h & 0x3F00) | pos]
                            |Tboard_b[(s_b & 0x3F00) | pos]
                            |Tboard_v[(s_v & 0x3F00) | pos]
                            |Tboard_s[(s_s & 0x3F00) | pos];
        new_my.append(pos);
        new_moves = new_moves.mind(pos);

        node->children[pos] = new Node(policies[bitboard_pos_to_common_pos[pos]], node->level + 1, node->opp, new_my, new_moves, node->oppTzone, newTzone);
    }
    return network_output.value;
}

void add_exploration_noise(Node* root){
    vector<uint64_t> actions;
    for(auto& child: root->children){
        actions.push_back(child.first);
    }

    vector<double> dirichlet(actions.size(), 0.0);
    double a = (actions.size() == 1) ? 10.0 : (10.0 / actions.size());//9.0
    std::gamma_distribution<double> gamma(a);
    std::random_device rd;
    double sum = 0;
    for (int i = 0; i < actions.size(); ++i) {
        dirichlet[i] = gamma(rd);
        sum += dirichlet[i];
    }
    for (int i=0; i < actions.size(); ++i) {
        dirichlet[i] = dirichlet[i] / sum;
    }
    for(int i = 0; i < actions.size(); ++i){
        root->children[actions[i]]->prior = root->children[actions[i]]->prior * (1.0 - frac) + dirichlet[i] * frac;
    }

}

float ucb_score(Node* parent, Node* child){
    float pb_c = C_PUCT;
    pb_c *= sqrt(parent->visit_count) / (child->visit_count + 1);
    pb_c += EPS;
    float prior_score = pb_c * child->prior;
    float value_score = child->value();
    
    return prior_score + value_score;
}

Node* select_child(Node* node, uint64_t& action){
    Node* best_node;
    float best_score = -1.1;
    uint64_t best_action;
    for(auto child: node->children){
        //katago
        float n_forced = sqrt(2.0 * child.second->prior * (float)node->visit_count);
        //cout << "n forced value: " << n_forced << endl;
        //if((float)child.second->visit_count < 1 && node->level == 0){
        if(node->level == 0 && (float)child.second->visit_count < n_forced){
            //cout << Output(child.first) << " go!" << " ";
            best_action = child.first;
            best_node = child.second;
            break;
        }
        
        float score = ucb_score(node, child.second);

        if(score > best_score){
            best_action = child.first;
            best_node = child.second;
            best_score = score;
        }
    }
    action = best_action;
    return best_node;
}

void backpropagate(vector<Node*> search_path, float value){
    //reverse(search_path.begin(), search_path.end());
    int to_play = search_path[search_path.size() - 1]->to_play;
    for(auto& node: search_path){
        node->reward = node->reward + (node->to_play == to_play ? (-1.0 * value) : (value));//???? 為什麼是 1 - value 不是 - value
        node->visit_count = node->visit_count + 1;
    }
}

int bolzman(vector<float> child_visit_count){
    random_device rd;
    mt19937 gen(rd());
    for(auto& visit_count: child_visit_count){
        visit_count = pow(visit_count, 1.0 / temp);
    }
    float visit_count_sum = accumulate(child_visit_count.begin(), child_visit_count.end(), 0.0);
    for(auto& visit_count: child_visit_count){
        visit_count /= visit_count_sum;
    }
    discrete_distribution<> dist(child_visit_count.begin(), child_visit_count.end());
    int index = dist(gen);
    return index;
}

uint64_t select_action(Game* game, Node* root){
    vector<float> child_visit_count;
    vector<uint64_t> actions;
    float visit_count = 0.0;
    for(auto& child: root->children){
        child_visit_count.push_back(child.second->visit_count);
        actions.push_back(child.first);
    }
    size_t idx;
    if(bolzman_noise && game->SIZE < noise_limit_step){
        //cout << "add noise" << endl;
        idx = bolzman(child_visit_count);
    }
    else{
        auto it = std::max_element(child_visit_count.begin(), child_visit_count.end());
        idx = std::distance(child_visit_count.begin(), it);
    }
    return actions[idx];
}

Node* run_mcts(Game* game, cppflow::model& model, uint64_t& mcts_action, vector<BitBoard> board_info){
    Node* root = new Node(0, 0, board_info[0], board_info[1], board_info[2], board_info[3], board_info[4]);
    expand_node(root, game, model);
    if(dirichlet_noise){
        add_exploration_noise(root);
    }
    uint64_t action;
    for(int i = 0; i < EVALUATION_COUNT; ++i){
        Node* node = root;
        Game* scratch_game = game->clone();
        vector<Node*> search_path;
        search_path.push_back(node);

        while(node->expanded()){
            node = select_child(node, action);
            scratch_game->apply(action);
            search_path.push_back(node);
        } 
        float value;
        if(scratch_game->terminal(node->opp, node->my, action)){
            value = -1;
            node->to_play = scratch_game->to_play();
        }
        else{
            value = expand_node(node, scratch_game, model);
        }

        backpropagate(search_path, value);
    }
    //root->PrintTree(0, "", true);
    mcts_action = select_action(game, root);
    //cout << "mcts_action: " << mcts_action << endl;
    return root;
}

Game* play_game(cppflow::model& model){
    Game* game = new Game();
    vector<BitBoard> board_info;
    uint64_t action = 1000;
    board_info = game->make_bitboard(-1);
    while(!game->terminal(board_info[1], board_info[0], action)){
        Node* root = run_mcts(game, model, action, board_info);
        game->apply(action);
        game->store_search_statistics(root);
        #ifdef DEBUG
            game->print();//self play
            cout << "action: " << Output(action) << endl;
            cout << "predict win rate: " << (1 + (-1 * game->child_values.back())) / 2 * 100 << "%" << endl << endl;
        #endif
        board_info = game->make_bitboard(-1);
        //delete root;
    }
    return game;
}

void self_play(cppflow::model& model){
    cout << "Start Self Play" << endl;
    vector<vector<int>> board_history;
    vector<vector<float>> policies_history;
    vector<vector<float>> auxiliary_policies_history;
    vector<float> value_history;
    int round = 1;
    double start = omp_get_wtime();
    while(board_history.size() < REPLAY_BUFFER_SIZE){
        Game* game = play_game(model);
        int auxiliary_policies_history_start_idx = policies_history.size();
        double end = omp_get_wtime();
        int winner = (game->SIZE - 1) % 2 == 0 ? 0 : 1;
        cout << round << ": Board Size: " << setw(5) << game->SIZE << ": total cost time: " << setw(6) << setprecision(3) << (end-start) / 60.0 << " min, avg step time: " << setw(3) << setprecision(3) << (end - start) / double(board_history.size() + game->SIZE) << " s, " << " Winner is " << winner << "." << endl;
        for(int i = 0; i < game->SIZE; ++i){//save training data
            if(board_history.size() == REPLAY_BUFFER_SIZE) break;
            vector<int> board_info(2*15*15);
            vector<vector<int>> board = game->make_image(i);
            if(i % 2 == 1){
                std::copy(board[0].begin(), board[0].end(), board_info.begin());
                std::copy(board[1].begin(), board[1].end(), board_info.begin() + board[0].size());
            }
            else{
                std::copy(board[1].begin(), board[1].end(), board_info.begin());
                std::copy(board[0].begin(), board[0].end(), board_info.begin() + board[1].size());
            }
            board_history.push_back(board_info);        

            policies_history.push_back(game->child_visits[i]);
            
            float value = i % 2 == winner ? -1.0 : 1.0;//！!！!
            
            if (ITER <= 50) {
                value_history.push_back((value + game->child_values[i]) / 2.0);
            }
            else {
                value_history.push_back(game->child_values[i]);
            }

            //katago
            if(i + 1 == game->SIZE){
                vector<float> last_auxiliary_policy(225, 0.0);
                auxiliary_policies_history.push_back(last_auxiliary_policy);
            }
            else{
                auxiliary_policies_history.push_back(game->child_visits[i + 1]);
            }    
        }
        round++;
    }
    /*
    for(int i = 0; i < board_history.size(); ++i){
        print_1d_vec(policies_history[i]);
        print_1d_vec(auxiliary_policies_history[i]);
        cout << endl;
    }
    */
    cout << board_history.size() << " " << policies_history.size() << " " << auxiliary_policies_history.size() << " " << value_history.size() << endl;
    time_t now = time(0);
    tm *ltm = localtime(&now);
    string ltm_mon = (1 + ltm->tm_mon) < 10 ? "0" + to_string(1 + ltm->tm_mon) : to_string(1 + ltm->tm_mon);
    string ltm_mday = ltm->tm_mday < 10 ? "0" + to_string(ltm->tm_mday) : to_string(ltm->tm_mday);
    string ltm_hour = ltm->tm_hour < 10 ? "0" + to_string(ltm->tm_hour) : to_string(ltm->tm_hour);
    string ltm_min = ltm->tm_min < 10 ? "0" + to_string(ltm->tm_min) : to_string(ltm->tm_min);
    string ltm_sec = ltm->tm_sec < 10 ? "0" + to_string(ltm->tm_sec) : to_string(ltm->tm_sec);
    string now_time = to_string(1900 + ltm->tm_year) +ltm_mon + ltm_mday + ltm_hour + ltm_min + ltm_sec;
    write_2D_vector_to_file(policies_history, "./train_data/policies/" + now_time + ".history");
    write_2D_vector_to_file(policies_history, "./train_data/auxiliary_policies/" + now_time + ".history");
    write_2D_vector_to_file(board_history, "./train_data/board/" + now_time + ".history");
    write_1D_vector_to_file(value_history, "./train_data/value/" + now_time + ".history");
}

void human_play(cppflow::model& model){
    cout << "Start Human Play" << endl;
    HUMAN_PLAY = true;
    Game* game = new Game();
    vector<BitBoard> board_info;
    uint64_t action = 1000;
    double game_count = 0.0;
    board_info = game->make_bitboard(-1);
    double start = omp_get_wtime();
    while(!game->terminal(board_info[1], board_info[0], action)){
        char c = getchar();
        int n;
        if(c == '\n'){
            Node* root = run_mcts(game, model, action, board_info);
            game->store_search_statistics(root);
            //delete root;
            game->apply(action);
            game->print();//human play
            cout << "action: " << Output(action) << endl;
            cout << "predict win rate: " << (1 + (-1 * game->child_values.back())) / 2 * 100 << "%" << endl << endl;
        }
        else{
            cin >> n;
            c = toupper(c);
            if(c >= 'A' && c <= 'O' && n >= 1 && n <= 15){
                action = Input(c, n);
                cout << "Play action: " << action << endl; 
                game->apply(action);
                game->print();//human play
                cout << "action: " << Output(action) << endl;
            }
            else{
                Node* root = run_mcts(game, model, action, board_info);
                game->store_search_statistics(root);
                //delete root;
                game->apply(action);
                game->print();//human play
                cout << "action: " << Output(action) << endl;
                cout << "predict win rate: " << (1 + (-1 * game->child_values.back())) / 2 * 100 << "%" << endl << endl;
            }
        }
        board_info = game->make_bitboard(-1);
        game_count++;
        
    }
    double end = omp_get_wtime();
    double elapsed = end - start;
    cout << "cost time: " << elapsed / game_count << " s." << endl;
    //delete game;
}

void evaluation(cppflow::model& curr_model, cppflow::model& pre_model){
    cout << "Start Evaluation" << endl;
    int curr_player;
    int cur_win = 0;

    for(int i = 1; i <= EVALUATION_PLAY_COUNT; ++i){
        cout << "Round " << i << ": " << endl; 
        curr_player = i <= (EVALUATION_PLAY_COUNT / 2) ? 0 : 1;
        Game* game = new Game();
        vector<BitBoard> board_info;
        uint64_t action = 1000;
        board_info = game->make_bitboard(-1);
        while(!game->terminal(board_info[1], board_info[0], action)){
            Node* root;
            if(game->SIZE % 2 == curr_player){
                root = run_mcts(game, curr_model, action, board_info);
            }
            else{
                root = run_mcts(game, pre_model, action, board_info);
            }
            game->apply(action);
            game->store_search_statistics(root);
            
            //cout << "action: " << Output(action) << endl;
            //cout << "predict win rate: " << (1 + (-1 * game->child_values.back())) / 2 * 100 << "%" << endl << endl;
            board_info = game->make_bitboard(-1);
            //delete root;
        }
        game->print();//evaluation
        if((game->SIZE - 1) % 2 == curr_player){
            cout << "New model win." << endl;
            cur_win++;
        }
        else{
            cout << "New model lose" << endl;
        }

        //delete game;
    }
    cout << "New model win rate: " << (double(cur_win) / EVALUATION_PLAY_COUNT) * 100.0 << "%" << endl;
}
int main(int argc, char* argv[]){
    //ios_base::sync_with_stdio(0);
    srand(time(NULL));
    init();
    //vector<uint8_t> config{0x32,0xb,0x9,0x34,0x33,0x33,0x33,0x33,0x33,0xd3,0x3f,0x20,0x1};//30%
    //vector<uint8_t> config{0x32,0xb,0x9,0x9a,0x99,0x99,0x99,0x99,0x99,0xd9,0x3f,0x20,0x1};//40%
    vector<uint8_t> config{0x32,0xb,0x9,0x0,0x0,0x0,0x0,0x0,0x0,0xe0,0x3f,0x20,0x1};//50%
    // Create new options with your configuration
    TFE_ContextOptions* options = TFE_NewContextOptions();
    TFE_ContextOptionsSetConfig(options, config.data(), config.size(), cppflow::context::get_status());
    // Replace the global context with your options
    cppflow::get_global_context() = cppflow::context(options);
    
    cppflow::model model("./c_model");
    cout << "load model accept" << endl;

    ITER = stoi(string(argv[2]));
    cout << ITER << " iter start." << endl;

    if(string(argv[1]) == "--self_play"){
        self_play(model);
    }
    else if(string(argv[1]) == "--evaluation"){
        temp = 0.4;
        EVALUATION_COUNT = 500;
        cppflow::model p_model("./p_model");
        evaluation(model, p_model);

        // cppflow::model p_model("./save_model/model25");
        // cppflow::model c_model("./save_model/model50");
        // evaluation(c_model, p_model);
        
    }
    else if(string(argv[1]) == "--human_play"){
        //dirichlet_noise = false;
        EVALUATION_COUNT = 2500;
        //bolzman_noise = false;
        temp = 0.1;
        //cppflow::model p_model("./save_model/model200");
        cppflow::model p_model("./c_model");
        human_play(p_model);
    }
    
    return 0;
}