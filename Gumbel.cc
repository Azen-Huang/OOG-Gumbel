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
//g++ -march=native Gumbel.cc -g -ltensorflow -fopenmp -O2 -o Gumbel && "./"Gumbel --human_play
////////////////////////////////
// #define DEBUG
// #define CHECK
///////////GUMBEL///////////////////////
bool use_mixed_value = true;
bool use_table = true;
float GUMBEL_PENALTY = 1.0;
int MAX_CONSIDERED_NUM = 100; //Top-k-Gumbel -> 16 32 80 128
int GUMBEL_SIMPLE = 1; //24:50 95:5
/////////////////////////////////////////
int EVALUATION_COUNT = 600; // expand node count
int REPLAY_BUFFER_SIZE = 4096;//4096 2048
int EVALUATION_PLAY_COUNT = 40;
bool HUMAN_PLAY = false;
int ITER;
/////////////////////////////
int ProcessCount = 8;
int ProcessID = 0;
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
        if(vec[i - 1] != 0){
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
unordered_map<string, Network> table;
Network predict(BitBoard my, BitBoard opp, cppflow::model& model){   
    vector<int> board(15 * 15 * 2, 1);
    string board_str(15 * 15 * 2, '1');
    uint64_t pos;
    while(my){
        pos = my.ls1b();
        board[bitboard_pos_to_common_pos[pos] * 2] = 0;
        board_str[bitboard_pos_to_common_pos[pos] * 2] = '0';
    }
    while(opp){
        pos = opp.ls1b();
        board[bitboard_pos_to_common_pos[pos] * 2 + 1] = 0;
        board_str[bitboard_pos_to_common_pos[pos] * 2 + 1] = '0';
    }
    if (use_table && table.count(board_str)) return table[board_str];
    auto TF_vec = cppflow::tensor(board, {15,15,2});
    auto TF_input = cppflow::cast(TF_vec, TF_UINT8, TF_FLOAT);    
    TF_input = cppflow::expand_dims(TF_input, 0);

    //katago
    auto output = model({{"serving_default_input_1:0", TF_input}},{"StatefulPartitionedCall:0", "StatefulPartitionedCall:1","StatefulPartitionedCall:2"});
    if (use_table) {
        table[board_str] = {output[0].get_data<float>(), output[2].get_data<float>()[0]};
        return table[board_str];
    }
    return {output[0].get_data<float>(), output[2].get_data<float>()[0]};
}

vector<float> softmax(const std::vector<float>& input) {
	float MAX = *max_element(input.begin(), input.end());
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

class Node{
public:
    float visit_count;
    int to_play;
    float prior;
    float reward;
    float raw_value;
    float complete_qvalue;
    int level;
    BitBoard my, opp, move, myTzone, oppTzone;
    vector<Node*> children;
    int action;
    float g;
    //shared_mutex children_lock;
    Node(int _action, float _prior,int level, BitBoard _my, BitBoard _opp, BitBoard _move, BitBoard _myTzone, BitBoard _oppTzeon);
    bool expanded();
    float qvalue();
    void PrintTree(uint64_t pos, string pre_output, bool isLast, float limit);
};

Node::Node(int _action, float _prior, int _level, BitBoard _my, BitBoard _opp, BitBoard _move, BitBoard _myTzone, BitBoard _oppTzone): 
action(_action), 
g(0), complete_qvalue(0.0), reward(0.0), raw_value(0.0), visit_count(0.0), to_play(-1),
my(_my), opp(_opp), move(_move), myTzone(_myTzone), oppTzone(_oppTzone),
prior(_prior), level(_level){}

bool Node::expanded(){
    return children.size() > 0;
}

float Node::qvalue(){
    // if(visit_count == 0.0){
    //     return 0.0;
    // }
    return reward == 0 ? (raw_value) : reward / visit_count;// / visit_count;
    //return (-1 * raw_value + reward);
}

void Node::PrintTree(uint64_t pos = 0, string pre_output = "", bool isLast = true, float limit = 10){
    auto node = this;
    // Condition when node is None
    if (node == NULL)
        return;
     
    cout << pre_output;
     int count = 2;
    if (isLast) {
        cout << "└── " << "action:" << Output(pos); 
        cout << ", level:" << node->level << ", to_play:" << node->to_play << ", n:" << node->visit_count << ", qvalue:" << node->qvalue() << ", policy:" << node->prior <<  ", g:" << node->g << ", raw value:" << node->raw_value << ", completed qvalue:" << node->complete_qvalue;
        /*
        cout << ", child: [";
        for(auto& child: node->children){
            cout << child->action << ", ";
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
        cout << ", level:" << node->level << ", to_play:" << node->to_play << ", n:" << node->visit_count << ", qvalue:" << node->qvalue() << ", policy:" << node->prior <<  ", g:" << node->g << ", raw value:" << node->raw_value << ", completed qvalue:" << node->complete_qvalue;
        /*
        cout << ", child: [";
        for(auto& child: node->children){
            cout << child->action << ", ";
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
        //if(child->visit_count >= limit){
        //if(child->level == 1){
        if(child->level == 1 || child->level == 2 || (child->level <= 3 && child->visit_count > 1)){
        //if(child->visit_count >= limit || child->level == 1){
            child->PrintTree(child->action, pre_output+"   ", it == (cnt - 1), limit);
            it++;
        }
    }
}

void deleteTree(Node* root) {
    if (root == nullptr) {
        return;
    }
    
    // 釋放所有子節點的記憶體
    for (auto child : root->children) {
        deleteTree(child);
    }
    
    // 釋放根節點的記憶體
    delete root;
}

class Game{
public:
    int history[225];
    vector<vector<float>> child_polices;
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
    void print_history();
    void Undo();
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
        actions = {19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                34, 50, 66, 82, 98, 114, 130, 146, 162, 178, 194,
                46, 62, 78, 94, 110, 126, 142, 158, 174, 190, 206,
                211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221};
        // for(int i = 18; i <= 24; ++i){
        //     actions.emplace_back(i);
        // }
        return actions;
    }

    BitBoard cTzone = myTzone & my & opp;  //迫著
    bool threat_flag = false;
    uint64_t pos, feat, s_h, s_b, s_v, s_s;
    // while(cTzone){//我方可連五
    //     pos = cTzone.ls1b();
    //     feat = feature(my, opp, pos);
    //     s_h = state[_pext_u64(feat, pext_h)];
    //     s_b = state[_pext_u64(feat, pext_b)];
    //     s_v = state[_pext_u64(feat, pext_v)];
    //     s_s = state[_pext_u64(feat, pext_s)];
    //     if((s_h | s_b | s_v | s_s) & 0x80000){ //L5
    //         actions.emplace_back(pos);
    //         threat_flag = true;
    //         break;
    //     }
    // }

    // if(!threat_flag){//對方連5就檔
    //     cTzone = oppTzone &  my & opp;
    //     while(cTzone){
    //         pos = cTzone.ls1b();
    //         feat = feature(opp, my, pos);
    //         s_h = state[_pext_u64(feat, pext_h)];
    //         s_b = state[_pext_u64(feat, pext_b)];
    //         s_v = state[_pext_u64(feat, pext_v)];
    //         s_s = state[_pext_u64(feat, pext_s)];
    //         if((s_h | s_b | s_v | s_s) & 0x80000){ //L5
    //             actions.emplace_back(pos);
    //             threat_flag = true;
    //             break;
    //         }
    //     }
    // }

    // if(!threat_flag){//我方可連四
    //     cTzone = myTzone & my & opp;;
    //     while(cTzone){
    //         pos = cTzone.ls1b();
    //         feat = feature(my, opp, pos);
    //         s_h = state[_pext_u64(feat, pext_h)];
    //         s_b = state[_pext_u64(feat, pext_b)];
    //         s_v = state[_pext_u64(feat, pext_v)];
    //         s_s = state[_pext_u64(feat, pext_s)];
    //         if((s_h | s_b | s_v | s_s) & 0xC0000){ //L5, L4
    //             actions.emplace_back(pos);
    //             threat_flag = true;
    //             break;
    //         }
    //     }
    // }

    if(!threat_flag){
        BitBoard mov = my & opp;
        //mov.print();
        //BitBoard mov = moves & my & opp;
        while(mov){
            pos = mov.ls1b();
            actions.emplace_back(pos);
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
        //exit(1);
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
    float max_q_value = -100000.0;
    float min_logit = -std::numeric_limits<float>::max();
    vector<float> child_logits_add_completed_q_value(225, min_logit);
    max_q_value = root->children[0]->qvalue(); // -1 * root->qvalue();
    for(auto& child: root->children){
        child_logits_add_completed_q_value[bitboard_pos_to_common_pos[child->action]] = child->prior + child->complete_qvalue;
    } 
    child_values.emplace_back(max_q_value);
    //cout << max_q_value << endl;
    child_polices.emplace_back(child_logits_add_completed_q_value);
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
    return {terminal_value(state_index), child_polices[state_index]};
}

int Game::to_play(){
    return SIZE % 2;
}

int Game::to_size(){
    return SIZE;
}
void Game::Undo() {
    if (SIZE <= 0) return;
    SIZE -= 1;
    history[SIZE] = -1;
}
void Game::print_history() {
    cout << "History : [";
    for(int i = 0; i < SIZE; ++i) {
        cout << Output(history[i]) << ", ";
    }
    cout << "]" << endl;
    return;
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

float expand_node(Node* node, Game* game, cppflow::model& model){ // ok
    Network network_output = predict(node->my, node->opp, model);
    //print_1d_vec(network_output.policies);
    node->to_play = game->to_play();
    vector<uint64_t> legal_actions = game->legal_actions(node->my, node->opp, node->move, node->myTzone, node->oppTzone);
    #ifdef CHECK
    if (network_output.value <= -1 || network_output.value >= 1) {
        game->print();
        cout << "Who" << node->to_play << endl;
        cout << "action is " << Output(node->action) << endl;
        cout << "value network predict value is " << network_output.value << endl;
        assert(network_output.value > -1 && network_output.value < 1);
    }   
    #endif
    
    #ifdef CHECK
    if (network_output.value == 0) {
        game->print();
        network_output.value = 0.0001;        
        assert(network_output.value != 0);
    }
    #endif

    float max_logit = *max_element(network_output.policies.begin(), network_output.policies.end());
    BitBoard new_my, new_moves, newTzone;
    uint64_t pos, feat, s_h, s_b, s_v, s_s;
    int idx = 0;
    node->children.resize(legal_actions.size());
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
        uint64_t legal_move = bitboard_pos_to_common_pos[pos];
        node->children[idx++] = new Node(pos, network_output.policies[legal_move] - max_logit, node->level + 1, node->opp, new_my, new_moves, node->oppTzone, newTzone);
    }
    return network_output.value;
}

vector<float>  complete_qvalues(const vector<float>& visited_counts, const vector<float>& q_values,const float& value) {
    int n = visited_counts.size();
    vector<float> completed_qvalues(n);
    for (int i = 0; i < n; ++i) {
        completed_qvalues[i] = (visited_counts[i] == 0) ? value : q_values[i];
        #ifdef CHECK
        if (completed_qvalues[i] == 0.0) {
            cout << visited_counts[i] << endl;
            cout << value << endl;
            cout << q_values[i] << endl;
            assert(completed_qvalues[i] != 0.0);
        }
        #endif
    }
    return completed_qvalues;
}

vector<float> rescale_qvalues(const vector<float>& qvalues,const float epsilon,const float visit_scale,const float value_scale) {
    std::vector<float> rescaled_qvalues(qvalues.size());
    auto min_value = *min_element(qvalues.begin(), qvalues.end());
    auto max_value = *max_element(qvalues.begin(), qvalues.end());
    for (size_t i = 0; i < qvalues.size(); ++i) {
        // 將q_value的值取在[0, 1]這個區間
        rescaled_qvalues[i] = (qvalues[i] - min_value) / max(max_value - min_value, epsilon);
        // 論文 σ(qˆ(a)) = (maxvisit_init + max(visit_counts)) * value_scale * qvalues
        rescaled_qvalues[i] = rescaled_qvalues[i] * visit_scale * value_scale;
    }

    return rescaled_qvalues;
}

float compute_mixed_value(const float& raw_value,const vector<float>& qvalues,const vector<float>& visit_counts,const vector<float>& prior_probs) {
    //sum_visit_counts = jnp.sum(visit_counts, axis=-1)
    assert(visit_counts.size() == qvalues.size() && qvalues.size() == prior_probs.size());
    float sum_visit_counts = reduce(visit_counts.begin(), visit_counts.end(), 0.0);

    //prior_probs = jnp.maximum(jnp.finfo(prior_probs.dtype).tiny, prior_probs)
    vector<float> none_zero_prior_probs(prior_probs.size());
    float tiny = -std::numeric_limits<float>::max();
    float sum_probs = 0.0;
    for (int i = 0; i < prior_probs.size(); ++i) {
        none_zero_prior_probs[i] = max(tiny, prior_probs[i]);

        // Summing the probabilities of the visited actions.
        // sum_probs = jnp.sum(jnp.where(visit_counts > 0, prior_probs, 0.0), axis=-1)
        sum_probs += visit_counts[i] > 0 ? none_zero_prior_probs[i] : 0.0;
    }
    
    // weighted_q = jnp.sum(jnp.where(
    //     visit_counts > 0,
    //     prior_probs * qvalues / jnp.where(visit_counts > 0, sum_probs, 1.0),
    //     0.0), axis=-1)
    float weighted_q = 0.0;
    for (int i = 0; i  < prior_probs.size(); ++i) {
        float sum_prob = visit_counts[i] > 0.0 ? sum_probs : 1.0;
        weighted_q += visit_counts[i] > 0.0 ? none_zero_prior_probs[i] * qvalues[i] / sum_prob : 0.0;
    }
    
    return (raw_value + sum_visit_counts * weighted_q) / (sum_visit_counts + 1.0);
}

vector<float>  qtransform(Node* node){
    int n = node->children.size();
    float value_scale = 0.1;
    float maxvisit_init = 50.0;
    float epsilon = 1e-8;

    float raw_value = node->raw_value;
    vector<float> qvalues(n);
    vector<float> visited_counts(n); // children visited count
    vector<float> prior_probs(n); // children prior

    float maxvisit = 0.0;
    for (int i = 0; i < n; ++i) {
        qvalues[i] = node->children[i]->qvalue();
        visited_counts[i] = node->children[i]->visit_count;
        #ifdef CHECK
        if (visited_counts[i] != 0 && qvalues[i] == 0) {
            node->PrintTree();
            cout << node->children[i]->reward << endl;
            cout << node->children[i]->raw_value << endl;
            cout << node->children[i]->children.size() << endl;
            assert(false);
        }
        #endif
        maxvisit = max(maxvisit, visited_counts[i]);
        prior_probs[i] = node->children[i]->prior;
    }
    prior_probs = softmax(prior_probs);
    //auto prior_probs_tensor = cppflow::tensor(prior_probs);
    //prior_probs = cppflow::softmax(prior_probs_tensor).get_data<float>();
    
    float value;
    if (use_mixed_value) {
        //todo
        value = compute_mixed_value(raw_value, qvalues, visited_counts, prior_probs);
        //cout << value << " " << raw_value << endl;
    }
    else {
        value = raw_value;
    }
    vector<float> completed_qvalues = complete_qvalues(visited_counts, qvalues, value);

    float visit_scale = maxvisit_init + maxvisit;

    completed_qvalues = rescale_qvalues(completed_qvalues, epsilon, visit_scale, value_scale);
    
    return completed_qvalues;
}

Node* select_child(Node* node, uint64_t& action){
    int n = node->children.size();
    vector<float> visited_counts(n);
    vector<float> completed_qvalues = qtransform(node);
    #ifdef CHECK
    assert(n == completed_qvalues.size());
    #endif
    vector<float> probs(n);
    for (int i = 0; i < n; ++i) {
        node->children[i]->complete_qvalue = completed_qvalues[i];
        probs[i] = completed_qvalues[i] + node->children[i]->prior;
        visited_counts[i] = node->children[i]->visit_count;
    }
    probs = softmax(probs);
    #ifdef CHECK
    //cout << "expand select_child child visited count sum = " << (visited_counts.begin(), visited_counts.end(), 0) << ", parent visited count" << node->visit_count << endl;
    //assert(node->visit_count == reduce(visited_counts.begin(), visited_counts.end(), 0));
    #endif

    int best_action_idx = 0;
    float max_prob = 0.0;
    for (int i = 0; i < n; ++i) {
        probs[i] = probs[i] - visited_counts[i] / (1 + node->visit_count);
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            best_action_idx = i;
        }
    }

    action = node->children[best_action_idx]->action;
    return node->children[best_action_idx];
}

void backpropagate(vector<Node*> search_path, float value){ // ok
    //reverse(search_path.begin(), search_path.end());
    int to_play = search_path[search_path.size() - 1]->to_play;
    for(auto& node: search_path){
        node->reward = node->reward + (node->to_play == to_play ? (value) : (-1.0 * value));//???? 為什麼是 1 - value 不是 - value
        node->visit_count = node->visit_count + 1.0;
    }
}

vector<float> GenerateGumbelNoise(int shape) { //ok
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> distribution(0, 1);

    vector<float> gumbel_noise(shape);
    for (int i = 0; i < shape; ++i) {
        float u = distribution(gen);
        gumbel_noise[i] = -log(-log(u + 1e-20) + 1e-20);
    }

    return gumbel_noise;
}

void add_Gumbel_noise(vector<Node*>& children, float count) { //ok
    int shape = children.size();
    for (int j = 0; j < count; ++j) {
        vector<float> gumbel_noise = GenerateGumbelNoise(shape);
        for (int i = 0; i < shape; ++i) {
            children[i]->g += gumbel_noise[i];
        }
    }
    
    for (int i = 0; i < shape; ++i) {
        children[i]->g = children[i]->g / count * GUMBEL_PENALTY;
    }
    return;
}

Node* seq_halving(Game* game, cppflow::model& model, uint64_t& selected_action,const vector<BitBoard>& board_info){
    int max_num_considered_actions = MAX_CONSIDERED_NUM; //16
    int num_simulations = EVALUATION_COUNT; //200

    Node* root = new Node(-1, 0.0, 0, board_info[0], board_info[1], board_info[2], board_info[3], board_info[4]);

	root->raw_value = expand_node(root, game, model);
    
	add_Gumbel_noise(root->children, GUMBEL_SIMPLE); //添加Gumbel noise
    
	int num_valid_actions = root->children.size();
    num_valid_actions = min(num_valid_actions, num_simulations); //ok
	max_num_considered_actions = min(num_valid_actions, max_num_considered_actions);
	int log2max = int(ceil(log2(max_num_considered_actions)));
    // cout << log2max << endl << endl; // check simulation count
    max_num_considered_actions = num_valid_actions == 1 ? 2 : max_num_considered_actions;
    int num_extra_visits = 0;
    int sum = 0;
	for (int num_considered = max_num_considered_actions; num_considered > 1; num_considered = num_considered == 3 ? num_considered - 1 : num_considered / 2) {
        num_considered = max(2, num_considered);
        // cout << "expand " << num_considered << " actions" << endl; // check simulation count
		vector<float> completed_qvalues = qtransform(root);
        for (int i = 0; i < root->children.size(); ++i) {
            root->children[i]->complete_qvalue = completed_qvalues[i];
        }
        //g先不要
        //sort(root->children.begin(), root->children.end(), [](const Node* a, const Node* b){return (a->visit_count == b->visit_count) ? ((a->prior + a->complete_qvalue) > (b->prior + b->complete_qvalue)) : (a->visit_count > b->visit_count);}); // 根據q value進行排序
        sort(root->children.begin(), root->children.end(), [](const Node* a, const Node* b){return (a->visit_count == b->visit_count) ? ((a->g + a->prior + a->complete_qvalue) > (b->g + b->prior + b->complete_qvalue)) : (a->visit_count > b->visit_count);}); // 根據q value進行排序  
        //sort(root->children.begin(), root->children.end(), [](const Node* a, const Node* b){return (a->g + a->prior + a->complete_qvalue) > (b->g + b->prior + b->complete_qvalue)}); // 根據q value進行排序      
        
        int num_extra_visits = num_considered == 2 ? max(1, (num_simulations - sum) / 2) : max(1, int(num_simulations / (log2max * num_considered)));
        // cout << "each expand node need simulate " << num_extra_visits << " times" << endl; // check simulation count
        
        if (num_valid_actions == 1) {
            num_extra_visits = num_simulations - 1;
            num_considered = 1;
            if(game->terminal(root->my, root->opp, root->children[0]->action)){
                // cout << "Just have 1 valied action" << endl;
                Node* child = root->children[0];
                Game* child_game = game->clone();
                child_game->apply(child->action);

                child->raw_value = 1;
                child->to_play = child_game->to_play();
                vector<Node*> root_child_path;
                root_child_path.emplace_back(root);
                root_child_path.emplace_back(child);
                for (int simulation_count = 0; simulation_count < num_extra_visits; ++simulation_count) {
                    backpropagate(root_child_path, child->raw_value);
                } 
                break;
            }
        }
        int count = 0;
        for (int i = 0; i < num_considered; ++i) { //展開root->children[0 : num_considered]
			Node* child = root->children[i]; //目前要模擬的雞欸但
            Game* child_game = game->clone();
            child_game->apply(child->action);
            int simulation_count = 0;
            if (!child->expanded()) { //如果child沒被展開過，就展開它
                simulation_count++;
                child->raw_value = expand_node(child, child_game, model);
                vector<Node*> root_child_path;
                root_child_path.emplace_back(root);
                root_child_path.emplace_back(child);
                backpropagate(root_child_path, child->raw_value);
            }
            uint64_t action;
			while ((simulation_count++) < num_extra_visits) {
                //cout << "child " << i << " expand count: " << j << endl;
				//展開num_extra_visits次root的child 算他們的q_value
				//***************************************************
				Node* node = child;
		        Game* scratch_game = child_game->clone();
		        vector<Node*> search_path;
                search_path.emplace_back(root);
		        search_path.emplace_back(node);

		        while(node->expanded()){
		            node = select_child(node, action);
		            scratch_game->apply(action);
		            search_path.emplace_back(node);
		        } 

		        if(scratch_game->terminal(node->opp, node->my, action)){
                    node->raw_value = 1;
		            node->to_play = scratch_game->to_play();
		        }
		        else{
		            node->raw_value = expand_node(node, scratch_game, model);
		        }

		        backpropagate(search_path, node->raw_value);
		        //***************************************************
            }
            count += num_extra_visits; 
            if (sum + count >= num_simulations) {
                break;
            }
		}
        //cout << " total visited " << count << endl; // check simulation count
        sum += count;
        if (sum >= num_simulations) {
            break;
        }
	}
    // cout << "Total count of expand node : " << sum << endl; //check simulation count
    vector<float> completed_qvalues = qtransform(root);
    for (int i = 0; i < root->children.size(); ++i) {
        root->children[i]->complete_qvalue = completed_qvalues[i];
    }
    //sort(root->children.begin(), root->children.end(), [](const Node* a, const Node* b){return (a->visit_count == b->visit_count) ? ((a->prior + a->complete_qvalue) > (b->prior + b->complete_qvalue)) : (a->visit_count > b->visit_count);}); // 根據q value進行排序
	sort(root->children.begin(), root->children.end(), [](const Node* a, const Node* b){return (a->visit_count == b->visit_count) ? ((a->g + a->prior + a->complete_qvalue) > (b->g + b->prior + b->complete_qvalue)) : (a->visit_count > b->visit_count);}); // 根據q value進行排序
    selected_action = root->children[0]->action;
	return root;
}

Game* play_game(cppflow::model& model){
    #ifdef DEBUG
        HUMAN_PLAY = true;
    #endif
    Game* game = new Game();
    vector<BitBoard> board_info;
    uint64_t action = 1000;
    board_info = game->make_bitboard(-1);
    while(!game->terminal(board_info[1], board_info[0], action)){
        Node* root = seq_halving(game, model, action, board_info);
        game->apply(action);
        game->store_search_statistics(root);
        #ifdef DEBUG
            root->PrintTree(0, "", false, 10);
            game->print();//self play
            //cout << "action: " << Output(action) << endl;
            //cout << "value: " <<  game->child_values.back() << endl;
            //cout << "policy: " << endl;
            //print_1d_vec(game->child_polices.back());
            //cout << "network value: " << root->raw_value << endl;
            //cout << "predict win rate: " << (1 + game->child_values.back()) / 2 * 100 << "%" << endl << endl;
        #endif
        board_info = game->make_bitboard(-1);
        deleteTree(root);
    }
    return game;
}

void self_play(cppflow::model& model){
    time_t now = time(0);
    tm *ltm = localtime(&now);
    string ltm_mon = (1 + ltm->tm_mon) < 10 ? "0" + to_string(1 + ltm->tm_mon) : to_string(1 + ltm->tm_mon);
    string ltm_mday = ltm->tm_mday < 10 ? "0" + to_string(ltm->tm_mday) : to_string(ltm->tm_mday);
    string ltm_hour = ltm->tm_hour < 10 ? "0" + to_string(ltm->tm_hour) : to_string(ltm->tm_hour);
    string ltm_min = ltm->tm_min < 10 ? "0" + to_string(ltm->tm_min) : to_string(ltm->tm_min);
    string ltm_sec = ltm->tm_sec < 10 ? "0" + to_string(ltm->tm_sec) : to_string(ltm->tm_sec);
    string now_time = to_string(1900 + ltm->tm_year) +ltm_mon + ltm_mday + ltm_hour + ltm_min + ltm_sec;
    
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
        if (ProcessID != 0) {
            cout << "[ProcessID: " << ProcessID << "] " << setw(2) << round << ": Total Size: " << setw(3) << board_history.size() + game->SIZE << ": Board Size: " << setw(3) << game->SIZE << ": total cost time: " << setw(4) << setprecision(3) << (end-start) / 60.0 << " min, avg step time: " << setw(3) << setprecision(3) << (end - start) / double(board_history.size() + game->SIZE) << " s, " << " Winner is " << winner << "." << endl;
            // cout << "[ProcessID: " << ProcessID << "] " << round << ": Board Size: " << setw(5) << game->SIZE << ": total cost time: " << setw(6) << setprecision(3) << (end-start) / 60.0 << " min, avg step time: " << setw(3) << setprecision(3) << (end - start) / double(board_history.size() + game->SIZE) << " s, " << " Winner is " << winner << "." << endl;
        }
        else {
            cout << round << ": Board Size: " << setw(5) << game->SIZE << ": total cost time: " << setw(6) << setprecision(3) << (end-start) / 60.0 << " min, avg step time: " << setw(3) << setprecision(3) << (end - start) / double(board_history.size() + game->SIZE) << " s, " << " Winner is " << winner << "." << endl;
        }
        for(int i = 0; i < game->SIZE; ++i){//save training data
            if(board_history.size() == REPLAY_BUFFER_SIZE) break;
            vector<int> board_info(2*15*15);
            vector<vector<int>> board = game->make_image(i);
            if(i % 2 == 1){
                // print_board(board[0], 'x');
                // print_board(board[1], 'o');
                std::copy(board[0].begin(), board[0].end(), board_info.begin());
                std::copy(board[1].begin(), board[1].end(), board_info.begin() + board[0].size());
            }
            else{
                // print_board(board[1], 'o');
                // print_board(board[0], 'x');
                std::copy(board[1].begin(), board[1].end(), board_info.begin());
                std::copy(board[0].begin(), board[0].end(), board_info.begin() + board[1].size());
            }
            board_history.emplace_back(board_info);        

            policies_history.emplace_back(game->child_polices[i]);
            // print_board(game->child_polices[i], 'p');
            float value = i % 2 == winner ? 1.0 : -1.0;//！!！!
            #ifdef DEBUG
                cout << value << " " << game->child_values[i] << endl;
            #endif
            if (game->SIZE >= 254) value_history.emplace_back(0);
            else value_history.emplace_back((value + game->child_values[i]) / 2.0);
            // cout << value << " " << game->child_values[i] << " " << (value + game->child_values[i]) / 2.0 << endl;
            //value_history.emplace_back(value);
            //value_history.emplace_back(value);
            // if (ITER <= 25) {
            //     value_history.emplace_back(value);
            // }
            // else {
            //     value_history.emplace_back((value + game->child_values[i]) / 2.0);
            // }
            
            // if (ITER <= 50) {
            //     value_history.emplace_back((value + game->child_values[i]) / 2.0);
            // }
            // else {
            //     value_history.emplace_back(game->child_values[i]);
            // }

            //katago
            if(i + 1 == game->SIZE){
                vector<float> last_auxiliary_policy(225, 0.0);
                auxiliary_policies_history.emplace_back(last_auxiliary_policy);
            }
            else{
                auxiliary_policies_history.emplace_back(game->child_polices[i + 1]);
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
    
    write_2D_vector_to_file(policies_history, "./original_train_data/policies/" + now_time + ".history");
    write_2D_vector_to_file(auxiliary_policies_history, "./original_train_data/auxiliary_policies/" + now_time + ".history");
    write_2D_vector_to_file(board_history, "./original_train_data/board/" + now_time + ".history");
    write_1D_vector_to_file(value_history, "./original_train_data/value/" + now_time + ".history");
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
        char c;
        cin >> c;
        int n;
        if(c == 'p'){
            Node* root = seq_halving(game, model, action, board_info);
            #ifdef DEBUG
                
            #endif
            sort(root->children.begin(), root->children.end(), [](const Node* a, const Node* b){return (a->visit_count == b->visit_count) ? ((a->g + a->prior + a->complete_qvalue) > (b->g + b->prior + b->complete_qvalue)) : (a->visit_count > b->visit_count);}); // 根據q value進行排序
            //root->PrintTree(0, "", true);
            game->store_search_statistics(root);
            //delete root;
            game->apply(action);
            game->print();//human play
            game->print_history();
            cout << "action: " << Output(action) << endl;
            cout << game->child_values.back() << endl;
            cout << "predict win rate: " << (game->child_values.back() + 1) / 2 * 100 << "%" << endl << endl;
        }
        else if (c == 'u') {
            game->Undo();
            game->print();
            game->print_history();
        }
        else{
            cin >> n;
            c = toupper(c);
            if(c >= 'A' && c <= 'O' && n >= 1 && n <= 15){
                action = Input(c, n);
                cout << "Play action: " << action << endl; 
                game->apply(action);
                game->print();//human play
                game->print_history();
                cout << "action: " << Output(action) << endl;
            }
            else{
                Node* root = seq_halving(game, model, action, board_info);
                #ifdef DEBUG
                    root->PrintTree(0, "", true, 100);
                #endif
                game->store_search_statistics(root);
                //delete root;
                game->apply(action);
                game->print();//human play
                game->print_history();
                cout << "action: " << Output(action) << endl;
                cout << "predict win rate: " << (1 + game->child_values.back()) / 2 * 100 << "%" << endl << endl;
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

void evaluation(cppflow::model& curr_model, cppflow::model& pre_model, string iter){
    cout << "Start Evaluation" << endl;
    int curr_player;
    vector<int> curr_win;

    for(int i = 1; i <= EVALUATION_PLAY_COUNT; ++i){
        cout << "Round " << dec << i << ": " << endl; 
        curr_player = i <= (EVALUATION_PLAY_COUNT / 2) ? 0 : 1;
        Game* game = new Game();
        vector<BitBoard> board_info;
        uint64_t action = 1000;
        board_info = game->make_bitboard(-1);
        while(!game->terminal(board_info[1], board_info[0], action)){
            Node* root;
            //GUMBEL_PENALTY *= 0.7;
            if(game->SIZE % 2 == curr_player){
                // MAX_CONSIDERED_NUM = 32;
                root = seq_halving(game, curr_model, action, board_info);
            }
            else{
                // MAX_CONSIDERED_NUM = 64;
                root = seq_halving(game, pre_model, action, board_info);
            }
            game->apply(action);
            game->store_search_statistics(root);
            
            #ifdef DEBUG
            root->PrintTree(0, "", false, 1);
            game->print();//self play
            cout << "action: " << Output(action) << endl;
            cout << "value: " <<  game->child_values.back() << endl;
            cout << "policy: " << endl;
            print_1d_vec(game->child_polices.back());
            cout << "network value: " << root->raw_value << endl;
            cout << "predict win rate: " << (1 + game->child_values.back()) / 2 * 100 << "%" << endl << endl;
            #endif
            board_info = game->make_bitboard(-1);
            deleteTree(root);
        }
        game->print();//evaluation
        if (game->SIZE >= 224) {
            cout << "Draw" << endl;
            curr_win.push_back(0.5);
        }
        else if((game->SIZE - 1) % 2 == curr_player){
            cout << "New model win." << endl;
            curr_win.push_back(1);
        }
        else{
            cout << "New model lose" << endl;
            curr_win.push_back(0);
        }
        //delete game;
    }
    int win = reduce(curr_win.begin(), curr_win.end(), 0);
    cout << "New model win rate: " << (double(win) / EVALUATION_PLAY_COUNT) * 100.0 << "%" << endl;
    cout << "ELO data" << endl;
    for(int i = 1; i <= EVALUATION_PLAY_COUNT; ++i){
        string black, white;
        if (i <= (EVALUATION_PLAY_COUNT / 2)) {
            black = "iter_" + iter;
            white = "iter_" + to_string(stoi(iter) - 10);
        }
        else {
            white = "iter_" + iter;
            black = "iter_" + to_string(stoi(iter) - 10);
        }
        cout << "[Event \"KataGo Elo\"]" << endl;
        cout << "[Site \"NTNU\"]" << endl;
        cout << "[Date \"\"]" << endl;
        cout << "[Round \"\"]" << endl;
        if (i <= (EVALUATION_PLAY_COUNT / 2)) {
            black = "iter_" + iter;
            white = "iter_" + to_string(stoi(iter) - 10);
            cout << "[Black \"" << black << "\"]" << endl;
            cout << "[White \"" << white << "\"]" << endl;
            cout << "[Result \"" << 1 - curr_win[i - 1] << "-" << curr_win[i - 1] << "\"]" << endl;
        }
        else {
            white = "iter_" + iter;
            black = "iter_" + to_string(stoi(iter) - 10);
            cout << "[White \"" << white << "\"]" << endl;
            cout << "[Black \"" << black << "\"]" << endl;
            cout << "[Result \"" << curr_win[i - 1] << "-" << 1 - curr_win[i - 1] << "\"]" << endl;
        }
        cout << i << endl << endl;
    }
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
    //cppflow::model model("./save_model/model1");
    cout << "load model accept" << endl;

    ITER = stoi(string(argv[2]));
    cout << ITER << " iter start." << endl;
    ProcessID = stoi(string(argv[3]));
    cout << "Process  " << ProcessID << " start." << endl;
    if (ProcessID != 0) {
        REPLAY_BUFFER_SIZE /= ProcessCount;
    }
    cout << "REPLAY_BUFFER_SIZE: " << REPLAY_BUFFER_SIZE << endl;
    if(string(argv[1]) == "--self_play"){
        // if (ITER <= 20) GUMBEL_SIMPLE = 50;
        // else if (ITER <= 50) GUMBEL_SIMPLE = 30;
        // else if (ITER <= 70) GUMBEL_SIMPLE = 20;
        // else if (ITER <= 80) GUMBEL_SIMPLE = 10;
        // else GUMBEL_SIMPLE = 5;
        cout << "Gumble simple: " << GUMBEL_SIMPLE << endl;
        // cppflow::model c_model("./save_model/model1");
        // self_play(c_model);
        self_play(model);
        cout << "Table size: " << table.size() << ":" << EVALUATION_COUNT * REPLAY_BUFFER_SIZE << endl;
        cout << "Process " << ProcessID << " Finish." << endl; 
    }
    else if(string(argv[1]) == "--evaluation"){
        use_table = false;
        if (ITER <= 20) GUMBEL_SIMPLE = 70;
        else if (ITER <= 50) GUMBEL_SIMPLE = 50;
        else if (ITER <= 70) GUMBEL_SIMPLE = 40;
        else if (ITER <= 80) GUMBEL_SIMPLE = 30;
        else GUMBEL_SIMPLE = 10;
        cout << "Gumble simple: " << GUMBEL_SIMPLE << endl;

        EVALUATION_COUNT = 200;
        MAX_CONSIDERED_NUM = 64;
        cppflow::model p_model("./p_model");
        evaluation(model, p_model, string(argv[2]));

        // GUMBEL_SIMPLE = 10;
        // EVALUATION_COUNT = 200;
        // MAX_CONSIDERED_NUM = 80;
        // // cppflow::model p_model("./gumbel32_200/save_model/model160");
        // // cppflow::model c_model("./gumbel64_200/save_model/model160");
        // cppflow::model p_model("./save_model/model1");
        // cppflow::model c_model("./save_model/model10");
        // evaluation(c_model, p_model, string(argv[2]));
        
    }
    else if(string(argv[1]) == "--human_play"){
        //GUMBEL_PENALTY = 0.1;
        EVALUATION_COUNT = 500;
        MAX_CONSIDERED_NUM = 64; //Top-k-Gumbel
        GUMBEL_SIMPLE = 1;
        use_mixed_value = true;
        //cppflow::model p_model("./save_model/model1");
        //cppflow::model p_model("./gumbel64_200/save_model/model160");
        cppflow::model p_model("./p_model");
        human_play(p_model);
    }
    
    return 0;
}