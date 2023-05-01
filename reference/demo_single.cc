#include <iostream>
#include <algorithm>
#include <ctime>
#include <cmath>
#include <cfloat>
#include <numeric>
#include <random>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
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
using namespace std;
#define BOARD_SIZE 15
#define RESET "\033[0m"
#define RED  "\033[31m" 
//36
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



//g++ -march=native demo_single.cc -g -ltensorflow -fopenmp -O3 -o demo_single && "/home/azon/Documents/OOG/OOG-BITBOARD/"demo_single | tee output.log
void init()
{
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

void Output(uint64_t pos)
{
    int i = pos / 16;
    int j = pos % 16;
    cout << char('A'+j-1);
    cout << 15-i;
    cout << std::endl;
}

uint64_t Input(char c, int n)
{
    return common_pos_to_bitboard_pos[(c - 'A') + (n - 1)*15];
}

void write_2D_float_vector_to_file(vector<vector<float>>& myVector, string filename)
{
    std::ofstream ofs(filename, std::ios::out | std::ofstream::binary);
    std::ostream_iterator<float> osi{ofs,", "};
    for(int i = 0; i < myVector.size(); ++i){
      std::copy(myVector.at(i).begin(), myVector.at(i).end(), osi);
      ofs << '\n';
    }
}

void write_2D_int_vector_to_file(vector<vector<int>>& myVector, string filename)
{
    std::ofstream ofs(filename, std::ios::out | std::ofstream::binary);
    std::ostream_iterator<int> osi{ofs,", "};
    for(int i = 0; i < myVector.size(); ++i){
      std::copy(myVector.at(i).begin(), myVector.at(i).end(), osi);
      ofs << '\n';
    }
}

void write_1D_int_vector_to_file(const vector<int>& myVector, string filename)
{
    std::ofstream ofs(filename, std::ios::out | std::ofstream::binary);
    std::ostream_iterator<int> osi{ofs,", "};
    std::copy(myVector.begin(), myVector.end(), osi);
}

void write_1D_float_vector_to_file(const vector<float>& myVector, string filename)
{
    std::ofstream ofs(filename, std::ios::out | std::ofstream::binary);
    std::ostream_iterator<float> osi{ofs,", "};
    std::copy(myVector.begin(), myVector.end(), osi);
}

void Print_vec(vector<int> board){
    cout << "black" << endl;
    for(int i = 0; i < 15; ++i){
        for(int j = 0; j < 15; ++j){
            cout << board[i * 15 + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    cout << "white" << endl;
    for(int i = 15; i < 30; ++i){
        for(int j = 0; j < 15; ++j){
            cout << board[i * 15 + j] << " ";
        }
        cout << endl;
    }
}
//------------Globel Value----------------
//cppflow::model model("./c_model");

// Serialized config options (example of 30% memory fraction)
// Read more to see how to obtain the serialized options
bool bolzman_noise = true, dirichlet_noise = true;
int EVALUATION_COUNT = 500;
int noise_limit_step = 10;
int THREAD_NUM = 10;
int SELF_PLAY_FILE_COUNT = 1500;
int SELF_PLAY_COUNT = 50;
double VIRTUAL_LOSS = 5.0;
double C_PUCT {1.0}; // 4.0
double EPS {1e-8};
//------------Globel Value----------------
void predict(BitBoard my, BitBoard opp, BitBoard mov, vector<uint64_t>& Vs, vector<float>& policies,int level, double& value, cppflow::model& model){   
    vector<int> board(15 * 15 * 2, 1);
    uint64_t pos;
    double legal_move_count = 0.0;
    while(my){
        pos = my.ls1b();
        board[bitboard_pos_to_common_pos[pos] * 2] = 0;
    }
    while(opp){
        pos = opp.ls1b();
        board[bitboard_pos_to_common_pos[pos] * 2 + 1] = 0;
    }
    while(mov){
        pos = mov.ls1b();
        ++legal_move_count;
    }
    
    auto TF_vec = cppflow::tensor(board, {15,15,2});
    auto TF_input = cppflow::cast(TF_vec, TF_UINT8, TF_FLOAT);
    TF_input = cppflow::expand_dims(TF_input, 0);

    auto output = model({{"serving_default_input_1:0", TF_input}},{"StatefulPartitionedCall:0", "StatefulPartitionedCall:1","StatefulPartitionedCall:2"});
    policies.resize(225, 0.0);
    uint64_t legal_move;

    //dirichlet分佈
    vector<double> dirichlet(225, 0.0);
    if(dirichlet_noise && level == 0){
        //cout << "add noise" << endl;
        double a = (Vs.size() == 1) ? 0.04 : (10.0 / legal_move_count);//9.0
        std::gamma_distribution<double> gamma(a);
        std::random_device rd;
        double sum = 0;
        for (int i = 0; i < 225; ++i) {
            dirichlet[i] = gamma(rd);
            sum += dirichlet[i];
        }
        for (int i=0; i < 225; ++i) {
            dirichlet[i] = dirichlet[i]/sum;
        }
    }
    //-----------------
    for(auto& pos: Vs){
        legal_move = bitboard_pos_to_common_pos[pos];
        if(dirichlet_noise && level == 0){
            policies[legal_move] = output[0].get_data<float>()[legal_move] * 0.75 + dirichlet[legal_move] * 0.25;
        }
        else{
            policies[legal_move] = output[0].get_data<float>()[legal_move];
        }
    }

    float sum = accumulate(policies.begin(), policies.end(), 0.0);
    if(sum > 0){
        for(auto& p:policies){
            p /= sum;
        }
    }
    else{
        //cout << "All valid moves were masked, doing a workaround. moves count:" << legal_move_count << endl;
        if(Vs.size() == 1){
            //cout << "Just have move: " << bitboard_pos_to_common_pos[Vs[0]] << endl;
            policies[bitboard_pos_to_common_pos[Vs[0]]] = 1.0;
        }
        else{
            cout << "WTF!!" << endl;
            policies = output[0].get_data<float>();
        }
        /*
        for(int i = 1; i <= policies.size(); ++i){
            legal_move = bitboard_pos_to_common_pos[i - 1];
            cout << (output[0].get_data<float>()[legal_move] != 0 ? 1 : 0) << " ";
            if(i % 15 == 0) cout << endl;
        }
        for(auto& pos: Vs){
            legal_move = bitboard_pos_to_common_pos[pos];
            cout << "pos: " << legal_move << endl;
            cout << "model output:" << output[0].get_data<float>()[legal_move] << endl;
        }
        cout << "level: " << level << endl;
        
        exit(1);
        */
    }
    value = output[2].get_data<float>()[0];
}

class Node {
public:
    BitBoard my, opp, moves, myTzone, oppTzone;
    vector<float> Ps; // 策略
    vector<uint64_t> Vs; //合法走步
    unordered_map<int, double> Qs; // 累積價值 也就是Q(v)的累加  
    unordered_map<int, double> Ns; // child的試驗次數
    double N;//自己的試驗次數
    int level; // 深度
    uint64_t pos; //當前盤面所下的位置
    unordered_map<int, Node*> child_nodes;// children
    double Evaluate(cppflow::model model);
    void PrintTree(vector<bool>& flag, bool isLast);

    Node(BitBoard _my,BitBoard _opp,BitBoard _moves,BitBoard _myTzone,BitBoard _oppTzone, uint64_t pos, int _level);
};


Node::Node(BitBoard _my,BitBoard _opp,BitBoard _moves,BitBoard _myTzone,BitBoard _oppTzone, uint64_t _pos, int _level)
:
    N(0), pos(_pos), level(_level),
    my(_my),
    opp(_opp),
    moves(_moves),
    myTzone(_myTzone),
    oppTzone(_oppTzone){}

double Node::Evaluate(cppflow::model model){
    auto node = this;
    double value;
    uint64_t pos, feat, s_h, s_b, s_v, s_s;
    if(node->level != 0){ // 如果是root直接去擴充節點
        if(!(node->opp & node->my)){//Draw 情況 1- 遊戲結束時
            value = 0.0;
            node->N += 1;
            return value;
        }
        
        feat = feature(node->opp, node->my, node->pos);
        s_h = state[_pext_u64(feat, pext_h)];
        s_b = state[_pext_u64(feat, pext_b)];
        s_v = state[_pext_u64(feat, pext_v)];
        s_s = state[_pext_u64(feat, pext_s)];
        if((s_h | s_b | s_v | s_s) & 0x80000){ //L5 情況 1 - 遊戲結束時  L4,l50xC0000
            value = -1.0;
            node->N += 1;
            return -value;
        }
    }
    
    if(node->Vs.empty()){// 情況 2 - 當子節點不存在時 
        BitBoard cTzone = node->myTzone & node->my & node->opp;  //迫著
        bool threat_flag = false;
        while(cTzone){//我方可連五
            pos = cTzone.ls1b();
            feat = feature(node->my, node->opp, pos);
            s_h = state[_pext_u64(feat, pext_h)];
            s_b = state[_pext_u64(feat, pext_b)];
            s_v = state[_pext_u64(feat, pext_v)];
            s_s = state[_pext_u64(feat, pext_s)];
            if((s_h | s_b | s_v | s_s) & 0x80000){ //L5
                node->Vs.push_back(pos);
                threat_flag = true;
                break;
            }
        }

        if(!threat_flag){//對方連5就檔
            cTzone = node->oppTzone & node->my & node->opp;
            while(cTzone){
                pos = cTzone.ls1b();
                feat = feature(node->opp, node->my, pos);
                s_h = state[_pext_u64(feat, pext_h)];
                s_b = state[_pext_u64(feat, pext_b)];
                s_v = state[_pext_u64(feat, pext_v)];
                s_s = state[_pext_u64(feat, pext_s)];
                if((s_h | s_b | s_v | s_s) & 0x80000){ //L5
                    node->Vs.push_back(pos);
                    threat_flag = true;
                    break;
                }
            }
        }

        if(!threat_flag){//我方可連四
            cTzone = node->myTzone & node->my & node->opp;;
            while(cTzone){
                pos = cTzone.ls1b();
                feat = feature(node->my, node->opp, pos);
                s_h = state[_pext_u64(feat, pext_h)];
                s_b = state[_pext_u64(feat, pext_b)];
                s_v = state[_pext_u64(feat, pext_v)];
                s_s = state[_pext_u64(feat, pext_s)];
                if((s_h | s_b | s_v | s_s) & 0xC0000){ //L5, L4
                    node->Vs.push_back(pos);
                    threat_flag = true;
                    break;
                }
            }
        }

        if(!threat_flag){
            BitBoard mov = node->moves & node->my & node->opp;
            while(mov){
                pos = mov.ls1b();
                node->Vs.push_back(pos);
            }
        }

        predict(node->my, node->opp, node->moves, node->Vs, node->Ps, node->level, value, model);
        return -value;
    }

    //情況 3 - 當子節點存在時 PUCT公式
    uint64_t best_move;
    double cur_best {-DBL_MAX}; 
    double u, n_forced;
    for(auto& i : node->Vs){
        n_forced = sqrt(2.0 * node->Ps[bitboard_pos_to_common_pos[i]] * (node->N ? EPS : node->N));
        if(node->level == 0 && node->Ns[i] < n_forced){//強迫搜尋第一層所有節點
            best_move = i;
            break;
        }
        
        if(node->Qs.find(i) == node->Qs.end()){
            u = C_PUCT * node->Ps[bitboard_pos_to_common_pos[i]] * sqrt(node->N + EPS);
        }
        else{
            u = node->Qs[i] + C_PUCT * node->Ps[bitboard_pos_to_common_pos[i]] * sqrt(node->N) / (1 + node->Ns[i]);
        }
        if(u > cur_best){
            cur_best = u;
            best_move = i;
        }
    }
    
    if(node->child_nodes.find(best_move) == node->child_nodes.end()){
        pos = best_move;
        BitBoard new_my = node->my;
        BitBoard new_moves = node->moves;
        feat = feature(new_my, node->opp, pos);
        s_h = state[_pext_u64(feat, pext_h)];
        s_b = state[_pext_u64(feat, pext_b)];
        s_v = state[_pext_u64(feat, pext_v)];
        s_s = state[_pext_u64(feat, pext_s)];
        BitBoard newTzone = node->myTzone
                            |Tboard_h[(s_h & 0x3F00) | pos]
                            |Tboard_b[(s_b & 0x3F00) | pos]
                            |Tboard_v[(s_v & 0x3F00) | pos]
                            |Tboard_s[(s_s & 0x3F00) | pos];
        new_my.append(pos);
        new_moves = new_moves.mind(pos);

        Node* new_node = new Node(node->opp, new_my, new_moves, node->oppTzone, newTzone, pos, node->level + 1);
        node->child_nodes[best_move] = new_node;
    }
    //node->Ns[best_move] += VIRTUAL_LOSS;

    value = ((node->child_nodes[best_move])->Evaluate(model));

    //node->Ns[best_move] -= VIRTUAL_LOSS;
    if(node->Qs.find(best_move) == node->Qs.end()){
        node->Qs[best_move] = value;
        node->Ns[best_move] = 1;
    }
    else{
        node->Qs[best_move] = (node->Ns[best_move] * node->Qs[best_move] + value) / (node->Ns[best_move] + 1);
        node->Ns[best_move] += 1;
    }
    node->N += 1;
    return -value;
}

void Node::PrintTree(vector<bool>& flag, bool isLast = false){
    auto node = this;
    // Condition when node is None
    if (node == NULL)
        return;
     
    for (int i = 1; i < node->level; ++i) {
        if (flag[i] == true) {
            cout << "│ " << "   ";
        }
        else {
            cout << "    ";
        }
    }
     
    if (node->level == 0)
        cout << "level: " << node->level << " move: " << node->pos << " N: " << node->N << " move size: " << node->Vs.size() << endl;
    else if (isLast) {
        cout << "└── " << "level: " << node->level << " move: " << node->pos << " N: " << node->N << " move size: " << node->Vs.size() << endl;
        flag[node->level] = false;
    }
    else {
        cout << "├── " << "level: " << node->level << " move: " << node->pos << " N: " << node->N << " move size: " << node->Vs.size() << endl;
    }
 
    int it = 0;
    int cnt = node->child_nodes.size();
    for(auto child: node->child_nodes){
        child.second->PrintTree(flag, it == (cnt - 1));
        it++;
    }
        
    flag[node->level] = true;
}

int boltzman(vector<double>& child_n, double temp){
    double sum_n = 0.0;
    for(auto& n: child_n){
        n = pow(n, 1.0/temp);
        sum_n += n;
    }
    for(auto& n: child_n){
        n /= sum_n;
    }
    random_device rd;
    mt19937 gen(rd());
    discrete_distribution<> d(child_n.begin(), child_n.end());
    int idx = d(gen);
    return idx;
}

int MCTS(BitBoard my, BitBoard opp, BitBoard moves, BitBoard myTzone, BitBoard oppTzone, int Simulation_Count, vector<float>& policies,float& Q_value, double temperature, int step, cppflow::model& model){
    Node* root = new Node(my, opp, moves, myTzone, oppTzone, 0, 0); 
    for(int i = 0; i < Simulation_Count; ++i){
        root->Evaluate(model);
        //root.PrintTree({}, true);
        //cout << endl;
    }
    vector<double> child_n;
    vector<int> child_idx;
    for(auto& N: root->Ns){
        child_idx.push_back(N.first);
        child_n.push_back(N.second);
    }
    
    int idx;
    if(step < noise_limit_step && temperature != 0.0 && bolzman_noise){//第7步後就不要隨機了
        idx = boltzman(child_n, temperature);
    }
    else{
        idx = max_element(child_n.begin(), child_n.end()) - child_n.begin();
    }
    idx = child_idx[idx];

    double n_forced; 
    double policy_sum = 0.0;
    Q_value = 0.0;
    for(int i = 0; i < policies.size(); ++i){
        policies[i] = 0.0;
    }

    for(auto& i: root->Vs){
        int iidx = bitboard_pos_to_common_pos[i];
        n_forced = sqrt(2.0 * root->Ps[iidx] * (Simulation_Count - 1));
        if(root->Ns[i] > n_forced){// >n_forced的值才被存進去
            policies[iidx] = root->Ps[iidx];
            policy_sum += policies[iidx];
        }
        
    }

    int max_child_n_idx = max_element(child_n.begin(), child_n.end()) - child_n.begin();
    int max_chile_idx = child_idx[max_child_n_idx];
    Q_value = -1.0 * root->Qs[max_chile_idx];

    policy_sum = (policy_sum == 0) ? 1 : policy_sum;
    for(int i = 0; i < policies.size(); ++i){
        policies[i] /= policy_sum;
    }
    
    return idx;
}

struct Game
{
    vector <char> board;
    vector <float> policies;
    float Q_value;
    vector <double> Ns;
    BitBoard black, white, moves, blackTzone, whiteTzone;
    BitBoard black_record[256], white_record[256], moves_record[256], blackTzone_record[256], whiteTzone_record[256];
    int size {0}, score {0}, score_record[256];


    Game():board(256,'-'){}
    void Start();
    void Move(uint64_t pos);
    void Undo();
    uint64_t Play_MCTS(int simulation_count, double temp,cppflow::model& model);
    bool isEnd(uint64_t pos);

    void Print_Board(uint64_t mov);
};

void Game::Start()
{
    black.init();
    white.init();
    moves.initZero();
    blackTzone.initZero();
    whiteTzone.initZero();
    policies.resize(225, 0);
    policies[bitboard_pos_to_common_pos[24]] = 1;
    Q_value = 0.0;
    size = 0;
    score = 0;
}

void Game::Move(uint64_t pos)
{
    black_record[size] = black;
    white_record[size] = white;
    moves_record[size] = moves;
    score_record[size] = score;
    blackTzone_record[size] = blackTzone;
    whiteTzone_record[size] = whiteTzone;
    uint64_t feat;
    if(size%2 == 0)
    {
        board[pos] = 'o';
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

        //cout << TSS(black,white,blackTzone,6);
        //cout << endl;
    }
    else
    {
        board[pos] = 'x';
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
    //moves.print();
    score += 0;/**by feat**/
    score = -score;
    size += 1;
}

void Game::Undo()
{
    if(size > 0)
    {
        size -= 1;
        black = black_record[size];
        white = white_record[size];
        moves = moves_record[size];
        score = score_record[size];
        blackTzone = blackTzone_record[size];
        whiteTzone = whiteTzone_record[size];
    }
}

uint64_t Game::Play_MCTS(int simulation_count, double temp, cppflow::model& model)
{
    uint64_t pos = 0;
    if(size%2 == 0)
    {
        if (size == 0)
        {
            moves.append(24);
        }
        pos = MCTS(black, white, moves, blackTzone, whiteTzone, simulation_count, policies, Q_value, temp, size, model);
    }
    else
    {
        pos = MCTS(white, black, moves, whiteTzone, blackTzone, simulation_count, policies, Q_value, temp, size, model);
        /*
        pos = 0;
        if(size == 1)
        {
            BitBoard tmp;
            tmp.append(114);
            if(!(black ^ tmp))
            {
                pos = 117;
                moves.append(117);
            }
            tmp.append(114);
            tmp.append(24);
            if(!(black ^ tmp))
            {
                pos = 72;
                moves.append(72);
            }
            tmp.append(24);
            tmp.append(126);
            if(!(black ^ tmp))
            {
                pos = 123;
                moves.append(123);
            }
            tmp.append(126);
            tmp.append(216);
            if(!(black ^ tmp))
            {
                pos = 168;
                moves.append(168);
            }
            tmp.append(216);
        }
        if(pos == 0)
        {
            pos = MCTS(white, black, moves, whiteTzone, blackTzone, simulation_count, policies, Q_value, temp, size, model);
        }
        */
    }
    Move(pos);
    return pos;
}

bool Game::isEnd(uint64_t pos){
    if(!(black & white)){//Draw 情況 1- 遊戲結束時
        return true;
    }
    if(size % 2 == 0){
        uint64_t feat = feature(white, black, pos);
        uint64_t s_h = state[_pext_u64(feat, pext_h)];
        uint64_t s_b = state[_pext_u64(feat, pext_b)];
        uint64_t s_v = state[_pext_u64(feat, pext_v)];
        uint64_t s_s = state[_pext_u64(feat, pext_s)];
        if((s_h | s_b | s_v | s_s) & 0x80000){ //L5 情況 1 - 遊戲結束時
            return true;
        }
    }
    else{
        uint64_t feat = feature(black, white, pos);
        uint64_t s_h = state[_pext_u64(feat, pext_h)];
        uint64_t s_b = state[_pext_u64(feat, pext_b)];
        uint64_t s_v = state[_pext_u64(feat, pext_v)];
        uint64_t s_s = state[_pext_u64(feat, pext_s)];
        if((s_h | s_b | s_v | s_s) & 0x80000){ //L5 情況 1 - 遊戲結束時
            return true;
        }
    }
    

    return false;
}

void Game::Print_Board(uint64_t mov = -1){
    cout << "   ";
    for(char i='A';i<='O';i++)
        cout << i << ' ';
    cout << endl;

    for(int i=1;i<=15;i++){
        cout << setw(2) << i << '|';
        for(int j=1;j<=15;j++){
            if(mov != -1 && (16*i+j-16) == mov){
                cout << RED << board[16*i+j-16] << RESET << ' ';
            }
            else{
                cout << board[16*i+j-16] << ' ';
            }
        }
        cout << endl;
    }
}

double play(Game& game, vector<vector<int>>& board_history, vector<vector<float>>& policies_history, vector<vector<float>>& ememy_policies_history, vector<float>& value_history, cppflow::model& model){
    int i, winner = 0;//draw : 0, black : 1, white : 2
    clock_t t1, t2;
    t1 = clock();
    uint64_t pos = 1000;
    vector<float> one_play_q_value;
    vector<vector<float>> one_play_policy;
    for(i = 0; i < 225; ++i){
        vector<int> board(15 * 15 * 2, 1);
        BitBoard my = (game.size % 2 == 0) ? game.black : game.white;
        BitBoard opp = (game.size % 2 == 0) ? game.white : game.black;

        while(my){
            uint64_t position = my.ls1b();
            board[bitboard_pos_to_common_pos[position]] = 0;
        }
        while(opp){
            uint64_t position = opp.ls1b();
            board[225 + bitboard_pos_to_common_pos[position]] = 0;
        }
        vector<float> policies(game.policies.begin(), game.policies.end());
        board_history.push_back(board);
        policies_history.push_back(policies);

        one_play_q_value.push_back(game.Q_value);
        one_play_policy.push_back(policies);

        if(i != 0 && game.isEnd(pos)){
            winner = (game.size % 2 == 0) ? 2 : 1;
            winner = (game.size == 224) ? 0 : winner;
            break;
        }

        pos = game.Play_MCTS(EVALUATION_COUNT, 1.0, model);
        //game.Print_Board();
    }
    t2 = clock();
    //dirichlet_noise = true;
    float value = (winner == 1) ? 1.0 : -1.0;
    value = (winner == 0) ? 0 : value;

    for(i = 0; i < one_play_q_value.size(); ++i){
        //Print_vec(board_history[board_history.size() - 1]);
        //cout << one_play_q_value[i] << " " <<  value << endl << endl;
        //value_history.push_back(value);
        value_history.push_back((value + one_play_q_value[i]) / 2); // 50以後
        value = -value;
        if(i == one_play_q_value.size() - 1){
            vector<float> end_policies(225, 1.0/225.0);
            ememy_policies_history.push_back(end_policies);
            break;
        }
        ememy_policies_history.push_back(one_play_policy[i + 1]);
    }
    //cout << endl;
    return (double)((t2 - t1) / (double)i)/CLOCKS_PER_SEC;
}

void self_play(int Game_Count, cppflow::model model){
    vector<vector<int>> board_history;
    vector<vector<float>> policies_history;
    vector<vector<float>> ememy_policies_history;
    vector<float> value_history;
    float sec;
    clock_t t1, t2;
    int i = 1;

    while(board_history.size() < SELF_PLAY_FILE_COUNT){
        Game game;
        game.Start();
        t1 = clock();
        sec = play(game, board_history, policies_history, ememy_policies_history, value_history, model);
        t2 = clock();
        cout << "Self Play Round " << (i++) << " : total step : "  << game.size << ", total cost " << float((t2 - t1) / CLOCKS_PER_SEC) << " sec, each step cost " <<  sec << " second." << endl;
    }

    // write file
    cout << board_history.size() << " " << policies_history.size() << " " << ememy_policies_history.size() << " " << value_history.size() << endl; 
    time_t now = time(0);
    tm *ltm = localtime(&now);
    string ltm_mon = (1 + ltm->tm_mon) < 10 ? "0" + to_string(1 + ltm->tm_mon) : to_string(1 + ltm->tm_mon);
    string ltm_mday = ltm->tm_mday < 10 ? "0" + to_string(ltm->tm_mday) : to_string(ltm->tm_mday);
    string ltm_hour = ltm->tm_hour < 10 ? "0" + to_string(ltm->tm_hour) : to_string(ltm->tm_hour);
    string ltm_min = ltm->tm_min < 10 ? "0" + to_string(ltm->tm_min) : to_string(ltm->tm_min);
    string ltm_sec = ltm->tm_sec < 10 ? "0" + to_string(ltm->tm_sec) : to_string(ltm->tm_sec);
    string now_time = to_string(1900 + ltm->tm_year) +ltm_mon + ltm_mday + ltm_hour + ltm_min + ltm_sec;
    
    write_2D_float_vector_to_file(policies_history, "./train_data/policies/" + now_time + ".history");
    write_2D_float_vector_to_file(ememy_policies_history, "./train_data/ememy_policies/" + now_time + ".history");
    write_2D_int_vector_to_file(board_history, "./train_data/board/" + now_time + ".history");
    //write_1D_int_vector_to_file(value_history, "./train_data/value/" + now_time + ".history");
    write_1D_float_vector_to_file(value_history, "./train_data/value/" + now_time + ".history");
    return;
}

void debug_play(cppflow::model& model){
    vector<vector<int>> board_history;
    vector<vector<float>> policies_history;
    vector<vector<float>> ememy_policies_history;
    vector<float> value_history;
    float sec;
    clock_t t1, t2;
    for(int i = 1; i <= 1; ++i){
        Game game;
        game.Start();
        t1 = clock();
        sec = play(game, board_history, policies_history, ememy_policies_history, value_history, model);
        t2 = clock();
        cout << "Self Play Round " << i << " : total step : "  << game.size << ", total cost " << float((t2 - t1) / CLOCKS_PER_SEC) << " sec, each step cost " <<  sec << " second." << endl;
    }
    /*
    Game game;
    game.Start();
    int i, winner = 0;//draw : 0, black : 1, white : 2
    clock_t t1, t2;
    t1 = clock();
    uint64_t pos = 1000;
    for(i = 0; i < 225; ++i){
        if(i != 0 && game.isEnd(pos)){
            winner = (game.size % 2 == 0) ? 2 : 1;
            winner = (game.size == 224) ? 0 : winner;
            break;
        }
        
        pos = game.Play_MCTS(EVALUATION_COUNT, 1.0, model);
        game.Print_Board();
        //cout << "This step cost time: " << (float)((t4 - t3) / (float)CLOCKS_PER_SEC) << endl;
    }
    t2 = clock();

    int value = (winner == 1) ? 1 : -1;
    value = (winner == 0) ? 0 : value;
    
    if(winner == 1){
        cout << "black win!" << endl;
    }
    else if(winner == 2){
        cout << "white win!" << endl;
    }
    else{
        cout << "draw!" << endl;
    }
    cout << "cost time: " << (float)(((t2 - t1) / (float)i) / CLOCKS_PER_SEC) << endl;
    */
    return;
}

void human_play(cppflow::model& model){
    Game game;
    game.Start();
    int i, winner = 0;//draw : 0, black : 1, white : 2
    clock_t t1, t2;
    t1 = clock();
    uint64_t pos = 1000;
    char c;
    int n;
    while(cin >> c >> n){
        if(game.isEnd(pos)){
            winner = (game.size % 2 == 0) ? 2 : 1;
            winner = (game.size == 224) ? 0 : winner;
            break;
        }

        if(c >= 'A' && c <= 'O'){     
            pos = Input(c, n);
            game.Move(pos);
            game.Print_Board();
        }
        else{
            pos = game.Play_MCTS(EVALUATION_COUNT, 0.0, model);
            game.Print_Board();
            
        }
        //cout << "This step cost time: " << (float)((t4 - t3) / (float)CLOCKS_PER_SEC) << endl;
    }
    t2 = clock();

    int value = (winner == 1) ? 1 : -1;
    value = (winner == 0) ? 0 : value;
    
    if(winner == 1){
        cout << "black win!" << endl;
    }
    else if(winner == 2){
        cout << "white win!" << endl;
    }
    else{
        cout << "draw!" << endl;
    }
    cout << "cost time: " << (float)(((t2 - t1) / (float)i) / CLOCKS_PER_SEC) << endl;
    return;
}

void evaluation(int n, cppflow::model& curr_model, cppflow::model& pre_model){
    clock_t t1, t2;
    t1 = clock();
    int i;
    double curr_win = 0, pre_win = 0;
    for(int count = 0; count < n; ++count){
        Game game;
        game.Start();
        int winner = 0;//draw : 0, black : 1, white : 2
        uint64_t pos = 1000;
        if(count < n / 2){
            for(i = 0; i < 225; ++i){
                if(i != 0 && game.isEnd(pos)){
                    winner = (game.size % 2 == 0) ? 2 : 1;
                    winner = (game.size == 224) ? 0 : winner;
                    break;
                }
                if(i % 2 == 0){
                    //cout << "curr move" << endl;
                    pos = game.Play_MCTS(EVALUATION_COUNT, 1.0, curr_model);
                }
                else{
                    //cout << "pre move" << endl;
                    pos = game.Play_MCTS(EVALUATION_COUNT, 1.0, pre_model);
                }
            }
            game.Print_Board();

            int value = (winner == 1) ? 1 : -1;
            value = (winner == 0) ? 0 : value;
            
            if(winner == 1){
                curr_win++;
                cout << "Evaluation " << count + 1 <<  ": new model win! total step: " << i << endl;
            }
            else if(winner == 2){
                pre_win++;
                cout << "Evaluation " << count + 1 <<  ": new model lose! total step: " << i << endl;
            }
            else{
                cout << "Evaluation " << count + 1 <<  ": draw! total step: " << i << endl;
            }
        }
        else{
            for(i = 0; i < 225; ++i){
                if(i != 0 && game.isEnd(pos)){
                    winner = (game.size % 2 == 0) ? 2 : 1;
                    winner = (game.size == 224) ? 0 : winner;
                    break;
                }
                if(i % 2 == 0){
                    pos = game.Play_MCTS(EVALUATION_COUNT, 1.0, pre_model);
                }
                else{
                    pos = game.Play_MCTS(EVALUATION_COUNT, 1.0, curr_model);
                }
                //game.Print_Board();
            }
            game.Print_Board();
            int value = (winner == 1) ? 1 : -1;
            value = (winner == 0) ? 0 : value;
            
            if(winner == 1){
                pre_win++;
                cout << "Evaluation " << count + 1<<  ": new model lose! total step: " << i << endl;
            }
            else if(winner == 2){
                curr_win++;
                cout << "Evaluation " << count + 1 <<  ": new model win! total step: " << i << endl;
            }
            else{
                cout << "Evaluation " << count + 1 << "draw! total step: " << i << endl;
            }
        }
    }
    cout << "New model Winning rate: " << (double)(curr_win / n) * 100.0<< "%" << endl;
    t2 = clock();
    cout << "cost time: " << (float)(((t2 - t1) / (float)i) / CLOCKS_PER_SEC) << endl;
    return;
}

int main(int argc, char *argv[])
{   
    //vector<uint8_t> config{0x32,0xb,0x9,0x34,0x33,0x33,0x33,0x33,0x33,0xd3,0x3f,0x20,0x1};//30%
    //vector<uint8_t> config{0x32,0xb,0x9,0x9a,0x99,0x99,0x99,0x99,0x99,0xd9,0x3f,0x20,0x1};//40%
    vector<uint8_t> config{0x32,0xb,0x9,0x0,0x0,0x0,0x0,0x0,0x0,0xe0,0x3f,0x20,0x1};//50%
    // Create new options with your configuration
    TFE_ContextOptions* options = TFE_NewContextOptions();
    TFE_ContextOptionsSetConfig(options, config.data(), config.size(), cppflow::context::get_status());
    // Replace the global context with your options
    cppflow::get_global_context() = cppflow::context(options);
    
    cppflow::model model("./c_model");
    srand(time(NULL));
    init();
    clock_t t1,t2;
    t1 = clock();
    
    self_play(SELF_PLAY_FILE_COUNT, model);
    //human_play(model);
    //debug_play(model);
    t2 = clock();
    cout << "total Self Play cost time : " << float((t2 - t1) / CLOCKS_PER_SEC) << endl;

    return 0;
}
