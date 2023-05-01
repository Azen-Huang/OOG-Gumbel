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

using namespace std;

// Generates gumbel noise with the given shape.
vector<float> GenerateGumbelNoise(int shape) {
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

// Implements the Gumbel-Top-K trick.
vector<int> GumbelTopK(const vector<float>& logits, int k) {
    int n = logits.size();
    vector<double> logits_with_noise = logits;
    vector<double> gumbel_noise = GenerateGumbelNoise(n);
    for (int i = 0; i < n; ++i)
    {
        logits_with_noise[i] += gumbel_noise[i];
        //cout << logits_with_noise[i] << ", ";
    }
    //cout << endl;

    vector<int> indices(n);
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(),
            [&](int a, int b) { return logits_with_noise[a] > logits_with_noise[b]; });

    vector<int> top_k_indices(indices.begin(), indices.begin() + k);
    return top_k_indices;
}

int m = 16;
int n = 20;

float expand_node(Node* node, Game* game, cppflow::model& model){
    Network network_output = predict(node->my, node->opp, model);
    node->to_play = game->to_play();
    vector<uint64_t> legal_actions = game->legal_actions(node->my, node->opp, node->move, node->myTzone, node->oppTzone);
    float policy_sum = 0.0;
    
    vector<float> policies(225, 0.0);
    policy_sum = 0.0;
    for(auto& pos: legal_actions){
        uint64_t legal_move = bitboard_pos_to_common_pos[pos];
        policies[legal_move] = network_output.policies[legal_move];
        policy_sum += policies[legal_move];
    }
    //////////////////gumbel-top-k/////////////
    vector<float> gumbel_policy = network_output.policies;
    vector<float> gumbel_noise = GenerateGumbelNoise(225);
    for (int i = 0; i < n; ++i) {
        logits_with_noise[i] += gumbel_noise[i];
    }

    vector<int> indices(n);
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(),
            [&](int a, int b) { return logits_with_noise[a] > logits_with_noise[b]; });

    vector<int> top_k_indices(indices.begin(), indices.begin() + m); //這些點需要展開
    //////////////////////////////////////////

    /*
    //////////////////soft max////////////////
    vector<float> soft_max_policy(225, 0.0);
    int MAX = *max_element(network_output.policies.begin(), network_output.policies.end());
    for(int i = 0; i < 225; ++i){
        soft_max_policy[i] = exp(network_output.policies[i] - MAX);
        policy_sum += soft_max_policy[i];
    }
    for(int i = 0; i < 225; ++i){
        soft_max_policy[i] = soft_max_policy[i] / policy_sum;
    }
    //////////////////////////////////////////
    */
    
    vector<float> policies(225, 0.0);
    policy_sum = 0.0;
    for(auto& pos: legal_actions){
        uint64_t legal_move = bitboard_pos_to_common_pos[pos];
        policies[legal_move] = gumbel_policy[legal_move];
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
        policies = gumbel_policy;
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

int main(){
    srand(time(NULL));
    map<int, double> gumbel_mp;
    vector<double> policy = {2, -1, 3, 4, 5, };
    int count = 100000;
    for(int i = 0; i < count; ++i){
        vector<int> top_k_indices = GumbelTopK(policy, policy.size(), 0);
        gumbel_mp[top_k_indices[0]]++;
    }
    
    cout << "gumbel max policy: " << endl;
    for(auto& [key, value] : gumbel_mp){
        cout << key << " : " << value / count << endl;
    }
    cout << endl;


    float policy_sum = 0.0;
    vector<float> soft_max_policy(policy.size(), 0.0);
    int MAX = *max_element(policy.begin(), policy.end());
    for(int i = 0; i < policy.size(); ++i){
        soft_max_policy[i] = exp(policy[i] - MAX);
        policy_sum += soft_max_policy[i];
    }
    for(int i = 0; i < policy.size(); ++i){
        soft_max_policy[i] = soft_max_policy[i] / policy_sum;
    }
        
    cout << "soft max policy: " << endl;
    int i = 0;
    for(auto& p : soft_max_policy){
        cout << i++ << " : " << p << endl;
    }

    return 0;
}




init(root)
int m = 16
vector<pair<g_l_q, Node*>> nodes = expand_node(m); //return 我之後要MCTS的node的list
for(; m >= 2; m /= 2){
    for(int i = 0; i < m; ++i){
        for(int j = 0; j < N(m); ++j){
            MCTS(nodes[i].second); //做MCTS
        }
    }
    sort(nodes.begin(), nodes.begin() + m, greater<>);
}

return nodes[0] //最前面的就是最好的