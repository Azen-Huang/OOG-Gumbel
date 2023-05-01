#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
#include "bitboard.h"
#include "state.h"
#include "Tboard.h"
#include "Cboard.h"
using namespace std;
//g++ -march=native alpha_beta.cc -g -O3 -o alpha_beta && "/home/azon/Documents/OOG/OOG-Alphazero/"alpha_beta | tee output.log
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
BitBoard endmy, endopp, endRzone;
int TSS(BitBoard my, BitBoard opp, BitBoard Tzone, int depth)
{
    if(depth <= 0)return 0;
    BitBoard cTzone = Tzone & my & opp;
    int check = 0;
    while(cTzone && !check)
    {
        uint64_t pos = cTzone.ls1b();
        uint64_t feat = feature(my, opp, pos);
        uint64_t s_h = state[_pext_u64(feat, pext_h)];
        uint64_t s_b = state[_pext_u64(feat, pext_b)];
        uint64_t s_v = state[_pext_u64(feat, pext_v)];
        uint64_t s_s = state[_pext_u64(feat, pext_s)];
        my.append(pos);
        if((s_h | s_b | s_v | s_s) & 0xC0000) //terminal condition
        {
            if(s_h & 0x40000)
            {
                opp = opp & Cboard_h[(pos << 5) | (s_h & 0x1F)];
            }
            else if (s_b & 0x40000)
            {
                opp = opp & Cboard_b[(pos << 5) | (s_b & 0x1F)];
            }
            else if (s_v & 0x40000)
            {
                opp = opp & Cboard_v[(pos << 5) | (s_v & 0x1F)];
            }
            else if (s_s & 0x40000)
            {
                opp = opp & Cboard_s[(pos << 5) | (s_s & 0x1F)];
            }
            endmy = my;
            endopp = opp;
            return depth;
        }
        if(s_h & 0x30000)
        {
            check = TSS(my,
                        opp & Cboard_h[(pos << 5) | (s_h & 0x1F)],
                        Tzone
                        |Tboard_h[(s_h & 0x3F00) | pos]
                        |Tboard_b[(s_b & 0x3F00) | pos]
                        |Tboard_v[(s_v & 0x3F00) | pos]
                        |Tboard_s[(s_s & 0x3F00) | pos],
                        depth-1);
        }
        else if(s_b & 0x30000)
        {
            check = TSS(my,
                        opp & Cboard_b[(pos << 5) | (s_b & 0x1F)],
                        Tzone
                        |Tboard_h[(s_h & 0x3F00) | pos]
                        |Tboard_b[(s_b & 0x3F00) | pos]
                        |Tboard_v[(s_v & 0x3F00) | pos]
                        |Tboard_s[(s_s & 0x3F00) | pos],
                        depth-1);
        }
        else if(s_v & 0x30000)
        {
            check = TSS(my,
                        opp & Cboard_v[(pos << 5) | (s_v & 0x1F)],
                        Tzone
                        |Tboard_h[(s_h & 0x3F00) | pos]
                        |Tboard_b[(s_b & 0x3F00) | pos]
                        |Tboard_v[(s_v & 0x3F00) | pos]
                        |Tboard_s[(s_s & 0x3F00) | pos],
                        depth-1);
        }
        else if(s_s & 0x30000)
        {
            check = TSS(my,
                        opp & Cboard_s[(pos << 5) | (s_s & 0x1F)],
                        Tzone
                        |Tboard_h[(s_h & 0x3F00) | pos]
                        |Tboard_b[(s_b & 0x3F00) | pos]
                        |Tboard_v[(s_v & 0x3F00) | pos]
                        |Tboard_s[(s_s & 0x3F00) | pos],
                        depth-1);
        }
        else
        {
            Tzone.append(pos);
        }
        my.append(pos);
    }
    return check;
}
bool canThreat(uint64_t feat)
{
    uint64_t s_h = state[_pext_u64(feat, pext_h)];
    uint64_t s_b = state[_pext_u64(feat, pext_b)];
    uint64_t s_v = state[_pext_u64(feat, pext_v)];
    uint64_t s_s = state[_pext_u64(feat, pext_s)];
    if((s_h | s_b | s_v | s_s) & 0xE0000) //L5, L4, D4
    {
        return true;
    }
    return false;
}
void NM(BitBoard my, BitBoard opp, BitBoard Tzone, int depth)
{
    int check = 0;
    for(int i = 1; i <= depth/2 && !check; i++)
    {
        check = TSS(my, opp, Tzone, i);
    }
    endRzone.initZero();
    if(check)
    {
        BitBoard empty = endmy & endopp;
        while(empty)
        {
            uint64_t pos = empty.ls1b();
            uint64_t feat = feature(endopp, endmy, pos);
            if(canThreat(feat))
            {
                endRzone.append(pos);
            }
        }
        endRzone = endRzone | (my^endmy) | (opp^endopp);
    }
}
int ABSearch(BitBoard my, BitBoard opp, BitBoard myTzone, BitBoard oppTzone, BitBoard moves,BitBoard Rzone, int alpha, int beta, int depth, int score)
{
    if(depth <= 0)return score;
    BitBoard mov = Rzone ? Rzone : (moves & my & opp);
    while(mov)
    {
        uint64_t pos = mov.ls1b();
        uint64_t feat = feature(my, opp, pos);
        uint64_t s_h = state[_pext_u64(feat, pext_h)];
        uint64_t s_b = state[_pext_u64(feat, pext_b)];
        uint64_t s_v = state[_pext_u64(feat, pext_v)];
        uint64_t s_s = state[_pext_u64(feat, pext_s)];
        BitBoard newTzone = myTzone
                            |Tboard_h[(s_h & 0x3F00) | pos]
                            |Tboard_b[(s_b & 0x3F00) | pos]
                            |Tboard_v[(s_v & 0x3F00) | pos]
                            |Tboard_s[(s_s & 0x3F00) | pos];
        if((s_h | s_b | s_v | s_s) & 0xC0000) //L5, L4
        {
            return 10000;
        }
        my.append(pos);
        NM(my, opp, newTzone, depth);
        int value = -ABSearch(opp, my, oppTzone, newTzone, moves.mind(pos), endRzone, -beta, -alpha, depth-1, -score);
        if(value >= 10000)return value;
        if(value >= beta)return value;
        if(value > alpha)alpha = value;
        my.append(pos);
    }
    return alpha;
}
uint64_t ABCaller(BitBoard my, BitBoard opp, BitBoard myTzone, BitBoard oppTzone, BitBoard moves, int alpha, int beta, int depth, int score)
{
    if(depth <= 0)return score;
    BitBoard mov = moves & my & opp;
    uint64_t bestmove = 0;
    while(mov)
    {
        uint64_t pos = mov.ls1b();
        uint64_t feat = feature(my, opp, pos);
        uint64_t s_h = state[_pext_u64(feat, pext_h)];
        uint64_t s_b = state[_pext_u64(feat, pext_b)];
        uint64_t s_v = state[_pext_u64(feat, pext_v)];
        uint64_t s_s = state[_pext_u64(feat, pext_s)];
        BitBoard newTzone = myTzone
                            |Tboard_h[(s_h & 0x3F00) | pos]
                            |Tboard_b[(s_b & 0x3F00) | pos]
                            |Tboard_v[(s_v & 0x3F00) | pos]
                            |Tboard_s[(s_s & 0x3F00) | pos];
        if((s_h | s_b | s_v | s_s) & 0x80000) //L5
        {
            return pos;
        }
        my.append(pos);
        NM(my, opp, newTzone, depth);
        int value = -ABSearch(opp, my, oppTzone, newTzone, moves.mind(pos), endRzone, -beta, -alpha, depth-1, -score);
        if(value > alpha)
        {
            alpha = value;
            bestmove = pos;
        }
        my.append(pos);
    }
    return bestmove;
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
    return (15-n)*16 + c - 'A' + 1;
}
struct Game
{
    BitBoard black, white, moves, blackTzone, whiteTzone;
    BitBoard black_record[256], white_record[256], moves_record[256], blackTzone_record[256], whiteTzone_record[256];
    int size = 0, score = 0, score_record[256];
    void Start();
    void Move(uint64_t pos);
    void Undo();
    void Play(int depth);
    void Print_Board();
};
void Game::Start()
{
    black.init();
    white.init();
    moves.initZero();
    blackTzone.initZero();
    whiteTzone.initZero();
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
void Game::Play(int depth)
{
    uint64_t pos = 0;
    if(size%2 == 0)
    {
        if (size == 0)
        {
            moves.append(114);
        }
        for(int i = 1; i <= depth; i++)
        {
            pos = ABCaller(black, white, blackTzone, whiteTzone, moves, -30000, 30000, i, score);
        }
    }
    else
    {
        pos = 0;
        if(size == 1)
        {
            BitBoard tmp;
            tmp.append(114);
            if(!(black ^ tmp))
            {
                pos = 117;
            }
            tmp.append(114);
            tmp.append(24);
            if(!(black ^ tmp))
            {
                pos = 72;
            }
            tmp.append(24);
            tmp.append(126);
            if(!(black ^ tmp))
            {
                pos = 123;
            }
            tmp.append(126);
            tmp.append(216);
            if(!(black ^ tmp))
            {
                pos = 168;
            }
            tmp.append(216);
        }
        if(pos == 0)
        {
            for(int i = 1; i <= depth; i++)
            {
                pos = ABCaller(white, black, whiteTzone, blackTzone, moves, -30000, 30000, i, score);
            }
        }
    }
    Move(pos);
    Output(pos);
    Print_Board();
}
void Game::Print_Board(){
    vector<int> Black(225,1);
    vector<int> White(225,1);
    BitBoard copy_black = black;
    BitBoard copy_white = white;
    while(copy_black){
        uint64_t pos = copy_black.ls1b();
        Black[bitboard_pos_to_common_pos[pos]] = 0;
    }
    while(copy_white){
        uint64_t pos = copy_white.ls1b();
        White[bitboard_pos_to_common_pos[pos]] = 0;
    }

    cout << "   ";
    for(char i = 'A'; i < 'O'; ++i){
        cout << i << " ";
    }
    cout << endl;
    cout << "   ";
    for(int i = 'A'; i < 'O'; ++i){
        cout << "--";
    }
    cout << endl;
    cout << 15 << "|";
    for(int i = 1; i <= 225; ++i){
        if(Black[i - 1] == 1){
            cout << "o" << " ";
        }
        else if(White[i - 1] == 1){
            cout << "x" << " ";
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
}
int main()
{
    init();
    Game game;
    int depth;
    string op;
    while(cin >> op)
    {
        if(op == "TERMINATE")
        {
            break;
        }
        else if(op == "START")
        {
            game.Start();
            cout << "ROGER" << endl;
        }
        else if(op == "PLAY")
        {
            cin >> depth;
            game.Play(depth);
        }
        else if(op == "UNDO")
        {
            game.Undo();
            cout << "ROGER" << endl;
        }
        else if('A' <= (op[0]&0xDF) && (op[0]&0xDF) <= 'O')
        {
            stringstream ss;
            op[0] &= 0xDF;
            ss << op;
            char c;
            int n;
            if(ss >> c >> n)
            {
                uint64_t pos = Input(c, n);
                game.Move(pos);
                cout << "ROGER" << endl;
            }
            else
            {
                cout << "WRONG" << endl;
            }
        }
        else
        {
            cout << "WRONG" << endl;
        }
    }
    return 0;
}
