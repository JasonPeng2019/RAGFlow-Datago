#ifndef DATAGO_STATS_SEARCH_H
#define DATAGO_STATS_SEARCH_H

#include "../search/search.h"
#include "../program/selfplaymanager.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void datago_collect_search_states(Search* search, SearchNode* rootNode, 
                                   Board& board, Player pla, int moveNumber) ;

void datago_record_move(Loc moveLoc, Player pla, const Board& board);

#ifdef __cplusplus
}
#include <string>

// Forward declarations
struct FinishedGameData;

#define W1 0.00
#define W2 0.00
#define W3 0.00
#define UNCERTAINTY_THRESHOLD 0.9
#define RAG_OUTPUT_DIR "./rag_data"

struct ChildNodeInfo {
    std::string move;
    std::string child_sym_hash;
    double value;
    double pUCT;
    double prior;
    int visits;
};

struct PerMoveRAGData {
    int move_number;
    std::string sym_hash;
    std::string state_hash;
    std::string player_to_move;
    
    // Moves history UP TO THIS POINT (for reconstruction)
    std::vector<std::pair<std::string, std::string>> moves_history;
    
    std::vector<ChildNodeInfo> children;
    double policy_entropy;
    double value_score;
    double value_variance;
    double combined_score;
    int black_stones;
    int white_stones;
    
    // Neural network outputs for this position
    std::vector<float> policy;      // Full policy vector (361 values for 19x19)
    double winrate;                   // Winrate from NN
    std::vector<float> ownership;   // Ownership map (361 values for 19x19)
};

struct GameRAGData {
    std::string game_id;
    //float komi; -> can copy from finishedgamedata
    //std::string rules; -> ^
    //int board_size; -> ^
    std::vector<std::pair<std::string, std::string>> moves_history;  // Full move list

    // Only flagged complex positions (each with its own moves_history)
    std::vector<PerMoveRAGData> flagged_positions;
};



extern thread_local GameRAGData currentGameRAGData;

void writeCompleteRAGDataJSON(float komi, int board_size, const std::string& rules, const GameRAGData* ragData, const FinishedGameData* gameData);

GameRAGData* datago_get_current_game_data();

#endif
#endif  // DATAGO_STATS_SEARCH_H