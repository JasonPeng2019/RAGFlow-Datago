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

#ifdef __cplusplus
}
#include <string>

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



thread_local GameRAGData currentGameRAGData;



void writeCompleteRAGDataJSON(float komi, int board_size, const std::string& rules, const FinishedGameData* gameData);
#endif
#endif  // DATAGO_STATS_SEARCH_H