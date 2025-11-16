#ifndef DATAGO_STATS_SEARCH_H
#define DATAGO_STATS_SEARCH_H

#include "../search/search.h"
#include "../program/selfplaymanager.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void datago_collect_search_states(Search* search, SearchNode* rootNode);

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

struct DeepSearchResult {
    bool available = false;
    std::string status;
    double elapsed = 0.0;
    int64_t visits = 0;
    double winrate = 0.0;
    double score_lead = 0.0;
    std::string sym_hash;
    std::string state_hash;
    std::string player_to_move;
    int symmetry_index = 0;
    int move_number = 0;
    double komi = 0.0;
    int black_stones = 0;
    int white_stones = 0;
    std::string best_child_move;
    std::string best_child_hash;
    int best_child_symmetry_index = 0;
    std::vector<float> policy;
    std::vector<float> ownership;
    std::vector<ChildNodeInfo> children;
};

struct PerMoveRAGData {
    int move_number;
    std::string sym_hash;
    std::string state_hash;
    int symmetry_index;             // 0-7: which symmetry maps sym_hash to state_hash
    std::string player_to_move;

    // Moves history UP TO THIS POINT (for reconstruction)
    std::vector<std::pair<std::string, std::string>> moves_history;

    // Best child (determined by highest visit count)
    std::string best_child_move;       // e.g., "Q16"
    std::string best_child_hash;       // child's sym_hash
    int best_child_symmetry_index;     // child's symmetry index (0-7)

    std::vector<ChildNodeInfo> children;
    double policy_entropy;
    double value_score;
    double value_variance;
    double combined_score;
    int black_stones;
    int white_stones;
    std::vector<int> board_state;       // Flattened board snapshot (1 black, -1 white, 0 empty)
    std::vector<int> superko_bans;      // Flattened ko-ban mask (0/1)
    int board_x_size = 0;
    int board_y_size = 0;
    int encore_phase = 0;
    int num_turns_this_phase = 0;
    int num_consec_valid_turns_this_game = 0;
    int consecutive_ending_passes = 0;
    bool assume_multiple_handicap = false;
    bool is_past_normal_phase_end = false;
    bool is_game_finished = false;
    std::string presumed_next_move;
    std::string search_params_json;

    // Neural network outputs for this position
    std::vector<float> policy;      // Full policy vector (361 values for 19x19)
    double winrate;                   // Winrate from NN
    std::vector<float> ownership;   // Ownership map (361 values for 19x19)

    // Inline deep search outputs
    bool has_deep_analysis = false;
    double deep_search_winrate = 0.0;
    double deep_search_score_lead = 0.0;
    int64_t deep_search_visits = 0;
    double deep_search_elapsed = 0.0;
    std::string deep_search_status;
    DeepSearchResult deep_result;
};

struct GameRAGData {
    std::string game_id;
    //float komi; -> can copy from finishedgamedata
    //std::string rules; -> ^
    //int board_size; -> ^
    std::vector<std::pair<std::string, std::string>> moves_history;  // Full move list

    // Only flagged complex positions (each with its own moves_history)
    std::vector<PerMoveRAGData> flagged_positions;

    int inline_deep_searches_run = 0;
};



extern thread_local GameRAGData currentGameRAGData;

void writeCompleteRAGDataJSON(float komi, int board_size, const std::string& rules, const GameRAGData* ragData, const FinishedGameData* gameData);

GameRAGData* datago_get_current_game_data();

#endif
#endif  // DATAGO_STATS_SEARCH_H
