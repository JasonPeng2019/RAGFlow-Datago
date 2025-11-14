#include "../stats_search.h"
#include "../../search/searchnode.h"
#include "../../dataio/trainingwrite.h"
#include <fstream>

int countStones(const Board& board) {
    int total = 0;
    for (int y = 0; y < board.y_size; y++) {
        for (int x = 0; x < board.x_size; x++) {
            Loc loc = Location::getLoc(x, y, board.x_size);
            Color color = board.colors[loc];
            if (color == C_BLACK || color == C_WHITE) {
                total++;
            }
        }
    }
    return total;
}

int countBlackStones(const Board& board) {
    int count = 0;
    for (int y = 0; y < board.y_size; y++) {
        for (int x = 0; x < board.x_size; x++) {
            Loc loc = Location::getLoc(x, y, board.x_size);
            if (board.colors[loc] == C_BLACK) {
                count++;
            }
        }
    }
    return count;
}

int countWhiteStones(const Board& board) {
    int count = 0;
    for (int y = 0; y < board.y_size; y++) {
        for (int x = 0; x < board.x_size; x++) {
            Loc loc = Location::getLoc(x, y, board.x_size);
            if (board.colors[loc] == C_WHITE) {
                count++;
            }
        }
    }
    return count;
}

double calculatePhaseWeight(int totalStones) {
    // Simple phase calculation - adjust as needed
    // Early game (0-80 stones): lower weight
    // Mid game (80-160): higher weight
    // Late game (160+): lower weight
    return 0;

    if (totalStones < 80) {
        return 0.5;
    } else if (totalStones < 160) {
        return 1.0;
    } else {
        return 0.7;
    }
}

double calculateValueVariance(SearchNode* rootNode) {
    ConstSearchNodeChildrenReference children = rootNode->getChildren();
    int numChildren = children.iterateAndCountChildren();
    
    // Collect all child values
    std::vector<double> childValues;
    
    for (int i = 0; i < numChildren; i++) {
        const SearchChildPointer& child = children[i];
        const SearchNode* childNode = child.getIfAllocated();
        if(childNode == nullptr) {continue;}
        if (childNode != nullptr) {
            NodeStats childStats(childNode->stats);
            childValues.push_back(childStats.winLossValueAvg);  // or utilityAvg
        }
    }
    
    // Calculate variance
    if (childValues.empty()) return 0.0;
    
    // Calculate mean
    double mean = 0.0;
    for (double val : childValues) {
        mean += val;
    }
    double childsize = childValues.size();
    mean /= childsize;
    
    // Calculate variance
    double variance = 0.0;
    for(size_t i = 0; i < childValues.size(); i++) {
        double val = childValues[i];
        double diff = val - mean;
        variance += diff * diff;
    }
    variance /= childValues.size();
    
    return variance;
}

bool if_uncertain(double combined) {
    //const double UNCERTAINTY_THRESHOLD = 0.7; // Example threshold
    //return combined > UNCERTAINTY_THRESHOLD;
    

    //if a random number between 1 and 5 = 2, then return true. 
    //make sure the rand number is different each time by seeding it with current time
    static bool seeded = false;
    if (!seeded) {
        srand(time(nullptr));
        seeded = true;
    }
    return (rand() % 5 + 1) == 2;
}


void datago_collect_search_states(Search* search, SearchNode* rootNode, 
                                   Board& board, Player pla, int moveNumber) {
    
    // 1. Calculate complexity metrics for THIS position
    double surprise, searchEntropy, E;
    search->getPolicySurpriseAndEntropy(surprise, searchEntropy, E); 
    double K = calculateValueVariance(rootNode);
    int totalStones = countStones(board);
    double phase = calculatePhaseWeight(totalStones);
    double combined = calculateCombinedUncertainty(E, K, phase);
    
    // 2. Only proceed if complex
    bool is_uncertain = if_uncertain(combined);
    if (is_uncertain) {
        
        PerMoveRAGData moveData;
        
        // 3. Populate basic fields
        moveData.move_number = moveNumber;

        // Compute symmetric hash (minimum hash across all 8 symmetries)
        Hash128 thisHash = board.getSitHashWithSimpleKo(pla);
        Hash128 symHash = thisHash;
        for(int symmetry = 1; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
            Board symBoard = SymmetryHelpers::getSymBoard(board, symmetry);
            Hash128 hash = symBoard.getSitHashWithSimpleKo(pla);
            if(hash < symHash)
                symHash = hash;
        }

        moveData.sym_hash = Global::uint64ToHexString(symHash.hash1) + Global::uint64ToHexString(symHash.hash0);
        moveData.state_hash = Global::uint64ToHexString(thisHash.hash1) + Global::uint64ToHexString(thisHash.hash0);
        moveData.player_to_move = (pla == P_BLACK) ? "B" : "W";
        
        // 4. Copy moves history from GameRAGData
        moveData.moves_history = currentGameRAGData.moves_history;
        
        // 5. Populate uncertainty metrics
        moveData.policy_entropy = E;
        moveData.value_variance = K;
        moveData.combined_score = combined;
        
        // 6. Get root stats for value_score
        NodeStats rootStats(rootNode->stats);
        moveData.value_score = rootStats.scoreMeanAvg;  // or utilityAvg depending on what you want
        
        // 7. Count stones
        moveData.black_stones = countBlackStones(board);
        moveData.white_stones = countWhiteStones(board);
        
        // 8. Extract ALL children
        SearchNodeChildrenReference children = rootNode->getChildren();
        int numChildren = children.iterateAndCountChildren();

        // Get policy probabilities from NN
        float policyProbs[NNPos::MAX_NN_POLICY_SIZE];
        bool hasPolicyData = search->getPolicy(rootNode, policyProbs);

        std::vector<ChildNodeInfo> childrenInfo;

        for (int i = 0; i < numChildren; i++) {
            const SearchChildPointer& child = children[i];  // Access via [] operator

             ChildNodeInfo info;

            // Populate move location
            Loc moveLoc = child.getMoveLoc();
            info.move = Location::toString(moveLoc, board);  // e.g., "Q10"

            // Populate prior (NN policy probability)
            if (hasPolicyData) {
                int pos = NNPos::locToPos(moveLoc, board.x_size, NNPos::MAX_BOARD_LEN);
                info.prior = policyProbs[pos];
            } else {
                info.prior = 0.0;
            }
            
            // Populate visits
            info.visits = child.getEdgeVisits();
            
            // Get child node if expanded
            const SearchNode* childNode = child.getIfAllocated();
            if (childNode != nullptr) {
                // Populate child_sym_hash using graphHash (graph search hash)
                info.child_sym_hash = Global::uint64ToHexString(childNode->graphHash.hash1) +
                                     Global::uint64ToHexString(childNode->graphHash.hash0);

                // Get child stats (thread-safe)
                NodeStats childStats(childNode->stats);

                // Populate value
                info.value = childStats.winLossValueAvg;  // or utilityAvg

                // pUCT not needed for RAG - used only during MCTS selection
                info.pUCT = 0.0;

            } else {
                // Child not expanded - set defaults
                info.child_sym_hash = "";
                info.value = 0.0;
                info.pUCT = 0.0;
            }
            
            childrenInfo.push_back(info);
        }
        
        moveData.children = childrenInfo;
        
        // 9. Store the completed PerMoveRAGData
        currentGameRAGData.flagged_positions.push_back(moveData);
    }
}

//Linear Function
double calculateCombinedUncertainty(double E, double K, double phase) 
 {
    return W1 * E + W2 * K + W3 * phase;
}

//Input-dependent temperature + energy (CTS+Energy) function
//implement later


void writeCompleteRAGDataJSON(float komi, int board_size, const std::string& rules, const FinishedGameData* gameData) {
    // Generate game_id from gameHash
    std::string game_id = "game_" + Global::uint64ToHexString(gameData->gameHash.hash1) +
                          Global::uint64ToHexString(gameData->gameHash.hash0);

    // Create filename in RAG_OUTPUT_DIR
    std::string filename = std::string(RAG_OUTPUT_DIR) + "/RAG_rawdata_" + game_id + ".json";

    // Open file for writing
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        throw StringError("Failed to open file: " + filename);
    }

    // Write JSON header
    outfile << "{\n";
    outfile << "  \"game_id\": \"" << game_id << "\",\n";
    outfile << "  \"settings\": {\n";
    outfile << "    \"komi\": " << komi << ",\n";
    outfile << "    \"rules\": \"" << rules << "\",\n";
    outfile << "    \"board_size\": " << board_size << ",\n";
    outfile << "    \"uncertainty_threshold\": " << UNCERTAINTY_THRESHOLD << ",\n";
    outfile << "    \"w1_policy_entropy\": " << W1 << ",\n";
    outfile << "    \"w2_value_variance\": " << W2 << "\n";
    outfile << "  },\n";

    // Write flagged positions array
    outfile << "  \"flagged_positions\": [\n";

    // Iterate through all flagged positions
    for (size_t i = 0; i < currentGameRAGData.flagged_positions.size(); i++) {
        const PerMoveRAGData& moveData = currentGameRAGData.flagged_positions[i];

        outfile << "    {\n";
        outfile << "      \"move_number\": " << moveData.move_number << ",\n";
        outfile << "      \"sym_hash\": \"" << moveData.sym_hash << "\",\n";
        outfile << "      \"state_hash\": \"" << moveData.state_hash << "\",\n";
        outfile << "      \"player_to_move\": \"" << moveData.player_to_move << "\",\n";

        // Write moves_history
        outfile << "      \"moves_history\": [\n";
        for (size_t j = 0; j < moveData.moves_history.size(); j++) {
            outfile << "        [\"" << moveData.moves_history[j].first << "\", \"" << moveData.moves_history[j].second << "\"]";
            if (j < moveData.moves_history.size() - 1) outfile << ",";
            outfile << "\n";
        }
        outfile << "      ],\n";

        // Write stone_count
        outfile << "      \"stone_count\": {\n";
        outfile << "        \"black\": " << moveData.black_stones << ",\n";
        outfile << "        \"white\": " << moveData.white_stones << ",\n";
        outfile << "        \"total\": " << (moveData.black_stones + moveData.white_stones) << "\n";
        outfile << "      },\n";

        // Write uncertainty_metrics
        outfile << "      \"uncertainty_metrics\": {\n";
        outfile << "        \"policy_entropy\": " << moveData.policy_entropy << ",\n";
        outfile << "        \"value_variance\": " << moveData.value_variance << ",\n";
        outfile << "        \"combined_score\": " << moveData.combined_score << "\n";
        outfile << "      },\n";

        // Write children array
        outfile << "      \"children\": [\n";
        for (size_t k = 0; k < moveData.children.size(); k++) {
            const ChildNodeInfo& child = moveData.children[k];
            outfile << "        {\n";
            outfile << "          \"move\": \"" << child.move << "\",\n";
            outfile << "          \"child_sym_hash\": \"" << child.child_sym_hash << "\",\n";
            outfile << "          \"value\": " << child.value << ",\n";
            outfile << "          \"prior\": " << child.prior << ",\n";
            outfile << "          \"visits\": " << child.visits << "\n";
            outfile << "        }";
            if (k < moveData.children.size() - 1) outfile << ",";
            outfile << "\n";
        }
        outfile << "      ]\n";

        outfile << "    }";
        if (i < currentGameRAGData.flagged_positions.size() - 1) outfile << ",";
        outfile << "\n";
    }

    outfile << "  ],\n";

    // Write summary
    int total_flagged = currentGameRAGData.flagged_positions.size();
    int total_moves = currentGameRAGData.moves_history.size();
    double flagging_rate = (total_moves > 0) ? ((double)total_flagged / total_moves) : 0.0;

    outfile << "  \"summary\": {\n";
    outfile << "    \"total_moves\": " << total_moves << ",\n";
    outfile << "    \"flagged_count\": " << total_flagged << ",\n";
    outfile << "    \"flagging_rate\": " << flagging_rate << "\n";
    outfile << "  }\n";
    outfile << "}\n";

    outfile.close();

    // Clear the thread-local data after writing
    currentGameRAGData.flagged_positions.clear();
    currentGameRAGData.moves_history.clear();
}



