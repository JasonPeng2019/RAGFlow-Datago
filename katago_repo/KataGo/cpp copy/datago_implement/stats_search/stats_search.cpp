#include "../stats_search.h"
#include "../../search/searchnode.h"
#include "../../dataio/trainingwrite.h"
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <cctype>

// Thread-local storage for current game's RAG data
thread_local GameRAGData currentGameRAGData;

#include "../../core/timer.h"

namespace {

struct InlineDeepSearchConfig {
    bool enabled = true;
    int samplingStride = 5;
    int visits = 4800;
    int maxPerMove = 1;
    int maxPerGame = 256;
    bool logTimings = false;
};

static bool parseBoolEnv(const char* name, bool defaultVal) {
    const char* env = std::getenv(name);
    if(env == nullptr)
        return defaultVal;
    std::string value(env);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    if(value == "1" || value == "true" || value == "on" || value == "yes")
        return true;
    if(value == "0" || value == "false" || value == "off" || value == "no")
        return false;
    return defaultVal;
}

static int parseIntEnv(const char* name, int defaultVal) {
    const char* env = std::getenv(name);
    if(env == nullptr)
        return defaultVal;
    char* end = nullptr;
    long parsed = std::strtol(env, &end, 10);
    if(end == env)
        return defaultVal;
    return (int)parsed;
}

static InlineDeepSearchConfig loadInlineDeepSearchConfig() {
    InlineDeepSearchConfig cfg;
    cfg.enabled = parseBoolEnv("DATAGO_INLINE_DEEP_ENABLED", true);
    cfg.samplingStride = std::max(1, parseIntEnv("DATAGO_INLINE_DEEP_STRIDE", 5));
    cfg.visits = std::max(0, parseIntEnv("DATAGO_INLINE_DEEP_VISITS", 4800));
    cfg.maxPerMove = std::max(0, parseIntEnv("DATAGO_INLINE_DEEP_MAX_PER_MOVE", 5));
    cfg.maxPerGame = std::max(0, parseIntEnv("DATAGO_INLINE_DEEP_MAX_PER_GAME", 256));
    cfg.logTimings = parseBoolEnv("DATAGO_INLINE_DEEP_LOG", false);
    return cfg;
}

static const InlineDeepSearchConfig& getInlineDeepSearchConfig() {
  static InlineDeepSearchConfig cfg = loadInlineDeepSearchConfig();
  return cfg;
}

static thread_local int inlineDeepSearchMoveNumber = -1;
static thread_local int inlineDeepSearchesThisMove = 0;

static std::string computeSymHashForMove(const Board& board, Player pla, Loc moveLoc, int& symmetryIdxOut) {
  Board childBoard = board;
  Player childPla = getOpp(pla);
  childBoard.playMove(moveLoc, childPla, true);
  Hash128 childHash = childBoard.getSitHashWithSimpleKo(childPla);
  Hash128 childSymHash = childHash;
  symmetryIdxOut = 0;
  for(int symmetry = 1; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
    Board symBoard = SymmetryHelpers::getSymBoard(childBoard, symmetry);
    Hash128 hash = symBoard.getSitHashWithSimpleKo(childPla);
    if(hash < childSymHash) {
      childSymHash = hash;
      symmetryIdxOut = symmetry;
    }
  }
  return Global::uint64ToHexString(childSymHash.hash1) + Global::uint64ToHexString(childSymHash.hash0);
}

}  // namespace

static bool shouldRunInlineDeepSearchForMove(int moveNumber) {
    const InlineDeepSearchConfig& cfg = getInlineDeepSearchConfig();
    if(!cfg.enabled)
        return false;
    if(cfg.visits <= 0)
        return false;
    if(cfg.maxPerGame > 0 && currentGameRAGData.inline_deep_searches_run >= cfg.maxPerGame)
        return false;
    if(inlineDeepSearchMoveNumber != moveNumber) {
        inlineDeepSearchMoveNumber = moveNumber;
        inlineDeepSearchesThisMove = 0;
    }
    if(cfg.maxPerMove > 0 && inlineDeepSearchesThisMove >= cfg.maxPerMove)
        return false;
    return true;
}

static bool runInlineDeepSearch(Search* baseSearch, const Board& board, const BoardHistory& history, Player pla, int moveNumber, PerMoveRAGData& moveData) {
    const InlineDeepSearchConfig& cfg = getInlineDeepSearchConfig();
    if(!shouldRunInlineDeepSearchForMove(moveNumber))
        return false;
    inlineDeepSearchesThisMove++;
    currentGameRAGData.inline_deep_searches_run++;

    SearchParams deepParams = baseSearch->searchParams;
    deepParams.maxVisits = cfg.visits;
    deepParams.maxPlayouts = cfg.visits;
    deepParams.maxVisitsPondering = cfg.visits;
    deepParams.maxPlayoutsPondering = cfg.visits;
    deepParams.maxTime = 1e30;
    deepParams.maxTimePondering = 1e30;
    deepParams.useEvalCache = false;
    deepParams.useGraphSearch = false;

    std::string deepSeed = baseSearch->randSeed + std::string("$inlineDeep$") + Global::intToString(currentGameRAGData.inline_deep_searches_run);
    Search deepSearch(deepParams, baseSearch->nnEvaluator, baseSearch->logger, deepSeed);

    Board deepBoard = board;
    BoardHistory deepHistory = history;

    Logger* logger = baseSearch->logger;
    if(cfg.logTimings && logger != NULL) {
        logger->write(Global::strprintf("[DATAGO] Inline deep search begin move %d (visits=%d, count=%d/%d)",
                                        moveNumber, cfg.visits, inlineDeepSearchesThisMove, cfg.maxPerMove));
    }

    ClockTimer timer;
    bool success = false;
    std::string status = "failed";
    try {
        deepSearch.setPosition(pla, deepBoard, deepHistory);
        deepSearch.runWholeSearch(pla,false);
        SearchNode* deepRoot = deepSearch.rootNode;
        if(deepRoot != nullptr) {
            NodeStats deepStats(deepRoot->stats);
            moveData.has_deep_analysis = true;
            moveData.deep_search_visits = deepStats.visits;
            moveData.deep_search_winrate = deepStats.winLossValueAvg;
            moveData.deep_search_score_lead = deepStats.scoreMeanAvg;
            DeepSearchResult& deepRes = moveData.deep_result;
            deepRes.available = true;
            deepRes.visits = deepStats.visits;
            deepRes.winrate = deepStats.winLossValueAvg;
            deepRes.score_lead = deepStats.scoreMeanAvg;
            deepRes.status = "ok";

            const NNOutput* nnOutput = deepRoot->getNNOutput();
            if(nnOutput != nullptr) {
                int policySize = NNPos::getPolicySize(board.x_size, board.y_size);
                deepRes.policy.resize(policySize);
                for(int i = 0; i < policySize; i++)
                    deepRes.policy[i] = nnOutput->policyProbs[i];
                if(nnOutput->whiteOwnerMap != nullptr) {
                    int ownershipSize = board.x_size * board.y_size;
                    deepRes.ownership.resize(ownershipSize);
                    for(int i = 0; i < ownershipSize; i++)
                        deepRes.ownership[i] = nnOutput->whiteOwnerMap[i];
                }
                else {
                    deepRes.ownership.clear();
                }
            }
            else {
                deepRes.policy.clear();
                deepRes.ownership.clear();
            }

            // Collect children info
            SearchNodeChildrenReference children = deepRoot->getChildren();
            int numChildren = children.iterateAndCountChildren();
            float policyProbs[NNPos::MAX_NN_POLICY_SIZE];
            bool hasPolicyData = deepSearch.getPolicy(deepRoot, policyProbs);
            std::vector<ChildNodeInfo> childInfos;
            childInfos.reserve(numChildren);
            for(int i = 0; i < numChildren; i++) {
                const SearchChildPointer& child = children[i];
                ChildNodeInfo info;
                Loc moveLoc = child.getMoveLoc();
                info.move = Location::toString(moveLoc, board);
                if(hasPolicyData) {
                    int pos = NNPos::locToPos(moveLoc, board.x_size, NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN);
                    info.prior = policyProbs[pos];
                }
                else {
                    info.prior = 0.0;
                }
                info.visits = child.getEdgeVisits();
                const SearchNode* childNode = child.getIfAllocated();
                bool graphHashValid = false;
                if(childNode != nullptr) {
                    if(childNode->graphHash.hash0 != 0 || childNode->graphHash.hash1 != 0) {
                        info.child_sym_hash = Global::uint64ToHexString(childNode->graphHash.hash1) +
                                              Global::uint64ToHexString(childNode->graphHash.hash0);
                        graphHashValid = true;
                    }
                    NodeStats childStats(childNode->stats);
                    info.value = childStats.winLossValueAvg;
                    info.pUCT = 0.0;
                }
                else {
                    info.value = 0.0;
                    info.pUCT = 0.0;
                }
                if(!graphHashValid) {
                    int dummySymIdx = 0;
                    info.child_sym_hash = computeSymHashForMove(board, pla, moveLoc, dummySymIdx);
                }
                childInfos.push_back(info);
            }
            deepRes.children = childInfos;

            int maxVisits = -1;
            std::string bestMove;
            std::string bestHash;
            for(const auto& child : childInfos) {
                if(child.visits > maxVisits) {
                    maxVisits = child.visits;
                    bestMove = child.move;
                    bestHash = child.child_sym_hash;
                }
            }
            deepRes.best_child_move = bestMove;
            deepRes.best_child_hash = bestHash;
            int bestSymIdx = 0;
            if(!bestMove.empty()) {
                Loc bestMoveLoc = Location::ofString(bestMove, board);
                bestHash = computeSymHashForMove(board, pla, bestMoveLoc, bestSymIdx);
            }
            deepRes.best_child_symmetry_index = bestSymIdx;
            deepRes.best_child_hash = bestHash;

            status = "ok";
            success = true;
        }
        else {
            status = "no_root";
        }
    }
    catch(const std::exception& e) {
        status = std::string("error: ") + e.what();
        if(logger != NULL) {
            logger->write(Global::strprintf("[DATAGO] Inline deep search error at move %d: %s", moveNumber, e.what()));
        }
    }
    catch(...) {
        status = "unknown_error";
        if(logger != NULL) {
            logger->write(Global::strprintf("[DATAGO] Inline deep search error at move %d: unknown exception", moveNumber));
        }
    }

    double elapsed = timer.getSeconds();
    moveData.deep_search_elapsed = elapsed;
    moveData.deep_search_status = status;
    moveData.deep_result.elapsed = elapsed;
    moveData.deep_result.status = status;
    moveData.deep_result.visits = moveData.deep_search_visits;
    moveData.deep_result.available = success;

    if(cfg.logTimings && logger != NULL) {
        logger->write(Global::strprintf("[DATAGO] Inline deep search end move %d status=%s elapsed=%.3fs visits=%lld",
                                        moveNumber,
                                        status.c_str(),
                                        moveData.deep_search_elapsed,
                                        (long long)moveData.deep_search_visits));
    }

    return success;
}

// Forward declarations
double calculateCombinedUncertainty(double E, double K, double phase);

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
    (void)combined;
    const InlineDeepSearchConfig& cfg = getInlineDeepSearchConfig();
    if(!cfg.enabled)
        return false;
    static thread_local int counter = 0;
    counter++;
    if(cfg.samplingStride <= 1)
        return true;
    return (counter % cfg.samplingStride) == 0;
}


// DEPRECATED: No longer needed - we now extract moves directly from rootHistory.moveHistory
// void datago_record_move(Loc moveLoc, Player pla, const Board& board) {
//     std::string moveStr = Location::toString(moveLoc, board);
//     std::string plaStr = (pla == P_BLACK) ? "B" : "W";
//     currentGameRAGData.moves_history.push_back(std::make_pair(plaStr, moveStr));
// }

GameRAGData* datago_get_current_game_data() {
    return new GameRAGData(currentGameRAGData);
}


void datago_collect_search_states(Search* search, SearchNode* rootNode) {
    // Get all data from the search object for consistency
    const Board& board = search->getRootBoard();
    const BoardHistory& rootHistory = search->getRootHist();
    Player pla = search->getRootPla();
    int moveNumber = (int)rootHistory.moveHistory.size();

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
        int symmetryIndex = 0;  // Track which symmetry gives the minimum hash
        for(int symmetry = 1; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
            Board symBoard = SymmetryHelpers::getSymBoard(board, symmetry);
            Hash128 hash = symBoard.getSitHashWithSimpleKo(pla);
            if(hash < symHash) {
                symHash = hash;
                symmetryIndex = symmetry;
            }
        }

        moveData.sym_hash = Global::uint64ToHexString(symHash.hash1) + Global::uint64ToHexString(symHash.hash0);
        moveData.state_hash = Global::uint64ToHexString(thisHash.hash1) + Global::uint64ToHexString(thisHash.hash0);
        moveData.symmetry_index = symmetryIndex;
        moveData.player_to_move = (pla == P_BLACK) ? "B" : "W";

        // 4. Extract moves history from rootHistory.moveHistory (KataGo's BoardHistory)
        moveData.moves_history.clear();
        for(size_t i = 0; i < rootHistory.moveHistory.size(); i++) {
            const Move& move = rootHistory.moveHistory[i];
            std::string plaStr = (move.pla == P_BLACK) ? "B" : "W";
            std::string moveStr = Location::toString(move.loc, board);
            moveData.moves_history.push_back(std::make_pair(plaStr, moveStr));
        }
        
        // 5. Populate uncertainty metrics
        moveData.policy_entropy = E;
        moveData.value_variance = K;
        moveData.combined_score = combined;
        
        // 6. Get root stats for value_score and winrate
        NodeStats rootStats(rootNode->stats);
        moveData.value_score = rootStats.scoreMeanAvg;  // or utilityAvg depending on what you want
        moveData.winrate = rootStats.winLossValueAvg;   // Winrate
        
        // 7. Count stones
        moveData.black_stones = countBlackStones(board);
        moveData.white_stones = countWhiteStones(board);
        moveData.board_x_size = board.x_size;
        moveData.board_y_size = board.y_size;

        int boardArea = board.x_size * board.y_size;
        moveData.board_state.assign(boardArea, 0);
        moveData.superko_bans.assign(boardArea, 0);
        for(int y = 0; y < board.y_size; y++) {
            for(int x = 0; x < board.x_size; x++) {
                Loc loc = Location::getLoc(x, y, board.x_size);
                int idx = y * board.x_size + x;
                Color c = board.colors[loc];
                if(c == C_BLACK)
                    moveData.board_state[idx] = 1;
                else if(c == C_WHITE)
                    moveData.board_state[idx] = -1;
                else
                    moveData.board_state[idx] = 0;

                if(rootHistory.superKoBanned[loc])
                    moveData.superko_bans[idx] = 1;
                else
                    moveData.superko_bans[idx] = 0;
            }
        }

        moveData.encore_phase = rootHistory.encorePhase;
        moveData.num_turns_this_phase = rootHistory.numTurnsThisPhase;
        moveData.num_consec_valid_turns_this_game = rootHistory.numConsecValidTurnsThisGame;
        moveData.consecutive_ending_passes = rootHistory.consecutiveEndingPasses;
        moveData.assume_multiple_handicap = rootHistory.assumeMultipleStartingBlackMovesAreHandicap;
        moveData.is_past_normal_phase_end = rootHistory.isPastNormalPhaseEnd;
        moveData.is_game_finished = rootHistory.isGameFinished;
        moveData.presumed_next_move = (rootHistory.presumedNextMovePla == P_BLACK ? "B" :
                                       rootHistory.presumedNextMovePla == P_WHITE ? "W" : "_");
        try {
            nlohmann::json paramJson;
            paramJson["winLossUtilityFactor"] = search->searchParams.winLossUtilityFactor;
            paramJson["staticScoreUtilityFactor"] = search->searchParams.staticScoreUtilityFactor;
            paramJson["dynamicScoreUtilityFactor"] = search->searchParams.dynamicScoreUtilityFactor;
            paramJson["drawEquivalentWinsForWhite"] = search->searchParams.drawEquivalentWinsForWhite;
            paramJson["cpuctExploration"] = search->searchParams.cpuctExploration;
            paramJson["cpuctExplorationLog"] = search->searchParams.cpuctExplorationLog;
            paramJson["rootPolicyTemperature"] = search->searchParams.rootPolicyTemperature;
            paramJson["rootPolicyTemperatureEarly"] = search->searchParams.rootPolicyTemperatureEarly;
            paramJson["chosenMoveTemperature"] = search->searchParams.chosenMoveTemperature;
            paramJson["chosenMoveTemperatureEarly"] = search->searchParams.chosenMoveTemperatureEarly;
            paramJson["maxVisits"] = search->searchParams.maxVisits;
            paramJson["maxPlayouts"] = search->searchParams.maxPlayouts;
            paramJson["maxTime"] = search->searchParams.maxTime;
            moveData.search_params_json = paramJson.dump();
        } catch(...) {
            moveData.search_params_json = "{}";
        }
        
        // 8. Extract policy vector and ownership from NN output
        NNOutput* nnOutput = rootNode->getNNOutput();
        if (nnOutput != nullptr) {
            // Get policy vector
            int policySize = NNPos::getPolicySize(board.x_size, board.y_size);
            moveData.policy.resize(policySize);
            for (int i = 0; i < policySize; i++) {
                moveData.policy[i] = nnOutput->policyProbs[i];
            }
            
            // Get ownership if available
            if (nnOutput->whiteOwnerMap != nullptr) {
                int ownershipSize = board.x_size * board.y_size;
                moveData.ownership.resize(ownershipSize);
                for (int i = 0; i < ownershipSize; i++) {
                    moveData.ownership[i] = nnOutput->whiteOwnerMap[i];
                }
            }
        }
        
        // 9. Extract ALL children
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
                int pos = NNPos::locToPos(moveLoc, board.x_size, NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN);
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

        // 10. Find best child (highest visit count)
        int maxVisits = -1;
        std::string bestMove = "";
        std::string bestHash = "";
        for (const auto& child : childrenInfo) {
            if (child.visits > maxVisits) {
                maxVisits = child.visits;
                bestMove = child.move;
                bestHash = child.child_sym_hash;
            }
        }
        moveData.best_child_move = bestMove;
        moveData.best_child_hash = bestHash;

        // 10b. Calculate best child's symmetry index
        int bestChildSymmetryIndex = 0;
        if (!bestMove.empty()) {
            // Parse move string to Loc
            Loc bestMoveLoc = Location::ofString(bestMove, board);

            // Create child board by copying parent and playing the move
            Board childBoard = board;
            Player childPla = getOpp(pla);
            childBoard.playMove(bestMoveLoc, childPla, true);  // true = preventSuicide

            // Calculate child's symmetric hash
            Hash128 childHash = childBoard.getSitHashWithSimpleKo(childPla);
            Hash128 childSymHash = childHash;
            bestChildSymmetryIndex = 0;

            for(int symmetry = 1; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
                Board symBoard = SymmetryHelpers::getSymBoard(childBoard, symmetry);
                Hash128 hash = symBoard.getSitHashWithSimpleKo(childPla);
                if(hash < childSymHash) {
                    childSymHash = hash;
                    bestChildSymmetryIndex = symmetry;
                }
            }
        }
        moveData.best_child_symmetry_index = bestChildSymmetryIndex;

        // Inline deep search (optional heavy path)
        moveData.deep_result = DeepSearchResult();
        moveData.deep_result.sym_hash = moveData.sym_hash;
        moveData.deep_result.state_hash = moveData.state_hash;
        moveData.deep_result.player_to_move = moveData.player_to_move;
        moveData.deep_result.symmetry_index = moveData.symmetry_index;
        moveData.deep_result.move_number = moveNumber;
        moveData.deep_result.komi = rootHistory.rules.komi;
        moveData.deep_result.black_stones = moveData.black_stones;
        moveData.deep_result.white_stones = moveData.white_stones;
        moveData.deep_search_status = "not_run";
        moveData.deep_search_elapsed = 0.0;
        moveData.deep_search_visits = 0;
        moveData.has_deep_analysis = false;
        runInlineDeepSearch(search, board, rootHistory, pla, moveNumber, moveData);

        // 11. Store the completed PerMoveRAGData
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


void writeCompleteRAGDataJSON(float komi, int board_size, const std::string& rules, const GameRAGData* ragData, const FinishedGameData* gameData) {
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
    for (size_t i = 0; i < ragData->flagged_positions.size(); i++) {
        const PerMoveRAGData& moveData = ragData->flagged_positions[i];

        outfile << "    {\n";
        // 1. sym_hash
        outfile << "      \"sym_hash\": \"" << moveData.sym_hash << "\",\n";

        // 2. state_hash
        outfile << "      \"state_hash\": \"" << moveData.state_hash << "\",\n";

        // 3. player_to_move
        outfile << "      \"player_to_move\": \"" << moveData.player_to_move << "\",\n";

        // 4. symmetry_index
        outfile << "      \"symmetry_index\": " << moveData.symmetry_index << ",\n";

        // 5. winrate
        outfile << "      \"winrate\": " << moveData.winrate << ",\n";

        // 6. score_lead (value_score)
        outfile << "      \"score_lead\": " << moveData.value_score << ",\n";

        // 7. move_number
        outfile << "      \"move_number\": " << moveData.move_number << ",\n";

        // 8. komi
        outfile << "      \"komi\": " << komi << ",\n";

        // 9. query_id (using game_id for now)
        outfile << "      \"query_id\": \"" << game_id << "_" << moveData.move_number << "\",\n";

        // 10. stone_count
        outfile << "      \"stone_count\": {\n";
        outfile << "        \"black\": " << moveData.black_stones << ",\n";
        outfile << "        \"white\": " << moveData.white_stones << ",\n";
        outfile << "        \"total\": " << (moveData.black_stones + moveData.white_stones) << "\n";
        outfile << "      },\n";
        outfile << "      \"board_state\": [";
        for(size_t idx = 0; idx < moveData.board_state.size(); idx++) {
            outfile << moveData.board_state[idx];
            if(idx + 1 < moveData.board_state.size()) outfile << ", ";
        }
        outfile << "],\n";
        outfile << "      \"superko_bans\": [";
        for(size_t idx = 0; idx < moveData.superko_bans.size(); idx++) {
            outfile << moveData.superko_bans[idx];
            if(idx + 1 < moveData.superko_bans.size()) outfile << ", ";
        }
        outfile << "],\n";
        outfile << "      \"history_context\": {\n";
        outfile << "        \"board_x_size\": " << moveData.board_x_size << ",\n";
        outfile << "        \"board_y_size\": " << moveData.board_y_size << ",\n";
        outfile << "        \"encore_phase\": " << moveData.encore_phase << ",\n";
        outfile << "        \"num_turns_this_phase\": " << moveData.num_turns_this_phase << ",\n";
        outfile << "        \"num_consec_valid_turns\": " << moveData.num_consec_valid_turns_this_game << ",\n";
        outfile << "        \"consecutive_ending_passes\": " << moveData.consecutive_ending_passes << ",\n";
        outfile << "        \"assume_multiple_handicap\": " << (moveData.assume_multiple_handicap ? "true" : "false") << ",\n";
        outfile << "        \"is_past_normal_phase_end\": " << (moveData.is_past_normal_phase_end ? "true" : "false") << ",\n";
        outfile << "        \"is_game_finished\": " << (moveData.is_game_finished ? "true" : "false") << ",\n";
        outfile << "        \"presumed_next_move\": \"" << moveData.presumed_next_move << "\"\n";
        outfile << "      },\n";
        outfile << "      \"search_params\": " << (moveData.search_params_json.empty() ? "{}" : moveData.search_params_json) << ",\n";

        // 11. best_move
        outfile << "      \"best_move\": {\n";
        outfile << "        \"symmetry_index\": " << moveData.best_child_symmetry_index << ",\n";
        outfile << "        \"hash\": \"" << moveData.best_child_hash << "\",\n";
        outfile << "        \"move\": \"" << moveData.best_child_move << "\"\n";
        outfile << "      },\n";

        outfile << "      \"deep_result\": {\n";
        outfile << "        \"available\": " << (moveData.deep_result.available ? "true" : "false") << ",\n";
        outfile << "        \"status\": \"" << moveData.deep_result.status << "\",\n";
        outfile << "        \"visits\": " << moveData.deep_result.visits << ",\n";
        outfile << "        \"elapsed_seconds\": " << moveData.deep_result.elapsed << ",\n";
        outfile << "        \"sym_hash\": \"" << moveData.deep_result.sym_hash << "\",\n";
        outfile << "        \"state_hash\": \"" << moveData.deep_result.state_hash << "\",\n";
        outfile << "        \"player_to_move\": \"" << moveData.deep_result.player_to_move << "\",\n";
        outfile << "        \"symmetry_index\": " << moveData.deep_result.symmetry_index << ",\n";
        outfile << "        \"move_number\": " << moveData.deep_result.move_number << ",\n";
        outfile << "        \"komi\": " << komi << ",\n";
        outfile << "        \"stone_count\": {\n";
        outfile << "          \"black\": " << moveData.deep_result.black_stones << ",\n";
        outfile << "          \"white\": " << moveData.deep_result.white_stones << ",\n";
        outfile << "          \"total\": " << (moveData.deep_result.black_stones + moveData.deep_result.white_stones) << "\n";
        outfile << "        },\n";
        outfile << "        \"best_move\": {\n";
        outfile << "          \"symmetry_index\": " << moveData.deep_result.best_child_symmetry_index << ",\n";
        outfile << "          \"hash\": \"" << moveData.deep_result.best_child_hash << "\",\n";
        outfile << "          \"move\": \"" << moveData.deep_result.best_child_move << "\"\n";
        outfile << "        },\n";
        outfile << "        \"winrate\": " << moveData.deep_result.winrate << ",\n";
        outfile << "        \"score_lead\": " << moveData.deep_result.score_lead << ",\n";
        outfile << "        \"policy\": [";
        for(size_t dp = 0; dp < moveData.deep_result.policy.size(); dp++) {
            outfile << moveData.deep_result.policy[dp];
            if(dp + 1 < moveData.deep_result.policy.size()) outfile << ", ";
        }
        outfile << "],\n";
        outfile << "        \"ownership\": [";
        for(size_t doIdx = 0; doIdx < moveData.deep_result.ownership.size(); doIdx++) {
            outfile << moveData.deep_result.ownership[doIdx];
            if(doIdx + 1 < moveData.deep_result.ownership.size()) outfile << ", ";
        }
        outfile << "],\n";
        outfile << "        \"children\": [\n";
        for(size_t k = 0; k < moveData.deep_result.children.size(); k++) {
            const ChildNodeInfo& child = moveData.deep_result.children[k];
            outfile << "          {\n";
            outfile << "            \"move\": \"" << child.move << "\",\n";
            outfile << "            \"child_sym_hash\": \"" << child.child_sym_hash << "\",\n";
            outfile << "            \"value\": " << child.value << ",\n";
            outfile << "            \"prior\": " << child.prior << ",\n";
            outfile << "            \"visits\": " << child.visits << "\n";
            outfile << "          }";
            if(k + 1 < moveData.deep_result.children.size()) outfile << ",";
            outfile << "\n";
        }
        outfile << "        ]\n";
        outfile << "      },\n";

        // 12. policy vector
        outfile << "      \"policy\": [";
        for (size_t p = 0; p < moveData.policy.size(); p++) {
            outfile << moveData.policy[p];
            if (p < moveData.policy.size() - 1) outfile << ", ";
        }
        outfile << "],\n";

        // 13. ownership vector
        outfile << "      \"ownership\": [";
        for (size_t o = 0; o < moveData.ownership.size(); o++) {
            outfile << moveData.ownership[o];
            if (o < moveData.ownership.size() - 1) outfile << ", ";
        }
        outfile << "],\n";

        // 14. moves_history
        outfile << "      \"moves_history\": [\n";
        for (size_t j = 0; j < moveData.moves_history.size(); j++) {
            outfile << "        [\"" << moveData.moves_history[j].first << "\", \"" << moveData.moves_history[j].second << "\"]";
            if (j < moveData.moves_history.size() - 1) outfile << ",";
            outfile << "\n";
        }
        outfile << "      ],\n";

        // 15. uncertainty_metrics (moved after moves_history)
        outfile << "      \"uncertainty_metrics\": {\n";
        outfile << "        \"policy_entropy\": " << moveData.policy_entropy << ",\n";
        outfile << "        \"value_variance\": " << moveData.value_variance << ",\n";
        outfile << "        \"combined_score\": " << moveData.combined_score << "\n";
        outfile << "      },\n";

        // 16. children array
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
        if (i < ragData->flagged_positions.size() - 1) outfile << ",";
        outfile << "\n";
    }

    outfile << "  ],\n";

    // Write summary
    int total_flagged = ragData->flagged_positions.size();
    int total_moves = gameData->endHist.moveHistory.size();
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
    currentGameRAGData.inline_deep_searches_run = 0;
    // moves_history is no longer maintained in currentGameRAGData - it's extracted from rootHistory
    // currentGameRAGData.moves_history.clear();
}
