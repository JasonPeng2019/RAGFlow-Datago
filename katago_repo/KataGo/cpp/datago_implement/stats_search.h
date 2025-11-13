#ifndef DATAGO_STATS_SEARCH_H
#define DATAGO_STATS_SEARCH_H

#include "../search/search.h"
/*pseudocode:
void collectRAGData(Search* search, SearchNode* rootNode) {
    // 1. Get all nodes in the search tree
    std::vector<SearchNode*> allNodes = search->enumerateTreePostOrder();
    
    // 2. For each node, extract the data you need
    for (SearchNode* node : allNodes) {
        // Get the node's hash (game state identifier)
        Hash128 stateHash = node->hash;  // From searchnode.h line 220
        
        // Get final values
        NodeStats stats;
        node->stats.copyInto(stats);  // Thread-safe copy of atomic values
        
        double utility = stats.utilityAvg;
        double winrate = stats.winLossValueAvg;
        double scoreMean = stats.scoreMeanAvg;
        int64_t visits = stats.visits;
        
        // Get children and their edges
        const SearchChildPointer* children = node->getChildren();
        int numChildren = node->numChildren;
        
        // 3. Extract policy distribution for children
        std::vector<double> policyValues;
        std::vector<Hash128> childHashes;
        std::vector<double> childUtilities;
        
        for (int i = 0; i < numChildren; i++) {
            const SearchChildPointer& child = children[i];
            
            // Policy prior for this child
            double prior = child.nnPolicyProb;  // searchnode.h line 143
            
            // Edge visits
            int64_t edgeVisits = child.getEdgeVisits();  // searchnode.h line 158
            
            // Child node (if expanded)
            SearchNode* childNode = child.getIfAllocated();
            if (childNode != nullptr) {
                Hash128 childHash = childNode->hash;
                
                NodeStats childStats;
                childNode->stats.copyInto(childStats);
                double childUtility = childStats.utilityAvg;
                
                childHashes.push_back(childHash);
                childUtilities.push_back(childUtility);
            }
            
            policyValues.push_back(prior);
        }
        
        // 4. Store to RAG (your storage logic here)
        storeToRAG(stateHash, utility, winrate, scoreMean, 
                   policyValues, childHashes, childUtilities, visits);
    }
}*/

void datago_collect_search_states(struct Search* search, struct SearchNode* rootNode);

#endif  // DATAGO_STATS_SEARCH_H