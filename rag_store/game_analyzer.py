import json
import subprocess
from typing import Optional

class GoStateEmbedding:
    """
    Stores a complete state embedding from KataGo analysis.
    All data comes from KataGo's JSON analysis response.
    """
    
    def __init__(self, katago_response: dict, query_info: dict):
        """
        Args:
            katago_response: The JSON response from KataGo analysis engine
            query_info: Your original query data (for komi, rules)
        """
        
        # 1. Compact feature vector (your "fingerprint")
        # Source: response['rootInfo']['thisHash']
        self.state_hash = katago_response['rootInfo']['thisHash']  # 128-bit hash string
        
        # 2. Neural net evaluation vectors
        # Source: response['policy'] - only present if you set includePolicy=True
        self.policy = katago_response.get('policy', None)  # 362 floats (361 + pass)
        
        # Source: response['ownership'] - only present if you set includeOwnership=True
        self.ownership = katago_response.get('ownership', None)  # 361 floats
        
        # 3. Value head outputs (from rootInfo after MCTS search)
        # Source: response['rootInfo']['winrate']
        self.winrate = katago_response['rootInfo']['winrate']
        
        # Source: response['rootInfo']['scoreLead']
        self.score_lead = katago_response['rootInfo']['scoreLead']
        
        # Source: response['rootInfo']['utility']
        self.utility = katago_response['rootInfo']['utility']
        
        # Source: response['rootInfo']['scoreStdev']
        self.score_stdev = katago_response['rootInfo']['scoreStdev']
        
        # Additional value head outputs (optional, check if present)
        # Source: response['rootInfo']['scoreSelfplay']
        self.score_selfplay = katago_response['rootInfo'].get('scoreSelfplay', None)
        
        # Raw neural net outputs (before search) - optional
        # Source: response['rootInfo']['rawWinrate'] etc.
        self.raw_winrate = katago_response['rootInfo'].get('rawWinrate', None)
        self.raw_score_lead = katago_response['rootInfo'].get('rawLead', None)
        self.raw_score_stdev = katago_response['rootInfo'].get('rawScoreSelfplayStdev', None)
        
        # 4. Search statistics (for context)
        # Source: response['rootInfo']['visits']
        self.visits = katago_response['rootInfo']['visits']
        
        # Source: response['rootInfo']['lcb']
        self.lcb = katago_response['rootInfo']['lcb']
        
        # 5. Game context
        # Source: response['turnNumber']
        self.turn_number = katago_response['turnNumber']
        
        # Source: response['rootInfo']['currentPlayer']
        self.player_to_move = katago_response['rootInfo']['currentPlayer']  # "B" or "W"
        
        # Source: from your original query
        self.komi = query_info['komi']
        self.rules = query_info['rules']
        self.board_size_x = query_info['boardXSize']
        self.board_size_y = query_info['boardYSize']
        
        # 6. Symmetry hash (useful for detecting equivalent positions)
        # Source: response['rootInfo']['symHash']
        self.sym_hash = katago_response['rootInfo']['symHash']
        
        # 7. Query identifier
        # Source: response['id']
        self.query_id = katago_response['id']
    
    def to_dict(self):
        """Convert to dictionary for storage in database"""
        return {
            'state_hash': self.state_hash,
            'sym_hash': self.sym_hash,
            'policy': self.policy,
            'ownership': self.ownership,
            'winrate': self.winrate,
            'score_lead': self.score_lead,
            'utility': self.utility,
            'score_stdev': self.score_stdev,
            'score_selfplay': self.score_selfplay,
            'raw_winrate': self.raw_winrate,
            'raw_score_lead': self.raw_score_lead,
            'raw_score_stdev': self.raw_score_stdev,
            'visits': self.visits,
            'lcb': self.lcb,
            'turn_number': self.turn_number,
            'player_to_move': self.player_to_move,
            'komi': self.komi,
            'rules': self.rules,
            'board_size_x': self.board_size_x,
            'board_size_y': self.board_size_y,
            'query_id': self.query_id,
        }

class KataGoAnalyzer:
    """Wrapper for KataGo analysis engine"""
    
    def __init__(self, katago_path: str, config_path: str, model_path: str):
        self.katago = subprocess.Popen(
            [katago_path, 'analysis', '-config', config_path, '-model', model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.query_counter = 0
    
    def analyze_position(self, moves: list, komi: float = 7.5, 
                        rules: str = "chinese",
                        board_size: int = 19,
                        max_visits: Optional[int] = None) -> GoStateEmbedding:
        """
        Analyze a position and return embedding.
        
        Args:
            moves: List of [player, location] like [["B","Q4"], ["W","D4"]]
            komi: Game komi
            rules: Rule set (chinese, japanese, tromp-taylor, etc.)
            board_size: Board size (default 19x19)
            max_visits: Optional visit limit
        
        Returns:
            GoStateEmbedding with all extracted features
        """
        
        # Build query
        query = {
            "id": f"query_{self.query_counter}",
            "moves": moves,
            "rules": rules,
            "komi": komi,
            "boardXSize": board_size,
            "boardYSize": board_size,
            
            # IMPORTANT: Request the data you need!
            "includePolicy": True,      # Get raw neural net policy
            "includeOwnership": True,   # Get territory predictions
            # Optional: "includeOwnershipStdev": True,
            # Optional: "maxVisits": max_visits,
        }
        
        self.query_counter += 1
        
        # Send query
        self.katago.stdin.write(json.dumps(query) + '\n')
        self.katago.stdin.flush()
        
        # Read response
        response_line = self.katago.stdout.readline()
        response = json.loads(response_line)
        
        # Check for errors
        if 'error' in response:
            raise RuntimeError(f"KataGo error: {response['error']}")
        
        # Create embedding
        return GoStateEmbedding(response, query)
    
    def close(self):
        self.katago.stdin.close()
        self.katago.wait()


# Example Usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = KataGoAnalyzer(
        katago_path="./katago",
        config_path="cpp/configs/analysis_example.cfg",
        model_path="cpp/tests/models/g170e-b10c128-s1141046784-d204142634.bin.gz"
    )
    
    # Analyze a position
    moves = [
        ["B", "Q4"],
        ["W", "D4"],
        ["B", "Q16"],
        ["W", "D16"],
    ]
    
    embedding = analyzer.analyze_position(
        moves=moves,
        komi=7.5,
        rules="chinese",
        max_visits=100  # Quick analysis
    )
    
    # Print results
    print(f"State Hash: {embedding.state_hash}")
    print(f"Turn: {embedding.turn_number}")
    print(f"Player to move: {embedding.player_to_move}")
    print(f"Winrate: {embedding.winrate:.3f}")
    print(f"Score Lead: {embedding.score_lead:.2f}")
    print(f"Visits: {embedding.visits}")
    print(f"Policy shape: {len(embedding.policy) if embedding.policy else 'None'}")
    print(f"Ownership shape: {len(embedding.ownership) if embedding.ownership else 'None'}")
    
    # Convert to dict for storage
    data = embedding.to_dict()
    print(f"\nStorable dict keys: {list(data.keys())}")
    
    analyzer.close()