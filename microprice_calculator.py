import pandas as pd
import numpy as np
from scipy.linalg import block_diag
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


class MicropriceCalculator:
    """
    Microprice estimator based on the paper:
    "The microprice: a high frequency estimator of future prices"
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694
    """
    
    def __init__(self, n_imb=10, n_spread=2, dt=1):
        """
        Initialize the microprice calculator.
        
        Parameters:
        -----------
        n_imb : int
            Number of imbalance buckets (default: 10)
        n_spread : int
            Maximum number of tick spreads to consider (default: 2)
        dt : int
            Time step for next price prediction (default: 1)
        """
        self.n_imb = n_imb
        self.n_spread = n_spread
        self.dt = dt
        
    def prepare_data(self, data):
        """
        Prepare and symmetrize market data for microprice calculation.
        Follows the exact logic from the original notebook.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with columns: ['bid', 'ask', 'bs', 'as'] where
            - bid: bid price
            - ask: ask price  
            - bs: bid size
            - as: ask size
            
        Returns:
        --------
        tuple: (prepared_data, tick_size)
        """
        T = data.copy()
        
        # Calculate spread and determine tick size (exactly as in notebook)
        spread = T.ask - T.bid
        ticksize = np.round(min(spread.loc[spread > 0]) * 100) / 100
        T.spread = T.ask - T.bid
        
        # Add spread and mid prices (exactly as in notebook)
        T['spread'] = np.round((T['ask'] - T['bid']) / ticksize) * ticksize
        T['mid'] = (T['bid'] + T['ask']) / 2
        
        # Filter out spreads >= n_spread (exactly as in notebook)
        T = T.loc[(T.spread <= self.n_spread * ticksize) & (T.spread > 0)]
        T['imb'] = T['bs'] / (T['bs'] + T['as'])
        
        # Discretize imbalance into percentiles (exactly as in notebook)
        T['imb_bucket'] = pd.qcut(T['imb'], self.n_imb, labels=False)
        T['next_mid'] = T['mid'].shift(-self.dt)
        
        # Step ahead state variables (exactly as in notebook)
        T['next_spread'] = T['spread'].shift(-self.dt)
        if 'time' in T.columns:
            T['next_time'] = T['time'].shift(-self.dt)
        T['next_imb_bucket'] = T['imb_bucket'].shift(-self.dt)
        
        # Step ahead change in price (exactly as in notebook)
        T['dM'] = np.round((T['next_mid'] - T['mid']) / ticksize * 2) * ticksize / 2
        T = T.loc[(T.dM <= ticksize * 1.1) & (T.dM >= -ticksize * 1.1)]
        
        # Symmetrize data (exactly as in notebook)
        T2 = T.copy(deep=True)
        T2['imb_bucket'] = self.n_imb - 1 - T2['imb_bucket']
        T2['next_imb_bucket'] = self.n_imb - 1 - T2['next_imb_bucket']
        T2['dM'] = -T2['dM']
        T2['mid'] = -T2['mid']
        T3 = pd.concat([T, T2])
        T3.index = pd.RangeIndex(len(T3.index))
        
        return T3, ticksize
    
    def estimate_transition_matrices(self, T):
        """
        Estimate transition probability matrices Q, R1, R2 from market data.
        Follows the exact logic from the original notebook.
        
        Parameters:
        -----------
        T : pandas.DataFrame
            Prepared market data from prepare_data()
            
        Returns:
        --------
        tuple: (G1, B, Q, Q2, R1, R2, K)
            - G1: First microprice adjustment
            - B: Transition matrix for recursive calculation
            - Q, Q2: Transient state transition matrices
            - R1, R2: Absorbing state transition matrices
            - K: Price change values
        """
        # States where mid price doesn't move (exactly as in notebook)
        no_move = T[T['dM'] == 0]
        no_move_counts = no_move.pivot_table(
            index=['next_imb_bucket'], 
            columns=['spread', 'imb_bucket'], 
            values='time' if 'time' in T.columns else T.columns[0],
            fill_value=0, 
            aggfunc='count'
        ).unstack()
        
        # Build Q matrix exactly as in notebook
        Q_counts = np.resize(
            np.array(no_move_counts[0:(self.n_imb * self.n_imb)]),
            (self.n_imb, self.n_imb)
        )
        
        # Loop over all spreads and add block matrices (exactly as in notebook)
        for i in range(1, self.n_spread):
            Qi = np.resize(
                np.array(no_move_counts[(i*self.n_imb*self.n_imb):(i+1)*(self.n_imb*self.n_imb)]),
                (self.n_imb, self.n_imb)
            )
            Q_counts = block_diag(Q_counts, Qi)
        
        # States where mid price moves (R1 matrix) - exactly as in notebook
        move_counts = T[(T['dM'] != 0)].pivot_table(
            index=['dM'], 
            columns=['spread', 'imb_bucket'], 
            values='time' if 'time' in T.columns else T.columns[0],
            fill_value=0, 
            aggfunc='count'
        ).unstack()
        
        R_counts = np.resize(np.array(move_counts), (self.n_imb * self.n_spread, 4))
        T1 = np.concatenate((Q_counts, R_counts), axis=1).astype(float)
        
        # Normalize rows (exactly as in notebook)
        for i in range(0, self.n_imb * self.n_spread):
            T1[i] = T1[i] / T1[i].sum()
        
        Q = T1[:, 0:(self.n_imb * self.n_spread)]
        R1 = T1[:, (self.n_imb * self.n_spread):]
        
        # Price change values (exactly as in notebook)
        K = np.array([-0.01, -0.005, 0.005, 0.01])
        
        # R2 matrix for next state transitions (exactly as in notebook)
        move_counts = T[(T['dM'] != 0)].pivot_table(
            index=['spread', 'imb_bucket'], 
            columns=['next_spread', 'next_imb_bucket'], 
            values='time' if 'time' in T.columns else T.columns[0],
            fill_value=0, 
            aggfunc='count'
        )  # Note: no .unstack() here as in original
        
        R2_counts = np.resize(np.array(move_counts), (self.n_imb * self.n_spread, self.n_imb * self.n_spread))
        T2 = np.concatenate((Q_counts, R2_counts), axis=1).astype(float)
        
        for i in range(0, self.n_imb * self.n_spread):
            T2[i] = T2[i] / T2[i].sum()
        
        R2 = T2[:, (self.n_imb * self.n_spread):]
        Q2 = T2[:, 0:(self.n_imb * self.n_spread)]
        
        # Calculate G1 and B (exactly as in notebook)
        G1 = np.dot(np.dot(np.linalg.inv(np.eye(self.n_imb * self.n_spread) - Q), R1), K)
        B = np.dot(np.linalg.inv(np.eye(self.n_imb * self.n_spread) - Q), R2)
        
        return G1, B, Q, Q2, R1, R2, K
    
    def calculate_microprice_adjustment(self, G1, B, n_steps=6):
        """
        Calculate microprice adjustment using recursive formula.
        Follows the exact logic from the original notebook.
        
        Parameters:
        -----------
        G1 : numpy.ndarray
            First microprice adjustment
        B : numpy.ndarray
            Transition matrix
        n_steps : int
            Number of price moves to consider (default: 6)
            
        Returns:
        --------
        numpy.ndarray: Cumulative microprice adjustment (G6)
        """
        # Follow exact calculation as in notebook cell-26
        G2 = np.dot(B, G1) + G1
        G3 = G2 + np.dot(np.dot(B, B), G1)
        G4 = G3 + np.dot(np.dot(np.dot(B, B), B), G1)
        G5 = G4 + np.dot(np.dot(np.dot(np.dot(B, B), B), B), G1)
        G6 = G5 + np.dot(np.dot(np.dot(np.dot(np.dot(B, B), B), B), B), G1)
        
        return G6
    
    def get_state_index(self, imbalance, spread, ticksize):
        """
        Get state index for given imbalance and spread.
        
        Parameters:
        -----------
        imbalance : float
            Order book imbalance (0 to 1)
        spread : float
            Bid-ask spread
        ticksize : float
            Tick size
            
        Returns:
        --------
        int: State index
        """
        # Discretize imbalance into buckets
        imb_bucket = min(int(imbalance * self.n_imb), self.n_imb - 1)
        
        # Discretize spread into ticks
        spread_bucket = min(int(np.round(spread / ticksize)) - 1, self.n_spread - 1)
        spread_bucket = max(0, spread_bucket)
        
        return spread_bucket * self.n_imb + imb_bucket
    
    def calculate_microprice(self, data, bid_price, ask_price, bid_size, ask_size):
        """
        Calculate microprice for given market state.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical market data for calibration
        bid_price : float
            Current bid price
        ask_price : float
            Current ask price
        bid_size : float
            Current bid size
        ask_size : float
            Current ask size
            
        Returns:
        --------
        dict: Dictionary containing microprice and related metrics
        """
        # Prepare historical data
        prepared_data, ticksize = self.prepare_data(data)
        
        # Estimate transition matrices
        G1, B, Q, Q2, R1, R2, K = self.estimate_transition_matrices(prepared_data)
        
        # Calculate microprice adjustment
        microprice_adjustment = self.calculate_microprice_adjustment(G1, B)
        
        # Calculate current market state
        mid_price = (bid_price + ask_price) / 2
        imbalance = bid_size / (bid_size + ask_size)
        spread = ask_price - bid_price
        
        # Get state index and microprice adjustment
        state_idx = self.get_state_index(imbalance, spread, ticksize)
        
        if state_idx < len(microprice_adjustment):
            adjustment = microprice_adjustment[state_idx]
        else:
            adjustment = 0.0
        
        # Calculate final microprice
        microprice = mid_price + adjustment
        
        return {
            'microprice': microprice,
            'mid_price': mid_price,
            'adjustment': adjustment,
            'imbalance': imbalance,
            'spread': spread,
            'tick_size': ticksize,
            'state_index': state_idx
        }


def calculate_microprice_simple(data, bid_price, ask_price, bid_size, ask_size, 
                               n_imb=10, n_spread=2, n_steps=6):
    """
    Simple wrapper function to calculate microprice.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Historical market data with columns ['bid', 'ask', 'bs', 'as']
    bid_price : float
        Current bid price
    ask_price : float
        Current ask price
    bid_size : float
        Current bid size  
    ask_size : float
        Current ask size
    n_imb : int
        Number of imbalance buckets (default: 10)
    n_spread : int
        Maximum spread in ticks (default: 2)
    n_steps : int
        Number of price moves for convergence (default: 6)
        
    Returns:
    --------
    float: Microprice estimate
    """
    calculator = MicropriceCalculator(n_imb=n_imb, n_spread=n_spread)
    result = calculator.calculate_microprice(data, bid_price, ask_price, bid_size, ask_size)
    return result['microprice']


if __name__ == "__main__":
    # Example usage
    print("Microprice Calculator")
    print("Usage example:")
    print("""
    import pandas as pd
    
    # Load your market data
    data = pd.DataFrame({
        'bid': [14.33, 14.32, 14.31],
        'ask': [14.34, 14.33, 14.32], 
        'bs': [100, 150, 200],
        'as': [200, 100, 180]
    })
    
    # Calculate microprice
    calculator = MicropriceCalculator()
    result = calculator.calculate_microprice(
        data, 
        bid_price=14.33, 
        ask_price=14.34, 
        bid_size=100, 
        ask_size=200
    )
    
    print(f"Microprice: {result['microprice']:.4f}")
    print(f"Mid price: {result['mid_price']:.4f}")
    print(f"Adjustment: {result['adjustment']:.4f}")
    """)