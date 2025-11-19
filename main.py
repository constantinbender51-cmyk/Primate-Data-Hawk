import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import seaborn as sns

class BitcoinCorrelationData:
    def __init__(self):
        self.data = {}
        
    def safe_yf_download(self, ticker, period="2y", interval="1d"):
        """Safely download data from Yahoo Finance with error handling"""
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if data.empty:
                print(f"Warning: No data returned for {ticker}")
                return None
            return data['Close']
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return None
        
    def fetch_btc_price(self, period="2y"):
        """Fetch Bitcoin price data from Yahoo Finance"""
        try:
            btc_data = self.safe_yf_download("BTC-USD", period)
            if btc_data is not None:
                self.data['BTC'] = btc_data
                print("✓ Bitcoin price data fetched")
                return btc_data
            else:
                print("✗ Failed to fetch Bitcoin data")
                return None
        except Exception as e:
            print(f"Error fetching BTC data: {e}")
            return None

    def fetch_sp500(self, period="2y"):
        """Fetch S&P 500 data"""
        try:
            sp500_data = self.safe_yf_download("^GSPC", period)
            if sp500_data is not None:
                self.data['SP500'] = sp500_data
                print("✓ S&P 500 data fetched")
                return sp500_data
            else:
                print("✗ Failed to fetch S&P 500 data")
                return None
        except Exception as e:
            print(f"Error fetching SP500 data: {e}")
            return None

    def fetch_dollar_index(self, period="2y"):
        """Fetch US Dollar Index (DXY)"""
        try:
            dxy_data = self.safe_yf_download("DX=F", period)
            if dxy_data is not None:
                self.data['DXY'] = dxy_data
                print("✓ Dollar Index (DXY) data fetched")
                return dxy_data
            else:
                print("✗ Failed to fetch DXY data")
                return None
        except Exception as e:
            print(f"Error fetching DXY data: {e}")
            return None

    def fetch_treasury_yields(self, period="2y"):
        """Fetch US Treasury yields (10-year)"""
        try:
            # Using Treasury ETF as proxy for yields
            tlt_data = self.safe_yf_download("TLT", period)
            if tlt_data is not None:
                self.data['Treasury_Yield_Proxy'] = tlt_data
                print("✓ Treasury yields data fetched")
                return tlt_data
            else:
                print("✗ Failed to fetch Treasury data")
                return None
        except Exception as e:
            print(f"Error fetching Treasury data: {e}")
            return None

    def fetch_gold_price(self, period="2y"):
        """Fetch Gold price"""
        try:
            gold_data = self.safe_yf_download("GC=F", period)
            if gold_data is not None:
                self.data['Gold'] = gold_data
                print("✓ Gold price data fetched")
                return gold_data
            else:
                print("✗ Failed to fetch Gold data")
                return None
        except Exception as e:
            print(f"Error fetching Gold data: {e}")
            return None

    def fetch_crypto_fear_greed(self, limit=365):
        """Fetch Crypto Fear & Greed Index from alternative source"""
        try:
            # Using Alternative.me API (free tier)
            url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()['data']
                # Convert to DataFrame with date index
                fear_greed_data = []
                for item in data:
                    fear_greed_data.append({
                        'Date': datetime.fromtimestamp(int(item['timestamp'])),
                        'Fear_Greed': int(item['value'])
                    })
                df = pd.DataFrame(fear_greed_data)
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)
                self.data['Fear_Greed'] = df['Fear_Greed']
                print("✓ Fear & Greed Index data fetched")
                return df['Fear_Greed']
            else:
                print("✗ Could not fetch Fear & Greed data")
                return None
        except Exception as e:
            print(f"Error fetching Fear & Greed data: {e}")
            return None

    def fetch_total_crypto_market_cap(self):
        """Fetch total crypto market cap (simplified using BTC dominance proxy)"""
        try:
            # Using multiple crypto assets as proxy for total market cap
            eth_data = self.safe_yf_download("ETH-USD", "2y")
            if eth_data is not None:
                self.data['ETH'] = eth_data
                print("✓ Ethereum data fetched as market proxy")
                return eth_data
            else:
                print("✗ Failed to fetch Ethereum data")
                return None
        except Exception as e:
            print(f"Error fetching crypto market data: {e}")
            return None

    def fetch_vix(self, period="2y"):
        """Fetch VIX volatility index"""
        try:
            vix_data = self.safe_yf_download("^VIX", period)
            if vix_data is not None:
                self.data['VIX'] = vix_data
                print("✓ VIX data fetched")
                return vix_data
            else:
                print("✗ Failed to fetch VIX data")
                return None
        except Exception as e:
            print(f"Error fetching VIX data: {e}")
            return None

    def fetch_on_chain_metrics_proxy(self):
        """Proxy for on-chain metrics using trading volume and price action"""
        try:
            btc_data = yf.download("BTC-USD", period="2y", interval="1d", progress=False)
            if btc_data.empty:
                print("✗ Failed to fetch BTC data for on-chain metrics")
                return None, None
                
            # Using volume and price range as proxies
            self.data['BTC_Volume'] = btc_data['Volume']
            # Calculate daily volatility
            self.data['BTC_Volatility'] = (btc_data['High'] - btc_data['Low']) / btc_data['Close']
            print("✓ On-chain metrics proxies created")
            return self.data['BTC_Volume'], self.data['BTC_Volatility']
        except Exception as e:
            print(f"Error creating on-chain proxies: {e}")
            return None, None

    def validate_data(self):
        """Validate that all data has proper index and is not scalar"""
        valid_data = {}
        for key, value in self.data.items():
            if hasattr(value, 'index') and len(value) > 1:
                valid_data[key] = value
                print(f"✓ {key}: Valid data with {len(value)} points")
            else:
                print(f"✗ {key}: Invalid data (scalar or empty)")
        
        self.data = valid_data
        return len(valid_data) > 0

    def compile_all_data(self, period="2y"):
        """Fetch all correlation data"""
        print("Starting data collection...")
        
        # Fetch all data sources
        self.fetch_btc_price(period)
        self.fetch_sp500(period)
        self.fetch_dollar_index(period)
        self.fetch_treasury_yields(period)
        self.fetch_gold_price(period)
        self.fetch_vix(period)
        self.fetch_total_crypto_market_cap()
        self.fetch_on_chain_metrics_proxy()
        
        # Fear & Greed index (slower API)
        self.fetch_crypto_fear_greed()
        
        # Validate data before combining
        if self.validate_data():
            # Combine all data into single DataFrame
            self.combine_data()
            print("✓ All data compiled successfully!")
        else:
            print("✗ Data validation failed - cannot compile data")

    def combine_data(self):
        """Combine all data sources into a single DataFrame"""
        if 'BTC' not in self.data:
            print("✗ No BTC data available for combination")
            return None
        
        # Create base DataFrame with BTC price
        combined_df = pd.DataFrame({'BTC_Price': self.data['BTC']})
        
        # Add other data sources
        data_mapping = {
            'SP500': 'SP500',
            'DXY': 'Dollar_Index',
            'Treasury_Yield_Proxy': 'Treasury_Yield',
            'Gold': 'Gold_Price',
            'VIX': 'VIX',
            'ETH': 'Ethereum_Price',
            'BTC_Volume': 'BTC_Volume',
            'BTC_Volatility': 'BTC_Volatility'
        }
        
        for key, col_name in data_mapping.items():
            if key in self.data:
                # Align the data with the base DataFrame index
                aligned_data = self.data[key].reindex(combined_df.index)
                combined_df[col_name] = aligned_data
        
        # Add Fear & Greed index with date alignment
        if 'Fear_Greed' in self.data:
            fear_greed = self.data['Fear_Greed']
            # Reindex to match the main DataFrame's dates
            fear_greed_aligned = fear_greed.reindex(combined_df.index, method='ffill')
            combined_df['Fear_Greed_Index'] = fear_greed_aligned
        
        self.combined_data = combined_df.dropna()
        print(f"✓ Combined data shape: {self.combined_data.shape}")
        return self.combined_data

    def calculate_correlations(self):
        """Calculate correlation matrix"""
        if not hasattr(self, 'combined_data') or self.combined_data.empty:
            print("No combined data available. Run compile_all_data() first.")
            return None
        
        correlation_matrix = self.combined_data.corr()
        
        # Display correlations with BTC
        btc_correlations = correlation_matrix['BTC_Price'].sort_values(ascending=False)
        
        print("\n" + "="*50)
        print("BITCOIN PRICE CORRELATIONS")
        print("="*50)
        for asset, corr in btc_correlations.items():
            if asset != 'BTC_Price':
                correlation_strength = "STRONG" if abs(corr) > 0.7 else "MODERATE" if abs(corr) > 0.3 else "WEAK"
                print(f"{asset:20} : {corr:+.4f} ({correlation_strength})")
        
        return correlation_matrix

    def plot_correlations(self):
        """Plot correlation heatmap"""
        if not hasattr(self, 'combined_data') or self.combined_data.empty:
            print("No combined data available.")
            return
        
        corr_matrix = self.calculate_correlations()
        if corr_matrix is None:
            return
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
        plt.title('Bitcoin Price Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_time_series(self):
        """Plot normalized time series of all assets"""
        if not hasattr(self, 'combined_data') or self.combined_data.empty:
            print("No combined data available.")
            return
        
        # Normalize data for comparison (set starting point to 100)
        normalized_data = self.combined_data.copy()
        for column in normalized_data.columns:
            if column != 'Fear_Greed_Index':  # Don't normalize the index
                if not normalized_data[column].empty:
                    normalized_data[column] = (normalized_data[column] / normalized_data[column].iloc[0]) * 100
        
        plt.figure(figsize=(14, 8))
        for column in normalized_data.columns:
            if column != 'Fear_Greed_Index' and not normalized_data[column].empty:
                plt.plot(normalized_data.index, normalized_data[column], label=column, linewidth=2)
        
        # Add Fear & Greed index on secondary axis
        if 'Fear_Greed_Index' in normalized_data.columns and not normalized_data['Fear_Greed_Index'].empty:
            ax2 = plt.gca().twinx()
            ax2.plot(normalized_data.index, normalized_data['Fear_Greed_Index'], 
                    label='Fear & Greed Index', color='black', linestyle='--', alpha=0.7)
            ax2.set_ylabel('Fear & Greed Index', fontsize=12)
            ax2.legend(loc='upper right')
        
        plt.title('Normalized Price Comparison (Base=100)', fontsize=16, fontweight='bold')
        plt.ylabel('Normalized Value (Base=100)', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def save_to_csv(self, filename="bitcoin_correlation_data.csv"):
        """Save the combined data to CSV"""
        if hasattr(self, 'combined_data') and not self.combined_data.empty:
            self.combined_data.to_csv(filename)
            print(f"✓ Data saved to {filename}")
        else:
            print("No data to save.")

# Example usage
if __name__ == "__main__":
    # Initialize the data fetcher
    btc_corr = BitcoinCorrelationData()
    
    # Fetch all data (default: 2 years)
    btc_corr.compile_all_data(period="2y")
    
    if hasattr(btc_corr, 'combined_data') and not btc_corr.combined_data.empty:
        # Calculate and display correlations
        correlations = btc_corr.calculate_correlations()
        
        # Create visualizations
        btc_corr.plot_correlations()
        btc_corr.plot_time_series()
        
        # Save data to CSV
        btc_corr.save_to_csv()
        
        # Display first few rows of data
        print("\nFirst 5 rows of compiled data:")
        print(btc_corr.combined_data.head())
        
        print("\nData summary:")
        print(btc_corr.combined_data.describe())
    else:
        print("Failed to compile data. Please check the error messages above.")
