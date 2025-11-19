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
            data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
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
            if btc_data is not None and len(btc_data) > 0:
                self.data['BTC'] = btc_data
                print(f"✓ Bitcoin price data fetched ({len(btc_data)} points)")
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
            if sp500_data is not None and len(sp500_data) > 0:
                self.data['SP500'] = sp500_data
                print(f"✓ S&P 500 data fetched ({len(sp500_data)} points)")
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
            if dxy_data is not None and len(dxy_data) > 0:
                self.data['DXY'] = dxy_data
                print(f"✓ Dollar Index (DXY) data fetched ({len(dxy_data)} points)")
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
            if tlt_data is not None and len(tlt_data) > 0:
                self.data['Treasury_Yield_Proxy'] = tlt_data
                print(f"✓ Treasury yields data fetched ({len(tlt_data)} points)")
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
            if gold_data is not None and len(gold_data) > 0:
                self.data['Gold'] = gold_data
                print(f"✓ Gold price data fetched ({len(gold_data)} points)")
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
                print(f"✓ Fear & Greed Index data fetched ({len(df)} points)")
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
            if eth_data is not None and len(eth_data) > 0:
                self.data['ETH'] = eth_data
                print(f"✓ Ethereum data fetched as market proxy ({len(eth_data)} points)")
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
            if vix_data is not None and len(vix_data) > 0:
                self.data['VIX'] = vix_data
                print(f"✓ VIX data fetched ({len(vix_data)} points)")
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
            btc_data = yf.download("BTC-USD", period="2y", interval="1d", progress=False, auto_adjust=True)
            if btc_data.empty:
                print("✗ Failed to fetch BTC data for on-chain metrics")
                return None, None
                
            # Using volume and price range as proxies
            self.data['BTC_Volume'] = btc_data['Volume']
            # Calculate daily volatility
            self.data['BTC_Volatility'] = (btc_data['High'] - btc_data['Low']) / btc_data['Close']
            print(f"✓ On-chain metrics proxies created ({len(btc_data)} points)")
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
            success = self.combine_data()
            if success:
                print("✓ All data compiled successfully!")
            else:
                print("✗ Failed to combine data")
        else:
            print("✗ Data validation failed - cannot compile data")

    def combine_data(self):
        """Combine all data sources into a single DataFrame"""
        if 'BTC' not in self.data:
            print("✗ No BTC data available for combination")
            return False
        
        try:
            # Create base DataFrame with BTC price - ensure we're using the Series properly
            btc_series = self.data['BTC']
            if not isinstance(btc_series, pd.Series):
                print("✗ BTC data is not a pandas Series")
                return False
                
            # Create the DataFrame properly
            combined_df = pd.DataFrame({
                'BTC_Price': btc_series.values
            }, index=btc_series.index)
            
            print(f"✓ Base DataFrame created with {len(combined_df)} rows")
            
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
            
            added_columns = 0
            for key, col_name in data_mapping.items():
                if key in self.data:
                    try:
                        source_data = self.data[key]
                        if isinstance(source_data, pd.Series):
                            # Align the data with the base DataFrame index
                            aligned_data = source_data.reindex(combined_df.index)
                            combined_df[col_name] = aligned_data
                            added_columns += 1
                            print(f"✓ Added {col_name}")
                        else:
                            print(f"✗ {key} is not a pandas Series")
                    except Exception as e:
                        print(f"✗ Error adding {col_name}: {e}")
            
            # Add Fear & Greed index with date alignment
            if 'Fear_Greed' in self.data:
                try:
                    fear_greed = self.data['Fear_Greed']
                    if isinstance(fear_greed, pd.Series):
                        # Reindex to match the main DataFrame's dates
                        fear_greed_aligned = fear_greed.reindex(combined_df.index, method='ffill')
                        combined_df['Fear_Greed_Index'] = fear_greed_aligned
                        added_columns += 1
                        print("✓ Added Fear_Greed_Index")
                    else:
                        print("✗ Fear_Greed is not a pandas Series")
                except Exception as e:
                    print(f"✗ Error adding Fear_Greed_Index: {e}")
            
            # Remove any columns that are all NaN
            combined_df = combined_df.dropna(axis=1, how='all')
            
            # Drop rows with any NaN values
            initial_rows = len(combined_df)
            combined_df = combined_df.dropna()
            final_rows = len(combined_df)
            
            self.combined_data = combined_df
            print(f"✓ Combined data shape: {combined_df.shape}")
            print(f"✓ Removed {initial_rows - final_rows} rows with missing data")
            print(f"✓ Total columns: {len(combined_df.columns)}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error combining data: {e}")
            return False

    def calculate_correlations(self):
        """Calculate correlation matrix"""
        if not hasattr(self, 'combined_data') or self.combined_data.empty:
            print("No combined data available. Run compile_all_data() first.")
            return None
        
        try:
            correlation_matrix = self.combined_data.corr()
            
            # Display correlations with BTC
            if 'BTC_Price' in correlation_matrix.columns:
                btc_correlations = correlation_matrix['BTC_Price'].sort_values(ascending=False)
                
                print("\n" + "="*50)
                print("BITCOIN PRICE CORRELATIONS")
                print("="*50)
                for asset, corr in btc_correlations.items():
                    if asset != 'BTC_Price':
                        correlation_strength = "STRONG" if abs(corr) > 0.7 else "MODERATE" if abs(corr) > 0.3 else "WEAK"
                        print(f"{asset:20} : {corr:+.4f} ({correlation_strength})")
                
                return correlation_matrix
            else:
                print("BTC_Price column not found in correlation matrix")
                return None
                
        except Exception as e:
            print(f"Error calculating correlations: {e}")
            return None

    def plot_correlations(self):
        """Plot correlation heatmap"""
        if not hasattr(self, 'combined_data') or self.combined_data.empty:
            print("No combined data available.")
            return
        
        corr_matrix = self.calculate_correlations()
        if corr_matrix is None:
            return
        
        try:
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
            plt.title('Bitcoin Price Correlation Matrix', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting correlations: {e}")

    def plot_time_series(self):
        """Plot normalized time series of all assets"""
        if not hasattr(self, 'combined_data') or self.combined_data.empty:
            print("No combined data available.")
            return
        
        try:
            # Normalize data for comparison (set starting point to 100)
            normalized_data = self.combined_data.copy()
            for column in normalized_data.columns:
                if column != 'Fear_Greed_Index':  # Don't normalize the index
                    if len(normalized_data[column]) > 0 and not pd.isna(normalized_data[column].iloc[0]):
                        normalized_data[column] = (normalized_data[column] / normalized_data[column].iloc[0]) * 100
            
            plt.figure(figsize=(14, 8))
            for column in normalized_data.columns:
                if column != 'Fear_Greed_Index' and len(normalized_data[column]) > 0:
                    plt.plot(normalized_data.index, normalized_data[column], label=column, linewidth=2)
            
            # Add Fear & Greed index on secondary axis
            if 'Fear_Greed_Index' in normalized_data.columns and len(normalized_data['Fear_Greed_Index']) > 0:
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
        except Exception as e:
            print(f"Error plotting time series: {e}")

    def save_to_csv(self, filename="bitcoin_correlation_data.csv"):
        """Save the combined data to CSV"""
        if hasattr(self, 'combined_data') and not self.combined_data.empty:
            try:
                self.combined_data.to_csv(filename)
                print(f"✓ Data saved to {filename}")
            except Exception as e:
                print(f"Error saving data: {e}")
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
