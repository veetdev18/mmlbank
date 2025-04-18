#!/usr/bin/env python3

import os
import ccxt
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VolumeBot:
    def __init__(self, exchange_name, symbol=None, use_market_orders=True):
        """
        Initialize volume bot for a specific exchange
        
        Args:
            exchange_name (str): Name of the exchange (lbank, mexc, bingx)
            symbol (str): Trading symbol, if None will be taken from .env
            use_market_orders (bool): Whether to use market orders for matching
        """
        self.exchange_name = exchange_name.lower()
        self.use_market_orders = use_market_orders
        self.own_orders = []  # Track orders placed by this bot
        self.initial_orders = []  # Track initial orders specifically
        self.initial_price = None  # Store initial price to maintain range
        self.price_control_range = float(os.getenv('PRICE_CONTROL_RANGE', '3'))  # Default 3% range
        self.buy_sell_ratio = 1.0  # Equal buy/sell ratio by default
        self.price_direction = os.getenv('PRICE_DIRECTION', 'maintain').lower()  # Price direction control
        self.cancel_initial_orders = os.getenv('CANCEL_INITIAL_ORDERS', 'false').lower() == 'true'  # Whether to cancel initial orders
        
        # Wave pattern settings
        self.use_daily_wave = os.getenv('USE_DAILY_WAVE', 'false').lower() == 'true'
        self.wave_amplitude = float(os.getenv('WAVE_AMPLITUDE', '3.0'))  # Default 3% wave height
        self.wave_cycle_hours = float(os.getenv('WAVE_CYCLE_HOURS', '6.0'))  # Default 6-hour cycle (3 up, 3 down)
        self.wave_reference_price = None  # Reference price for the wave
        self.wave_reference_time = None  # Reference time for the wave
        
        # Balance recovery strategy
        self.balance_recovery_strategy = os.getenv('BALANCE_RECOVERY_STRATEGY', 'cancel_orders').lower()  # 'cancel_orders' or 'sell_assets'
        
        # Order book depth maintenance
        self.maintain_order_book = os.getenv('MAINTAIN_ORDER_BOOK', 'false').lower() == 'true'
        self.order_book_depth = float(os.getenv('ORDER_BOOK_DEPTH', '2.0'))  # Depth range in percentage (2%)
        self.bid_orders_count = int(os.getenv('BID_ORDERS_COUNT', '5'))  # Number of buy orders to maintain
        self.ask_orders_count = int(os.getenv('ASK_ORDERS_COUNT', '5'))  # Number of sell orders to maintain
        self.min_bid_amount = float(os.getenv('MIN_BID_AMOUNT', '10'))  # Minimum amount for buy orders
        self.max_bid_amount = float(os.getenv('MAX_BID_AMOUNT', '50'))  # Maximum amount for buy orders
        self.min_ask_amount = float(os.getenv('MIN_ASK_AMOUNT', '10'))  # Minimum amount for sell orders
        self.max_ask_amount = float(os.getenv('MAX_ASK_AMOUNT', '50'))  # Maximum amount for sell orders
        self.target_bid_value = float(os.getenv('TARGET_BID_VALUE', '250'))  # Target total USDT value for buy orders
        self.target_ask_value = float(os.getenv('TARGET_ASK_VALUE', '250'))  # Target total USDT value for sell orders
        self.depth_orders = []  # Track orders placed for depth maintenance
        
        # Configure exchanges based on name
        exchange_configs = {
            'lbank': {
                'api_key_env': 'LBANK_API_KEY',
                'secret_key_env': 'LBANK_SECRET_KEY',
                'symbol_env': 'LBANK_SYMBOL',
                'class': ccxt.lbank
            },
            'mexc': {
                'api_key_env': 'MEXC_API_KEY',
                'secret_key_env': 'MEXC_SECRET_KEY',
                'symbol_env': 'MEXC_SYMBOL',
                'class': ccxt.mexc,
                'options': {
                    'defaultType': 'spot',
                    'recvWindow': 60000,  # 60 seconds recvWindow for MEXC
                    'adjustForTimeDifference': True  # Auto-adjust for time difference
                }
            },
            'bingx': {
                'api_key_env': 'BINGX_API_KEY',
                'secret_key_env': 'BINGX_SECRET_KEY',
                'symbol_env': 'BINGX_SYMBOL',
                'class': ccxt.bingx
            }
        }
        
        # Check if exchange is supported
        if self.exchange_name not in exchange_configs:
            raise ValueError(f"Exchange {exchange_name} not supported")
        
        # Get config for the selected exchange
        config = exchange_configs[self.exchange_name]
        
        # Get API credentials
        api_key = os.getenv(config['api_key_env'])
        secret_key = os.getenv(config['secret_key_env'])
        
        if not api_key or not secret_key:
            raise ValueError(f"Missing API credentials for {exchange_name}")
        
        # Initialize exchange with options specific to each exchange
        exchange_options = {
            'apiKey': api_key,
            'secret': secret_key,
            'enableRateLimit': True,
            'options': config.get('options', {'defaultType': 'spot'})
        }
        
        # Special handling for MEXC to fix timestamp issue
        if self.exchange_name == 'mexc':
            # Create with time adjustment and large recvWindow
            exchange_options['options']['adjustForTimeDifference'] = True
            exchange_options['options']['recvWindow'] = 60000  # 5 minutes
            
            # Try to determine server time difference
            try:
                temp_exchange = config['class']({'enableRateLimit': True})
                server_time = temp_exchange.fetch_time()
                local_time = int(time.time() * 1000)
                time_diff = server_time - local_time
                
                print(f"MEXC server time: {server_time}")
                print(f"Local time: {local_time}")
                print(f"Time difference: {time_diff} ms")
                
                exchange_options['options']['timeDifference'] = time_diff
                
            except Exception as e:
                print(f"Warning: Could not determine time difference with MEXC server: {e}")
                print("Using default time synchronization mechanism")
            
        self.exchange = config['class'](exchange_options)
        
        # Additional setup for MEXC after creation
        if self.exchange_name == 'mexc':
            try:
                # Force a new time sync after exchange creation
                self.exchange.load_time_difference()
                time_diff = self.exchange.options.get('timeDifference', 0)
                print(f"MEXC time difference after sync: {time_diff} ms")
                
                # Set extremely large recvWindow
                self.exchange.options['recvWindow'] = 60000  # 1000 seconds
                
            except Exception as e:
                print(f"Warning: Could not synchronize time with MEXC: {e}")
                # As a last resort, use a fixed time offset
                self.exchange.options['timeDifference'] = 0
        
        # Set symbol
        self.symbol = symbol or os.getenv(config['symbol_env'])
        if not self.symbol:
            raise ValueError(f"Trading symbol not provided for {exchange_name}")
        
        # Validate that the symbol is supported on this exchange
        try:
            print(f"Validating that {self.symbol} is supported on {exchange_name}...")
            markets = self.exchange.load_markets()
            
            # Different exchanges format symbols differently
            # Try both formats: with slash (BTC/USDT) and with underscore (btc_usdt)
            symbol_slash = self.symbol.replace('_', '/').upper()
            symbol_underscore = self.symbol.lower().replace('/', '_')
            
            if symbol_slash not in markets and symbol_underscore not in markets:
                # Try the exchange's default symbol format
                if self.symbol not in markets:
                    supported_symbols = ', '.join(list(markets.keys())[:10])
                    raise ValueError(f"Symbol {self.symbol} is not supported on {exchange_name}. Some supported symbols: {supported_symbols}...")
        except Exception as e:
            print(f"Warning: Could not validate symbol: {str(e)}")
            print("Will attempt to continue, but operations may fail if the symbol is not supported")
            
        # Load trading parameters from .env
        self.order_count = int(os.getenv('ORDER_COUNT', '3'))
        self.min_amount = float(os.getenv('MIN_ORDER_AMOUNT', '10'))
        self.max_amount = float(os.getenv('MAX_ORDER_AMOUNT', '100'))
        self.price_range_percent = float(os.getenv('PRICE_RANGE_PERCENT', '2'))
        self.cycle_delay = int(os.getenv('CYCLE_DELAY', '60'))
        
        print(f"Initialized {exchange_name} volume bot for {self.symbol}")
        print(f"Order count: {self.order_count}, Amount range: {self.min_amount}-{self.max_amount}")
    
    def fetch_order_book(self):
        """Fetch order book for the symbol"""
        try:
            order_book = self.exchange.fetch_order_book(self.symbol)
            print(f"Fetched order book for {self.symbol} on {self.exchange_name}")
            print(f"Best bid: {order_book['bids'][0][0] if order_book['bids'] else 'None'}")
            print(f"Best ask: {order_book['asks'][0][0] if order_book['asks'] else 'None'}")
            return order_book
        except Exception as e:
            print(f"Error fetching order book: {str(e)}")
            return None
    
    def get_current_price(self):
        """Get current price for the symbol"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            print(f"Error fetching current price: {str(e)}")
            
            # If ticker fails, try to get price from order book
            order_book = self.fetch_order_book()
            if order_book and order_book['bids'] and order_book['asks']:
                mid_price = (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2
                print(f"Using mid price from order book: {mid_price}")
                return mid_price
            
            return None
    
    def get_orders_in_price_range(self, order_book, current_price, range_percent):
        """
        Filter orders from the order book based on price range
        
        Args:
            order_book (dict): Order book with bids and asks
            current_price (float): Current market price
            range_percent (float): Price range percentage (e.g. 2 for 2%)
        
        Returns:
            tuple: (filtered_bids, filtered_asks)
        """
        if not order_book or not current_price:
            return [], []
        
        # Calculate price range
        min_price = current_price * (1 - range_percent / 100)
        max_price = current_price * (1 + range_percent / 100)
        
        # Filter bids and asks within range
        filtered_bids = [bid for bid in order_book['bids'] if min_price <= bid[0] <= current_price]
        filtered_asks = [ask for ask in order_book['asks'] if current_price <= ask[0] <= max_price]
        
        print(f"Price range: {min_price:.8f} - {max_price:.8f}")
        print(f"Found {len(filtered_bids)} bids and {len(filtered_asks)} asks in range")
        
        return filtered_bids, filtered_asks

    def create_market_order(self, side, amount):
        """Create a market order to match with existing orders"""
        try:
            print(f"Placing {side} market order: {amount} {self.symbol}")
            
            # For LBank specifically, market buy orders require price parameter
            if self.exchange_name == 'lbank' and side == 'buy':
                # Get current price to calculate total cost
                current_price = self.get_current_price()
                if not current_price:
                    print("Cannot get current price for market buy order")
                    return None
                
                print(f"Using price {current_price} for LBank market buy order")
                
                # For LBank market buys, we need to include the price
                result = self.exchange.create_order(
                    symbol=self.symbol,
                    type='market',
                    side=side,
                    amount=amount,
                    price=current_price
                )
            else:
                # Normal market order for other exchanges or sell orders
                result = self.exchange.create_order(
                    symbol=self.symbol,
                    type='market',
                    side=side,
                    amount=amount
                )
            
            print(f"Market order executed: ID {result['id']}")
            return result
        except Exception as e:
            print(f"Error creating market order: {str(e)}")
            
            # Fallback to taker limit order if market order fails
            try:
                print(f"Attempting fallback to taker limit order...")
                
                # Get order book to determine price
                order_book = self.fetch_order_book()
                if not order_book:
                    print("Could not fetch order book for fallback limit order")
                    return None
                
                # Calculate aggressive price that will be filled immediately (taker)
                price = None
                if side == 'buy' and order_book['asks']:
                    # For buy, set price slightly higher than best ask
                    price = order_book['asks'][0][0] * 1.001  # 0.1% higher
                elif side == 'sell' and order_book['bids']:
                    # For sell, set price slightly lower than best bid
                    price = order_book['bids'][0][0] * 0.999  # 0.1% lower
                
                if not price:
                    print("Could not determine price for fallback limit order")
                    return None
                
                price = round(price, 8)
                print(f"Placing {side} limit order (taker): {amount} {self.symbol} @ {price}")
                
                # Create aggressive limit order that should be filled immediately
                result = self.exchange.create_order(
                    symbol=self.symbol,
                    type='limit',
                    side=side,
                    amount=amount,
                    price=price
                )
                
                print(f"Limit order executed: ID {result['id']}")
                return result
                
            except Exception as e2:
                print(f"Error creating fallback limit order: {str(e2)}")
                return None
            
            return None
    
    def create_matched_orders(self, order_book):
        """
        Create pairs of orders that will match with existing orders in the order book
        
        This function creates orders that will be immediately executed by matching
        with existing orders in the order book, generating actual trading volume.
        """
        results = []
        
        # Check if we have both bids and asks
        if not order_book or not order_book['bids'] or not order_book['asks']:
            print("Order book is empty or missing bids/asks")
            return results
        
        # Store initial price if not set
        if self.initial_price is None:
            self.initial_price = self.get_current_price()
            print(f"Setting initial price reference: {self.initial_price}")
        
        # Get current price to check range
        current_price = self.get_current_price()
        
        # Calculate price range limits
        min_price = self.initial_price * (1 - self.price_control_range / 100)
        max_price = self.initial_price * (1 + self.price_control_range / 100)
        
        print(f"Target price range: {min_price:.8f} - {max_price:.8f}")
        print(f"Current price: {current_price:.8f}")
        print(f"Price direction strategy: {self.price_direction}")
        
        # Adjust buy/sell ratio based on price direction strategy
        if self.price_direction == 'increase':
            # Favor buy orders to increase price
            self.buy_sell_ratio = 2.0  # 67% buy, 33% sell
            print("Using INCREASE strategy: Favoring BUY orders to push price up")
        elif self.price_direction == 'decrease':
            # Favor sell orders to decrease price
            self.buy_sell_ratio = 0.3  # 30% buy, 70% sell
            print("Using DECREASE strategy: Favoring SELL orders to push price down")
        else:  # 'maintain' or any other value
            # If we have a daily wave active, the buy_sell_ratio is already set in update_price_direction_from_wave()
            if not self.use_daily_wave:
                # Adjust buy/sell ratio based on current price position in the range
                if current_price > max_price:
                    # Price too high, favor sell orders
                    self.buy_sell_ratio = 0.3  # 30% buy, 70% sell
                    print("Price above target range. Favoring SELL orders to bring price down.")
                elif current_price < min_price:
                    # Price too low, favor buy orders
                    self.buy_sell_ratio = 2.0  # 67% buy, 33% sell
                    print("Price below target range. Favoring BUY orders to bring price up.")
                else:
                    # Price in range, balance orders
                    # Calculate position within range (0 = at min, 1 = at max)
                    range_position = (current_price - min_price) / (max_price - min_price)
                    # Adjust ratio from 1.5 (at min) to 0.5 (at max)
                    self.buy_sell_ratio = 1.5 - range_position
                    print(f"Price within target range. Setting buy/sell ratio to {self.buy_sell_ratio:.2f}")
        
        # Determine order counts based on ratio
        total_orders = self.order_count
        buy_orders = int(round(total_orders * (self.buy_sell_ratio / (1 + self.buy_sell_ratio))))
        sell_orders = total_orders - buy_orders
        
        print(f"Order distribution: {buy_orders} buy orders, {sell_orders} sell orders")
        
        # First place our own orders to match against
        own_orders_results = self.place_own_orders_to_match(buy_orders, sell_orders, current_price)
        results.extend(own_orders_results)
        
        # Wait for orders to be placed
        time.sleep(2)
        
        # Now match against our own orders
        for i in range(total_orders):
            try:
                # Attempt to match with our own orders
                matched_result = self.match_with_own_orders(i < buy_orders)
                if matched_result:
                    results.append(matched_result)
                    time.sleep(1)
            except Exception as e:
                print(f"Error matching orders: {str(e)}")
        
        return results
    
    def place_own_orders_to_match(self, buy_count, sell_count, current_price):
        """Place our own orders that we'll match against"""
        results = []
        
        # Place buy orders
        for i in range(buy_count):
            try:
                amount = self.create_random_amount()
                # Place slightly below current price
                price = round(current_price * (1 - 0.1 / 100), 8)  # 0.1% below
                
                order = self.exchange.create_order(
                    symbol=self.symbol,
                    type='limit',
                    side='buy',
                    amount=amount,
                    price=price
                )
                
                self.own_orders.append({
                    'id': order['id'],
                    'side': 'buy',
                    'price': price,
                    'amount': amount,
                    'matched': False
                })
                
                print(f"Placed own buy order: ID {order['id']} at {price}")
                results.append(order)
                time.sleep(0.5)
            except Exception as e:
                print(f"Error placing own buy order: {str(e)}")
        
        # Place sell orders
        for i in range(sell_count):
            try:
                amount = self.create_random_amount()
                # Place slightly above current price
                price = round(current_price * (1 + 0.1 / 100), 8)  # 0.1% above
                
                order = self.exchange.create_order(
                    symbol=self.symbol,
                    type='limit',
                    side='sell',
                    amount=amount,
                    price=price
                )
                
                self.own_orders.append({
                    'id': order['id'],
                    'side': 'sell',
                    'price': price,
                    'amount': amount,
                    'matched': False
                })
                
                print(f"Placed own sell order: ID {order['id']} at {price}")
                results.append(order)
                time.sleep(0.5)
            except Exception as e:
                print(f"Error placing own sell order: {str(e)}")
        
        return results
    
    def match_with_own_orders(self, is_buy):
        """Match with our own orders"""
        # Find unmatched orders of opposite side
        side_to_match = 'sell' if is_buy else 'buy'
        orders_to_match = [o for o in self.own_orders if o['side'] == side_to_match and not o['matched']]
        
        if not orders_to_match:
            print(f"No unmatched {side_to_match} orders to match against")
            return None
        
        # Take the first unmatched order
        order_to_match = orders_to_match[0]
        
        try:
            # Create opposite order to match
            side = 'buy' if side_to_match == 'sell' else 'sell'
            price = order_to_match['price']
            amount = order_to_match['amount']
            
            print(f"Matching with own {side_to_match} order: ID {order_to_match['id']} at {price}")
            
            # Create matching order
            result = self.exchange.create_order(
                symbol=self.symbol,
                type='limit',
                side=side,
                amount=amount,
                price=price
            )
            
            # Mark the matched order
            order_to_match['matched'] = True
            
            print(f"Created matching {side} order: ID {result['id']}")
            return result
        except Exception as e:
            print(f"Error matching with own order: {str(e)}")
            return None
    
    def create_self_matching_orders(self):
        """Create self-matching orders (buy and sell at same price)"""
        results = []
        
        # Get current price
        current_price = self.get_current_price()
        if not current_price:
            print("Cannot get current price for self-matching orders")
            return results
        
        # Store initial price if not set
        if self.initial_price is None:
            self.initial_price = current_price
            print(f"Setting initial price reference: {self.initial_price}")
        
        # Calculate price range limits
        min_price = self.initial_price * (1 - self.price_control_range / 100)
        max_price = self.initial_price * (1 + self.price_control_range / 100)
        
        print(f"Target price range: {min_price:.8f} - {max_price:.8f}")
        print(f"Current price: {current_price:.8f}")
        print(f"Price direction strategy: {self.price_direction}")
        
        # Adjust price based on price direction strategy
        if self.price_direction == 'increase':
            # Create orders above current price to push it up
            match_price = round(current_price * 1.003, 8)  # 0.3% above
            print(f"Using INCREASE strategy: Creating orders above current at {match_price}")
        elif self.price_direction == 'decrease':
            # Create orders below current price to push it down
            match_price = round(current_price * 0.997, 8)  # 0.3% below
            print(f"Using DECREASE strategy: Creating orders below current at {match_price}")
        else:  # 'maintain' or any other value
            # Keep original logic for maintaining price in range
            if current_price > max_price:
                # Price too high, create orders below current
                match_price = round(current_price * 0.997, 8)  # 0.3% below
                print(f"Price above target range. Creating orders below current at {match_price}")
            elif current_price < min_price:
                # Price too low, create orders above current
                match_price = round(current_price * 1.003, 8)  # 0.3% above
                print(f"Price below target range. Creating orders above current at {match_price}")
            else:
                # Price in range, create orders at mid range
                mid_range = (min_price + max_price) / 2
                # Bias slightly toward center of range
                if current_price > mid_range:
                    match_price = round(current_price * 0.999, 8)  # Slight bias down
                else:
                    match_price = round(current_price * 1.001, 8)  # Slight bias up
                
                print(f"Price within target range. Creating orders at {match_price}")
        
        for i in range(self.order_count):
            try:
                # Generate random amount for the pair (same amount for buy and sell)
                amount = self.create_random_amount()
                print(f"Using same amount ({amount}) for self-matching buy and sell orders")
                
                # First place a limit buy order
                buy_result = self.exchange.create_order(
                    symbol=self.symbol,
                    type='limit',
                    side='buy',
                    amount=amount,
                    price=match_price
                )
                
                print(f"Buy order placed: ID {buy_result['id']}")
                results.append(buy_result)
                
                # Small delay
                time.sleep(1)
                
                # Then place a matching limit sell order at the same price with the same amount
                sell_result = self.exchange.create_order(
                    symbol=self.symbol,
                    type='limit',
                    side='sell',
                    amount=amount,  # Using the exact same amount as the buy order
                    price=match_price
                )
                
                print(f"Sell order placed: ID {sell_result['id']}")
                results.append(sell_result)
                
                # Give time for orders to match
                time.sleep(3)
                
                # Check if orders have been filled
                try:
                    buy_order_status = self.exchange.fetch_order(buy_result['id'], self.symbol)
                    sell_order_status = self.exchange.fetch_order(sell_result['id'], self.symbol)
                    
                    print(f"Buy order status: {buy_order_status['status']}")
                    print(f"Sell order status: {sell_order_status['status']}")
                    
                    # Report on balance impact
                    print(f"Balance impact: {amount} {self.symbol} bought and sold (net zero)")
                    
                    # If orders are still open after waiting, cancel them
                    if buy_order_status['status'] == 'open':
                        print(f"Canceling unfilled buy order {buy_result['id']}")
                        self.exchange.cancel_order(buy_result['id'], self.symbol)
                    
                    if sell_order_status['status'] == 'open':
                        print(f"Canceling unfilled sell order {sell_result['id']}")
                        self.exchange.cancel_order(sell_result['id'], self.symbol)
                        
                except Exception as e:
                    print(f"Error checking order status: {str(e)}")
                
            except Exception as e:
                print(f"Error creating self-matching orders: {str(e)}")
        
        return results
    
    def create_initial_orders(self, current_price):
        """
        Place initial orders above and below the mid price to establish order book
        
        Args:
            current_price (float): Current market price
            
        Returns:
            list: List of created orders
        """
        results = []
        print("\nPlacing initial orders around mid price...")
        
        # Store initial price if not set
        if self.initial_price is None:
            self.initial_price = current_price
            print(f"Setting initial price reference: {self.initial_price}")
        
        # Number of initial orders to place on each side (buy/sell)
        initial_order_count = int(os.getenv('INITIAL_ORDER_COUNT', '3'))
        
        # Price spread for initial orders (percentage)
        price_spread = float(os.getenv('INITIAL_PRICE_SPREAD', '5'))
        
        # Ensure spread is within our control range
        price_spread = min(price_spread, self.price_control_range)
        
        # Calculate price range
        min_price = current_price * (1 - price_spread / 100)
        max_price = current_price * (1 + price_spread / 100)
        
        print(f"Creating {initial_order_count} buy and {initial_order_count} sell orders between {min_price:.8f} and {max_price:.8f}")
        print(f"Price direction strategy: {self.price_direction}")
        
        # Adjust the number of buy/sell orders based on price direction
        buy_order_count = initial_order_count
        sell_order_count = initial_order_count
        
        if self.price_direction == 'increase':
            # More buy orders than sell orders
            buy_order_count = int(initial_order_count * 1.5)
            sell_order_count = int(initial_order_count * 0.5)
            print(f"Using INCREASE strategy: Placing {buy_order_count} buy orders and {sell_order_count} sell orders")
        elif self.price_direction == 'decrease':
            # More sell orders than buy orders
            buy_order_count = int(initial_order_count * 0.5)
            sell_order_count = int(initial_order_count * 1.5)
            print(f"Using DECREASE strategy: Placing {buy_order_count} buy orders and {sell_order_count} sell orders")
        
        # Place buy orders below current price
        for i in range(buy_order_count):
            try:
                # Calculate price - evenly distribute across the range
                price_factor = 1 - ((i + 1) / (buy_order_count + 1)) * (price_spread / 100)
                price = round(current_price * price_factor, 8)
                
                # Generate random amount
                amount = self.create_random_amount()
                
                print(f"Placing buy limit order: {amount} {self.symbol} @ {price} ({price_factor*100:.2f}% of mid price)")
                
                # Create buy limit order
                buy_result = self.exchange.create_order(
                    symbol=self.symbol,
                    type='limit',
                    side='buy',
                    amount=amount,
                    price=price
                )
                
                # Track as our own order
                order_info = {
                    'id': buy_result['id'],
                    'side': 'buy',
                    'price': price,
                    'amount': amount,
                    'matched': False
                }
                self.own_orders.append(order_info)
                self.initial_orders.append(order_info)  # Also track as initial order
                
                print(f"Buy order placed: ID {buy_result['id']}")
                results.append(buy_result)
                
                # Small delay between orders
                time.sleep(1)
                
            except Exception as e:
                print(f"Error creating initial buy order: {str(e)}")
        
        # Place sell orders above current price
        for i in range(sell_order_count):
            try:
                # Calculate price - evenly distribute across the range
                price_factor = 1 + ((i + 1) / (sell_order_count + 1)) * (price_spread / 100)
                price = round(current_price * price_factor, 8)
                
                # Generate random amount
                amount = self.create_random_amount()
                
                print(f"Placing sell limit order: {amount} {self.symbol} @ {price} ({price_factor*100:.2f}% of mid price)")
                
                # Create sell limit order
                sell_result = self.exchange.create_order(
                    symbol=self.symbol,
                    type='limit',
                    side='sell',
                    amount=amount,
                    price=price
                )
                
                # Track as our own order
                order_info = {
                    'id': sell_result['id'],
                    'side': 'sell',
                    'price': price,
                    'amount': amount,
                    'matched': False
                }
                self.own_orders.append(order_info)
                self.initial_orders.append(order_info)  # Also track as initial order
                
                print(f"Sell order placed: ID {sell_result['id']}")
                results.append(sell_result)
                
                # Small delay between orders
                time.sleep(1)
                
            except Exception as e:
                print(f"Error creating initial sell order: {str(e)}")
        
        print(f"Placed {len(results)} initial orders around mid price")
        return results
    
    def cancel_all_initial_orders(self):
        """Cancel all initial orders that haven't been matched"""
        print("\nCanceling remaining initial orders...")
        
        canceled_count = 0
        remaining_initial_orders = [o for o in self.initial_orders if not o['matched']]
        
        if not remaining_initial_orders:
            print("No initial orders to cancel")
            return
        
        for order in remaining_initial_orders:
            try:
                print(f"Canceling {order['side']} order ID {order['id']} at price {order['price']}")
                self.exchange.cancel_order(order['id'], self.symbol)
                order['matched'] = True  # Mark as handled
                canceled_count += 1
                
                # Small delay between cancellations
                time.sleep(0.5)
            except Exception as e:
                print(f"Error canceling order {order['id']}: {str(e)}")
        
        print(f"Canceled {canceled_count} initial orders")
    
    def create_random_amount(self):
        """Generate random order amount between min and max"""
        return round(random.uniform(self.min_amount, self.max_amount), 8)
    
    def get_usdt_balance(self):
        """Get the current USDT balance"""
        try:
            balance = self.exchange.fetch_balance()
            
            # Different exchanges might format currency differently
            usdt_balance = 0
            
            # Try different possible USDT keys
            for currency in ['USDT', 'usdt']:
                if currency in balance:
                    usdt_balance = float(balance[currency].get('free', 0))
                    print(f"Current {currency} balance: {usdt_balance}")
                    return usdt_balance
            
            print("USDT balance not found")
            return 0
        except Exception as e:
            print(f"Error fetching balance: {str(e)}")
            return 0
    
    def cancel_recent_buy_orders(self, min_balance=100, current_balance=0):
        """
        Cancel recent buy orders until enough funds are freed to reach min_balance
        
        Args:
            min_balance: Minimum USDT balance threshold to aim for
            current_balance: Current USDT balance
            
        Returns:
            int: Number of orders canceled
        """
        try:
            print(f"\nFetching open orders to free up funds...")
            
            # Calculate how much we need to free up
            balance_deficit = min_balance - current_balance
            if balance_deficit <= 0:
                print(f"Current balance ({current_balance} USDT) already above minimum ({min_balance} USDT)")
                return 0
                
            print(f"Need to free up at least {balance_deficit:.2f} USDT to reach minimum balance")
            
            # Fetch all open orders
            open_orders = self.exchange.fetch_open_orders(self.symbol)
            
            # Filter to get only buy orders
            buy_orders = [order for order in open_orders if order['side'].lower() == 'buy']
            
            if not buy_orders:
                print("No buy orders found to cancel")
                return 0
            
            print(f"Found {len(buy_orders)} open buy orders")
            
            # Sort by timestamp (most recent first)
            buy_orders.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Start canceling orders until we've freed up enough or run out of orders
            canceled_count = 0
            freed_value = 0
            
            for i, order in enumerate(buy_orders):
                # Calculate order value (how much USDT it's using)
                order_price = float(order['price'])
                order_amount = float(order['amount'])
                order_value = order_price * order_amount
                
                try:
                    print(f"Canceling buy order {i+1}/{len(buy_orders)}: ID {order['id']} - {order_amount} @ {order_price} = {order_value:.2f} USDT")
                    self.exchange.cancel_order(order['id'], self.symbol)
                    canceled_count += 1
                    freed_value += order_value
                    
                    print(f"Freed up approximately {order_value:.2f} USDT (total: {freed_value:.2f} USDT)")
                    
                    # Check if we've freed up enough
                    if freed_value >= balance_deficit:
                        print(f"Freed up enough funds ({freed_value:.2f} USDT) to reach minimum balance")
                        break
                    
                    # Small delay between cancellations
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error canceling order {order['id']}: {str(e)}")
            
            print(f"Successfully canceled {canceled_count} buy orders, freeing approximately {freed_value:.2f} USDT")
            estimated_new_balance = current_balance + freed_value
            print(f"Estimated new balance: {estimated_new_balance:.2f} USDT (minimum required: {min_balance} USDT)")
            
            return canceled_count
            
        except Exception as e:
            print(f"Error canceling buy orders: {str(e)}")
            return 0
    
    def get_wave_target_percentage(self):
        """
        Calculate the target percentage change based on the wave pattern
        Returns a value between -wave_amplitude and +wave_amplitude
        """
        if not self.use_daily_wave:
            return 0.0
            
        now = datetime.now()
        
        # Calculate total hours including fractional part
        current_hour = now.hour
        current_minute = now.minute
        current_second = now.second
        hours_decimal = current_hour + (current_minute / 60.0) + (current_second / 3600.0)
        
        # Calculate position in the wave cycle (0 to 1 represents a full cycle)
        # Using modulo to wrap around after each cycle_hours period
        cycle_position = (hours_decimal % self.wave_cycle_hours) / self.wave_cycle_hours
        
        # Using a sine wave to model the price movement
        # We offset by -π/2 so we start at zero, peak at half cycle, and return to zero
        angle = (cycle_position * 2 * np.pi) - (np.pi / 2)
        wave_factor = np.sin(angle)
        
        # Scale by the amplitude to get the target percentage
        target_percentage = wave_factor * self.wave_amplitude
        
        current_time = now.strftime('%H:%M:%S')
        print(f"Wave at {current_time}: Cycle position {cycle_position:.2f}, Target: {target_percentage:.2f}%")
        
        return target_percentage
        
    def update_price_direction_from_wave(self):
        """Update price direction strategy based on the wave pattern"""
        if not self.use_daily_wave:
            return
            
        # Get current position in the wave
        target_percentage = self.get_wave_target_percentage()
        
        # Get current price to establish reference if needed
        current_price = self.get_current_price()
        now = datetime.now()
        
        # Initialize or update reference price every cycle_hours hours
        # or if it's not set yet
        if (self.wave_reference_price is None or 
            self.wave_reference_time is None or 
            (now - self.wave_reference_time).total_seconds() > self.wave_cycle_hours * 3600):
            
            self.wave_reference_price = current_price
            self.wave_reference_time = now
            print(f"Setting new wave reference price: {self.wave_reference_price} at {now.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Calculate target price based on reference and wave position
        target_price = self.wave_reference_price * (1 + (target_percentage / 100))
        
        # Calculate how far current price is from target
        price_difference_pct = ((current_price / target_price) - 1) * 100
        
        print(f"Wave analysis: Reference {self.wave_reference_price:.8f}, Target {target_price:.8f}, Current {current_price:.8f}")
        print(f"Price difference from target: {price_difference_pct:.2f}%")
        
        # Determine direction based on difference from target
        if price_difference_pct < -0.5:  # Current price is more than 0.5% below target
            self.price_direction = 'increase'
            print("Wave strategy: INCREASE price to reach target")
            # Set aggressive buy/sell ratio to push price up
            self.buy_sell_ratio = 3.0  # 75% buy, 25% sell
        elif price_difference_pct > 0.5:  # Current price is more than 0.5% above target
            self.price_direction = 'decrease'
            print("Wave strategy: DECREASE price to reach target")
            # Set aggressive buy/sell ratio to push price down
            self.buy_sell_ratio = 0.25  # 20% buy, 80% sell
        else:
            self.price_direction = 'maintain'
            print("Wave strategy: MAINTAIN price near target")
            # Set balanced ratio
            self.buy_sell_ratio = 1.0  # 50% buy, 50% sell

    def sell_base_assets_to_reach_balance(self, min_balance=100, current_balance=0):
        """
        Sell base assets (e.g., BTC in BTC/USDT) to reach minimum USDT balance
        
        Args:
            min_balance: Minimum USDT balance threshold to aim for
            current_balance: Current USDT balance
            
        Returns:
            float: Approximate USDT value obtained from selling
        """
        try:
            print(f"\nSelling base assets to free up funds...")
            
            # Calculate how much we need to free up
            balance_deficit = min_balance - current_balance
            if balance_deficit <= 0:
                print(f"Current balance ({current_balance} USDT) already above minimum ({min_balance} USDT)")
                return 0
            
            print(f"Need to obtain at least {balance_deficit:.2f} USDT")
            
            # Parse the trading symbol to get base asset
            base_asset = None
            quote_asset = None
            
            # Different exchanges format symbols differently, try to handle various formats
            if '/' in self.symbol:
                base_asset, quote_asset = self.symbol.split('/')
            elif '_' in self.symbol:
                base_asset, quote_asset = self.symbol.split('_')
            
            if not base_asset or not quote_asset:
                print(f"Unable to parse base and quote assets from symbol {self.symbol}")
                return 0
            
            # Normalize asset names
            base_asset = base_asset.upper()
            quote_asset = quote_asset.upper()
            
            if quote_asset != 'USDT':
                print(f"Quote asset is {quote_asset}, not USDT. Cannot sell to increase USDT balance.")
                return 0
            
            # Get available balance of base asset
            try:
                balance = self.exchange.fetch_balance()
                base_balance = float(balance.get(base_asset, {}).get('free', 0))
                
                print(f"Available {base_asset} balance: {base_balance}")
                
                if base_balance <= 0:
                    print(f"No {base_asset} available to sell")
                    return 0
                
                # Get current price to estimate value
                current_price = self.get_current_price()
                if not current_price:
                    print("Cannot get current price to estimate value")
                    return 0
                
                estimated_value = base_balance * current_price
                print(f"Estimated value of {base_balance} {base_asset}: {estimated_value:.2f} USDT")
                
                # Determine how much to sell
                amount_to_sell = min(base_balance, balance_deficit / current_price)
                if amount_to_sell < 0.0001:
                    print(f"Amount to sell ({amount_to_sell} {base_asset}) is too small")
                    return 0
                
                estimated_proceeds = amount_to_sell * current_price
                print(f"Planning to sell {amount_to_sell} {base_asset} (approx. {estimated_proceeds:.2f} USDT)")
                
                # Create market sell order
                result = self.exchange.create_order(
                    symbol=self.symbol,
                    type='market',
                    side='sell',
                    amount=amount_to_sell
                )
                
                print(f"Market sell order executed: ID {result['id']}")
                print(f"Sold {amount_to_sell} {base_asset} for approximately {estimated_proceeds:.2f} USDT")
                
                return estimated_proceeds
                
            except Exception as e:
                print(f"Error fetching balance or selling assets: {str(e)}")
                return 0
            
        except Exception as e:
            print(f"Error selling assets: {str(e)}")
            return 0

    def recover_balance(self, min_balance=100, current_balance=0):
        """
        Recover balance using the selected strategy: cancel orders or sell assets
        
        Args:
            min_balance: Minimum USDT balance threshold to aim for
            current_balance: Current USDT balance
            
        Returns:
            bool: True if balance recovery was attempted, False otherwise
        """
        try:
            balance_deficit = min_balance - current_balance
            if balance_deficit <= 0:
                print(f"Current balance ({current_balance} USDT) already above minimum ({min_balance} USDT)")
                return False
                
            print(f"\n⚠️ WARNING: USDT balance ({current_balance:.2f}) is below minimum threshold of {min_balance}")
            
            if self.balance_recovery_strategy == 'sell_assets':
                print("Using SELL ASSETS strategy to recover balance...")
                proceeds = self.sell_base_assets_to_reach_balance(min_balance, current_balance)
                
                if proceeds > 0:
                    print(f"Sold assets for approximately {proceeds:.2f} USDT. Waiting 10 seconds for balance to update...")
                    time.sleep(10)  # Wait longer for balance to update
                    
                    # Check balance again
                    new_balance = self.get_usdt_balance()
                    print(f"Updated USDT balance: {new_balance:.2f}")
                    
                    # If still below minimum, warn but continue
                    if new_balance < min_balance:
                        print(f"⚠️ Balance still below minimum threshold. Proceeding with caution.")
                    
                    return True
                else:
                    print("Failed to sell assets. Falling back to canceling orders...")
                    # Fall back to canceling orders
                    return self.cancel_orders_to_recover_balance(min_balance, current_balance)
            else:
                # Default strategy: cancel orders
                print("Using CANCEL ORDERS strategy to recover balance...")
                return self.cancel_orders_to_recover_balance(min_balance, current_balance)
                
        except Exception as e:
            print(f"Error recovering balance: {str(e)}")
            return False

    def cancel_orders_to_recover_balance(self, min_balance=100, current_balance=0):
        """
        Cancel orders to recover balance
        
        Args:
            min_balance: Minimum USDT balance threshold to aim for
            current_balance: Current USDT balance
            
        Returns:
            bool: True if orders were canceled, False otherwise
        """
        print("Canceling buy orders to free up enough funds...")
        canceled_count = self.cancel_recent_buy_orders(min_balance, current_balance)
        
        if canceled_count > 0:
            print(f"Canceled {canceled_count} buy orders. Waiting 10 seconds for balance to update...")
            time.sleep(10)  # Wait longer for balance to update
            
            # Check balance again
            new_balance = self.get_usdt_balance()
            print(f"Updated USDT balance: {new_balance:.2f}")
            
            # If still below minimum, warn but continue
            if new_balance < min_balance:
                print(f"⚠️ Balance still below minimum threshold. Proceeding with caution.")
            
            return True
        else:
            print("No buy orders were canceled. Proceeding with caution.")
            return False

    def maintain_order_book_depth(self, current_price):
        """
        Maintain a specific number of orders in the order book within defined depth
        
        Args:
            current_price (float): Current market price
            
        Returns:
            list: Newly created orders
        """
        if not self.maintain_order_book:
            return []
            
        print(f"\nMaintaining order book depth ({self.order_book_depth}% range)...")
        print(f"Target values: {self.target_bid_value} USDT in bids, {self.target_ask_value} USDT in asks")
        
        # Calculate price range
        min_price = current_price * (1 - self.order_book_depth / 100)
        max_price = current_price * (1 + self.order_book_depth / 100)
        
        print(f"Price range for depth maintenance: {min_price:.8f} - {max_price:.8f}")
        
        # Get existing open orders
        try:
            open_orders = self.exchange.fetch_open_orders(self.symbol)
            existing_bids = [o for o in open_orders if o['side'].lower() == 'buy' and float(o['price']) >= min_price and float(o['price']) <= current_price]
            existing_asks = [o for o in open_orders if o['side'].lower() == 'sell' and float(o['price']) >= current_price and float(o['price']) <= max_price]
            
            print(f"Found {len(existing_bids)} existing buy orders and {len(existing_asks)} existing sell orders in depth range")
            
            # Filter to get only orders placed by this bot for depth maintenance
            bot_depth_order_ids = [o['id'] for o in self.depth_orders]
            bot_bids = [o for o in existing_bids if o['id'] in bot_depth_order_ids]
            bot_asks = [o for o in existing_asks if o['id'] in bot_depth_order_ids]
            
            print(f"Of these, {len(bot_bids)} buy orders and {len(bot_asks)} sell orders were placed by this bot")
            
            # Calculate existing order values
            existing_bid_value = sum(float(o['price']) * float(o['amount']) for o in bot_bids)
            existing_ask_value = sum(current_price * float(o['amount']) for o in bot_asks)  # Using current price for consistency
            
            print(f"Current buy order value: {existing_bid_value:.2f} USDT")
            print(f"Current sell order value: {existing_ask_value:.2f} USDT")
            
            # Calculate how many new orders we need to place
            bids_needed = max(0, self.bid_orders_count - len(bot_bids))
            asks_needed = max(0, self.ask_orders_count - len(bot_asks))
            
            # Calculate remaining value needed
            remaining_bid_value = max(0, self.target_bid_value - existing_bid_value)
            remaining_ask_value = max(0, self.target_ask_value - existing_ask_value)
            
            # Update target values for new orders
            if bids_needed > 0:
                self.target_bid_value = remaining_bid_value
                print(f"Need to place {bids_needed} new buy orders worth {remaining_bid_value:.2f} USDT")
            
            if asks_needed > 0:
                self.target_ask_value = remaining_ask_value
                print(f"Need to place {asks_needed} new sell orders worth {remaining_ask_value:.2f} USDT")
            
            # Clean up old depth orders that are no longer open
            open_order_ids = [o['id'] for o in open_orders]
            self.depth_orders = [o for o in self.depth_orders if o['id'] in open_order_ids]
            
            results = []
            
            # Place new bid orders if needed
            if bids_needed > 0 and remaining_bid_value > 0:
                results.extend(self.place_depth_bids(bids_needed, min_price, current_price))
                
            # Place new ask orders if needed
            if asks_needed > 0 and remaining_ask_value > 0:
                results.extend(self.place_depth_asks(asks_needed, current_price, max_price))
                
            # Restore original target values for next cycle
            self.target_bid_value = float(os.getenv('TARGET_BID_VALUE', '150'))
            self.target_ask_value = float(os.getenv('TARGET_ASK_VALUE', '150'))
            
            print(f"Placed {len(results)} new orders for depth maintenance")
            return results
            
        except Exception as e:
            print(f"Error maintaining order book depth: {str(e)}")
            return []
    
    def place_depth_bids(self, count, min_price, current_price):
        """
        Place buy orders distributed across the bid range with a specific total USDT value
        
        Args:
            count: Number of orders to place
            min_price: Minimum price in the range
            current_price: Current market price
            
        Returns:
            list: Created orders
        """
        results = []
        
        if count <= 0:
            return results
            
        # Evenly distribute prices across the range
        price_range = current_price - min_price
        price_step = price_range / (count + 1)
        
        # Target USDT value per order (divide total target by number of orders)
        target_value_per_order = self.target_bid_value / count
        print(f"Target USDT value per buy order: {target_value_per_order:.2f} USDT")
        
        total_value_placed = 0
        
        for i in range(count):
            try:
                # Calculate price - evenly distribute across the range
                price = round(current_price - ((i + 1) * price_step), 8)
                
                # Calculate amount based on target value and price
                # Amount = USDT value / price
                base_amount = target_value_per_order / price
                
                # Add some randomness but ensure we stay close to target value
                # Random factor between 0.9 and 1.1 (±10%)
                random_factor = random.uniform(0.9, 1.1)
                amount = round(base_amount * random_factor, 8)
                
                # Ensure amount is within min/max bounds
                amount = max(min(amount, self.max_bid_amount), self.min_bid_amount)
                
                # Calculate the actual USDT value of this order
                usdt_value = amount * price
                total_value_placed += usdt_value
                
                print(f"Placing depth buy order: {amount} {self.symbol} @ {price} = {usdt_value:.2f} USDT ({(price/current_price*100):.2f}% of current price)")
                
                # Create buy limit order
                result = self.exchange.create_order(
                    symbol=self.symbol,
                    type='limit',
                    side='buy',
                    amount=amount,
                    price=price
                )
                
                # Track as depth maintenance order
                order_info = {
                    'id': result['id'],
                    'side': 'buy',
                    'price': price,
                    'amount': amount,
                    'timestamp': int(time.time() * 1000),
                    'value': usdt_value
                }
                self.depth_orders.append(order_info)
                
                print(f"Buy depth order placed: ID {result['id']}")
                results.append(result)
                
                # Small delay between orders
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error creating depth buy order: {str(e)}")
        
        print(f"Total buy order value placed: {total_value_placed:.2f} USDT (target: {self.target_bid_value:.2f} USDT)")
        return results
    
    def place_depth_asks(self, count, current_price, max_price):
        """
        Place sell orders distributed across the ask range with a specific total USDT value
        
        Args:
            count: Number of orders to place
            current_price: Current market price
            max_price: Maximum price in the range
            
        Returns:
            list: Created orders
        """
        results = []
        
        if count <= 0:
            return results
            
        # Evenly distribute prices across the range
        price_range = max_price - current_price
        price_step = price_range / (count + 1)
        
        # Target USDT value per order (divide total target by number of orders)
        target_value_per_order = self.target_ask_value / count
        print(f"Target USDT value per sell order: {target_value_per_order:.2f} USDT")
        
        total_value_placed = 0
        
        for i in range(count):
            try:
                # Calculate price - evenly distribute across the range
                price = round(current_price + ((i + 1) * price_step), 8)
                
                # Calculate amount based on target value and price
                # For sell orders, we calculate using current_price to determine USDT equivalent
                base_amount = target_value_per_order / current_price
                
                # Add some randomness but ensure we stay close to target value
                # Random factor between 0.9 and 1.1 (±10%)
                random_factor = random.uniform(0.9, 1.1)
                amount = round(base_amount * random_factor, 8)
                
                # Ensure amount is within min/max bounds
                amount = max(min(amount, self.max_ask_amount), self.min_ask_amount)
                
                # Calculate the actual USDT value of this order
                usdt_value = amount * current_price  # Use current price for value calculation
                total_value_placed += usdt_value
                
                print(f"Placing depth sell order: {amount} {self.symbol} @ {price} = {usdt_value:.2f} USDT ({(price/current_price*100):.2f}% of current price)")
                
                # Create sell limit order
                result = self.exchange.create_order(
                    symbol=self.symbol,
                    type='limit',
                    side='sell',
                    amount=amount,
                    price=price
                )
                
                # Track as depth maintenance order
                order_info = {
                    'id': result['id'],
                    'side': 'sell',
                    'price': price,
                    'amount': amount,
                    'timestamp': int(time.time() * 1000),
                    'value': usdt_value
                }
                self.depth_orders.append(order_info)
                
                print(f"Sell depth order placed: ID {result['id']}")
                results.append(result)
                
                # Small delay between orders
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error creating depth sell order: {str(e)}")
        
        print(f"Total sell order value placed: {total_value_placed:.2f} USDT (target: {self.target_ask_value:.2f} USDT)")
        return results
    
    def run_cycle(self):
        """Run a single cycle of the volume bot"""
        print(f"\n--- Starting cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        
        # Check USDT balance first
        usdt_balance = self.get_usdt_balance()
        min_balance_threshold = float(os.getenv('MIN_USDT_BALANCE', '100'))
        
        if usdt_balance < min_balance_threshold:
            # Use the selected balance recovery strategy
            self.recover_balance(min_balance=min_balance_threshold, current_balance=usdt_balance)
        
        # Fetch order book
        order_book = self.fetch_order_book()
        if not order_book:
            print("Failed to fetch order book, skipping cycle")
            return
        
        # Get current price
        current_price = self.get_current_price()
        if not current_price:
            print("Failed to get current price, skipping cycle")
            return
        
        print(f"Current price for {self.symbol}: {current_price}")
        
        # Update price direction strategy based on daily wave pattern if enabled
        if self.use_daily_wave:
            print("\nApplying daily wave price pattern strategy...")
            self.update_price_direction_from_wave()
        
        # Maintain order book depth if enabled
        if self.maintain_order_book:
            print("\nMaintaining order book depth...")
            depth_results = self.maintain_order_book_depth(current_price)
            if depth_results:
                print(f"Order book depth maintenance complete. {len(depth_results)} orders placed.")
        
        # Phase 1: Place initial orders around mid price (on every cycle)
        print("\nPhase 1: Setting up initial orders around mid price...")
        initial_results = self.create_initial_orders(current_price)
        if initial_results:
            print(f"Initial order setup complete. {len(initial_results)} orders placed.")
            
            # Wait a bit before continuing with volume trading
            wait_time = int(os.getenv('WAIT_AFTER_INIT', '20'))
            print(f"Waiting {wait_time} seconds before continuing with volume trading...")
            time.sleep(wait_time)
            
            # Refresh the order book
            order_book = self.fetch_order_book()
            current_price = self.get_current_price()
        else:
            print("Failed to set up initial orders, proceeding with volume trading")
        
        # Phase 2: Match against existing orders in the order book
        print("\nPhase 2: Matching against existing orders...")
        matched_results = self.create_matched_orders(order_book)
        
        # Wait a bit before next strategy
        if matched_results:
            print(f"Executed {len(matched_results)} market orders")
            print("Waiting 5 seconds before next strategy...")
            time.sleep(5)
        
        # Update order book after market orders
        updated_order_book = self.fetch_order_book()
        
        # Phase 3: Create self-matching orders
        print("\nPhase 3: Creating self-matching orders...")
        self_match_results = self.create_self_matching_orders()
        
        if self_match_results:
            print(f"Created {len(self_match_results)} self-matching orders")
        
        # Phase 4: Cancel initial orders if configured to do so
        if self.cancel_initial_orders:
            print("\nPhase 4: Canceling initial orders...")
            self.cancel_all_initial_orders()
        else:
            print("\nLeaving initial orders active as configured.")
        
        print(f"--- Cycle completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    
    def run(self, cycles=None):
        """
        Run volume bot for specified number of cycles
        
        Args:
            cycles (int): Number of cycles to run, None for infinite
        """
        cycle_count = 0
        
        try:
            print(f"Starting volume bot for {self.exchange_name} ({self.symbol})")
            print("This bot will execute REAL trades to generate volume")
            print("Press Ctrl+C to stop at any time")
            
            while cycles is None or cycle_count < cycles:
                self.run_cycle()
                cycle_count += 1
                
                if cycles is None or cycle_count < cycles:
                    print(f"Waiting {self.cycle_delay} seconds until next cycle...")
                    time.sleep(self.cycle_delay)
            
            print(f"Volume bot completed {cycle_count} cycles")
            
        except KeyboardInterrupt:
            print("\nBot stopped by user")
        except Exception as e:
            print(f"Error running bot: {str(e)}")

    def get_supported_symbols(self, limit=10):
        """Get a list of supported symbols on the exchange"""
        try:
            markets = self.exchange.load_markets()
            symbols = list(markets.keys())
            return symbols[:limit]
        except Exception as e:
            print(f"Error fetching supported symbols: {str(e)}")
            return []

def main():
    """Main function to run the volume bot"""
    parser = argparse.ArgumentParser(description='Volume trading bot for multiple exchanges')
    
    parser.add_argument('-e', '--exchange', type=str, required=True,
                      choices=['lbank', 'mexc', 'bingx', 'all'],
                      help='Exchange to use (lbank, mexc, bingx, or all)')
    
    parser.add_argument('-s', '--symbol', type=str,
                      help='Trading symbol (overrides .env)')
    
    parser.add_argument('-c', '--cycles', type=int, default=1,
                      help='Number of cycles to run (default: 1, 0 for infinite)')
    
    parser.add_argument('-l', '--limit-only', action='store_true',
                      help='Use only limit orders (no market orders)')
    
    parser.add_argument('--list-symbols', action='store_true',
                      help='List supported symbols on the specified exchange and exit')
    
    parser.add_argument('--cancel-initial', action='store_true',
                      help='Cancel initial orders at the end of each cycle (overrides .env setting)')
    
    parser.add_argument('-d', '--direction', type=str, choices=['maintain', 'increase', 'decrease', 'wave'],
                      help='Price direction strategy (overrides .env setting)')
    
    parser.add_argument('--wave-amplitude', type=float,
                      help='Amplitude of the daily price wave in percent (default: 3%)')
    
    parser.add_argument('--wave-cycle', type=float,
                      help='Wave cycle duration in hours (default: 6 hours - 3 up, 3 down)')
    
    parser.add_argument('--balance-strategy', type=str, choices=['cancel_orders', 'sell_assets'],
                      help='Strategy to use when balance is below minimum (cancel_orders or sell_assets)')
    
    parser.add_argument('--maintain-depth', action='store_true',
                      help='Maintain order book depth according to settings')
    
    parser.add_argument('--depth-range', type=float,
                      help='Order book depth range in percentage (default: 2%)')
    
    parser.add_argument('--bid-count', type=int,
                      help='Number of buy orders to maintain in the order book')
    
    parser.add_argument('--ask-count', type=int, 
                      help='Number of sell orders to maintain in the order book')
    
    parser.add_argument('--bid-value', type=float,
                      help='Target total USDT value for buy orders in the order book')
    
    parser.add_argument('--ask-value', type=float,
                      help='Target total USDT value for sell orders in the order book')
    
    args = parser.parse_args()
    
    # If user requested to list symbols
    if args.list_symbols:
        # Can't list symbols for 'all' exchanges at once
        if args.exchange == 'all':
            print("Cannot list symbols for 'all' exchanges at once. Please specify a single exchange.")
            return
        
        try:
            # Create a temporary bot instance just to get symbols
            print(f"\nListing supported symbols for {args.exchange}...")
            
            # Get exchange class from config
            exchange_configs = {
                'lbank': ccxt.lbank,
                'mexc': ccxt.mexc,
                'bingx': ccxt.bingx
            }
            
            exchange_class = exchange_configs[args.exchange]
            exchange_instance = exchange_class({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            
            # Load markets and get symbols
            markets = exchange_instance.load_markets()
            symbols = list(markets.keys())
            
            print(f"\nFound {len(symbols)} supported symbols on {args.exchange}:")
            
            # Group by quote currency (USDT, BTC, etc.)
            grouped_symbols = {}
            for symbol in symbols:
                # Most exchanges use / as separator
                if '/' in symbol:
                    base, quote = symbol.split('/')
                    if quote not in grouped_symbols:
                        grouped_symbols[quote] = []
                    grouped_symbols[quote].append(symbol)
            
            # Print grouped by quote currency
            for quote in sorted(grouped_symbols.keys()):
                if len(grouped_symbols[quote]) > 0:
                    print(f"\n{quote} pairs ({len(grouped_symbols[quote])}):")
                    # Print 10 examples for each quote currency
                    for symbol in sorted(grouped_symbols[quote])[:10]:
                        print(f"  {symbol}")
                    if len(grouped_symbols[quote]) > 10:
                        print(f"  ... and {len(grouped_symbols[quote]) - 10} more")
            
            # Also show symbols in the exchange's format
            print("\nNote: When configuring in .env, use the exchange's native format:")
            if args.exchange == 'lbank':
                print("  For LBank: Use lowercase with underscore (e.g., btc_usdt)")
            elif args.exchange == 'bingx':
                print("  For BingX: Use lowercase with underscore (e.g., btc_usdt)")
            elif args.exchange == 'mexc':
                print("  For MEXC: Use uppercase with underscore (e.g., BTC_USDT)")
            
            # Save to text file
            filename = f"{args.exchange}_symbols.txt"
            print(f"\nSaving complete symbol list to {filename}...")
            
            with open(filename, 'w') as f:
                # Write header
                f.write(f"Supported trading pairs on {args.exchange.upper()} ({len(symbols)} total)\n")
                f.write("=" * 60 + "\n\n")
                
                # Write grouped by quote currency
                for quote in sorted(grouped_symbols.keys()):
                    if len(grouped_symbols[quote]) > 0:
                        f.write(f"{quote} pairs ({len(grouped_symbols[quote])}):\n")
                        # Write all symbols for this quote currency
                        for symbol in sorted(grouped_symbols[quote]):
                            # Convert to exchange's native format
                            if args.exchange == 'lbank' or args.exchange == 'bingx':
                                native_format = symbol.replace('/', '_').lower()
                            elif args.exchange == 'mexc':
                                native_format = symbol.replace('/', '_').upper()
                            else:
                                native_format = symbol
                            
                            f.write(f"  {symbol}  (native format: {native_format})\n")
                        f.write("\n")
                        
                # Write footer with usage instructions
                f.write("\nUsage instructions:\n")
                f.write("-----------------\n")
                f.write("When configuring in .env, use the exchange's native format:\n")
                if args.exchange == 'lbank':
                    f.write("For LBank: Use lowercase with underscore (e.g., btc_usdt)\n")
                elif args.exchange == 'bingx':
                    f.write("For BingX: Use lowercase with underscore (e.g., btc_usdt)\n")
                elif args.exchange == 'mexc':
                    f.write("For MEXC: Use uppercase with underscore (e.g., BTC_USDT)\n")
            
            print(f"Symbol list saved to {filename}")
            
            # Create a second file with just the native format symbols, one per line
            native_filename = f"{args.exchange}_native_symbols.txt"
            print(f"Saving native format symbol list to {native_filename}...")
            
            with open(native_filename, 'w') as f:
                # Write all symbols in exchange's native format, one per line
                for symbol in sorted(symbols):
                    if '/' in symbol:
                        if args.exchange == 'lbank' or args.exchange == 'bingx':
                            native_format = symbol.replace('/', '_').lower()
                        elif args.exchange == 'mexc':
                            native_format = symbol.replace('/', '_').upper()
                        else:
                            native_format = symbol
                        
                        f.write(f"{native_format}\n")
            
            print(f"Native format symbol list saved to {native_filename}")
            return
        except Exception as e:
            print(f"Error listing symbols: {str(e)}")
            return
    
    # Convert 0 cycles to None for infinite running
    cycles = None if args.cycles == 0 else args.cycles
    
    # Whether to use market orders
    use_market_orders = not args.limit_only
    
    if args.exchange == 'all':
        # Run for all exchanges
        exchanges = ['lbank', 'mexc', 'bingx']
        
        for exchange in exchanges:
            try:
                bot = VolumeBot(exchange, args.symbol, use_market_orders)
                print(f"\nRunning bot for {exchange}...")
                bot.run(cycles)
            except Exception as e:
                print(f"Error with {exchange}: {str(e)}")
    else:
        # Run for single exchange
        bot = VolumeBot(args.exchange, args.symbol, use_market_orders)
        
        # Override env settings with command line arguments if provided
        if args.cancel_initial:
            bot.cancel_initial_orders = True
            print("Command-line override: Will cancel initial orders at end of cycle")
        
        if args.direction:
            if args.direction == 'wave':
                bot.use_daily_wave = True
                print("Command-line override: Using wave pattern for price movement")
                if args.wave_amplitude:
                    bot.wave_amplitude = args.wave_amplitude
                    print(f"Command-line override: Setting wave amplitude to {bot.wave_amplitude}%")
                if args.wave_cycle:
                    bot.wave_cycle_hours = args.wave_cycle
                    print(f"Command-line override: Setting wave cycle to {bot.wave_cycle_hours} hours")
            else:
                bot.price_direction = args.direction.lower()
                bot.use_daily_wave = False
                print(f"Command-line override: Using price direction strategy: {bot.price_direction}")
        
        if args.balance_strategy:
            bot.balance_recovery_strategy = args.balance_strategy
            print(f"Command-line override: Using {bot.balance_recovery_strategy} strategy for balance recovery")
        
        if args.maintain_depth:
            bot.maintain_order_book = True
            print("Command-line override: Maintaining order book depth")
            
            if args.depth_range:
                bot.order_book_depth = args.depth_range
                print(f"Command-line override: Setting depth range to {bot.order_book_depth}%")
                
            if args.bid_count:
                bot.bid_orders_count = args.bid_count
                print(f"Command-line override: Maintaining {bot.bid_orders_count} buy orders in depth")
                
            if args.ask_count:
                bot.ask_orders_count = args.ask_count
                print(f"Command-line override: Maintaining {bot.ask_orders_count} sell orders in depth")
            
            if args.bid_value:
                bot.target_bid_value = args.bid_value
                print(f"Command-line override: Setting target buy order value to {bot.target_bid_value} USDT")
            
            if args.ask_value:
                bot.target_ask_value = args.ask_value
                print(f"Command-line override: Setting target sell order value to {bot.target_ask_value} USDT")
        
        bot.run(cycles)

if __name__ == "__main__":
    main() 
