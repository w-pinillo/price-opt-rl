Updated TODOS:

   1. Refactor the evaluation script (`evaluate_multi_product_agent.py`) to implement the "Two-Stream" data approach.
       * Modify src/utils.py's make_multi_product_env to accept raw_data_df. (in_progress)
       * Modify src/envs/price_env.py's PriceEnv.__init__ to store raw_data_df and historical_avg_prices (for fixed cost). (pending)
       * Modify src/envs/price_env.py's PriceEnv.step to use self.raw_data_df for reward calculation and action translation, calculating
         actual sales based on real prices. (pending)
       * Update evaluate_multi_product_agent.py to load top100_daily.parquet as raw_data_df and pass it to make_multi_product_env. (pending)
       * Calculate Historical_Avg_Price (median of prices in the entire raw dataset) for each product from raw_data_df and pass it to
         PriceEnv for fixed cost calculation. (pending)

   2. Implement the "Do-Nothing" and "Rule-Based Heuristic" baselines within `evaluate_multi_product_agent.py`.
       * Calculate Median Historical Price for "Do-Nothing" baseline for each product from raw_data_df. (pending)
       * Implement Trend-Based Heuristic logic (e.g., using 7-day and 30-day moving averages of avg_price to adjust pricing) to generate a
         baseline price sequence. (pending)

   3. Expand evaluation metrics to include Gross Profit and Price Volatility.
       * Calculate Estimated Gross Profit for both agent and baselines using Cost_Fixed = Historical_Avg_Price * 0.70. (pending)
       * Calculate Price Volatility (StdDev(Price_t - Price_{t-1})) for both agent and baselines. (pending)

   4. Incorporate segmented analysis of results based on product velocity and volatility.
       * Categorize products (e.g., High/Low Sales Velocity, High/Low Price Volatility). (pending)
       * Summarize agent and baseline performance within these segments. (pending)

   5. Generate a scatter plot of Agent Improvement % (e.g., Gross Profit Improvement) vs. Historical Sales Volatility.
       * Create and save the visualization as part of the evaluation output. (pending)