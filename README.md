## Souk-Sense A machine learning project that estimates the fair market price of a used car and flags whether a listing is overpriced, underpriced, or fairly priced — tuned for the Lebanese resale market. It factors in brand, model, age, mileage, engine size, fuel type, transmission, and running costs to output a fair-price estimate and a pricing verdict through a Streamlit dashboard.

🚗 Souk Sense — Used Car Overpricing Detector A machine learning project that predicts what a used car should cost, and tells you how far off a real listing is from that number — scaled for the Lebanese market, where import duties and VAT roughly double sticker prices from source markets.

### Overview

Used car listings are noisy. Two nearly identical cars can be priced thousands of dollars apart depending on the seller, the platform, or how badly someone needs to sell. This project puts a number on what "fair" actually looks like, so a listing can be judged against the market instead of against a gut feeling.

### The model outputs two things:

Estimated Fair Price — a dollar figure predicted from the car's specs
Verdict — whether the listing is Overpriced, Underpriced, or Fairly Priced, based on how far the asking price deviates from that estimate
How It Works

Rather than feeding raw listing fields straight into a model, the pipeline does a bit of groundwork first:

Model Frequency Encoding — rare vs. common models are weighted by how often they appear in the market, instead of one-hot exploding the feature space
Category Preprocessing — brand, fuel type, and transmission are encoded through a fitted preprocessor so the same transformation is guaranteed at inference time
Lebanon Price Scaling — predictions are scaled by an import duty + VAT multiplier (~1.75×) to approximate local market pricing from UK-sourced training data
Deviation Banding — the gap between listed price and predicted fair price is converted into a percentage and bucketed against a tuned threshold to produce the final verdict
Task	Output	Score
Price Regression	Estimated Fair Price (USD)	R² ≈ 0.96
Verdict	Overpriced / Fair / Underpriced	Threshold-based on regression residuals
Features Used

Brand · model (frequency-encoded) · year · mileage · engineSize · fuelType · transmission · tax · mpg

 ### Web App

Built with Streamlit — enter a car's specs and its listing price, and get an instant verdict with a color-coded result card and a deviation gauge showing exactly where the listing lands between underpriced and overpriced.

🟢 Underpriced — listed well below the model's fair-price estimate
🟡 Fairly Priced — within the tolerance band around fair value
🔴 Overpriced — listed well above the model's fair-price estimate
Stack

Python · Pandas · Scikit-learn · Joblib Streamlit

Dataset

Real UK used-car listings across Audi, BMW, Ford, Hyundai, Mercedes, Skoda, Toyota, Vauxhall, and Volkswagen, price-scaled to approximate Lebanese market pricing.

Limitations
The 1.75× scaling multiplier is a flat approximation — real Lebanese import costs vary by engine size, vehicle age, and customs bracket
Trained on UK listings, so it doesn't capture Lebanon-specific factors like grey-market imports, accident history, or local demand quirks
The over/underpriced threshold is a tuned constant, not learned per brand or segment

Built as an end-to-end ML project — from raw listings and feature engineering to a deployed pricing dashboard.
