# Week 2 Data Cleaning & EDA Assessment

## Dataset
Google Play Store Apps (Kaggle)

## Task 1: Data Understanding
Identified issues:
1. Rating has missing values
2. Reviews stored as object instead of numeric
3. Installs contains symbols like '+' and ','
4. Price includes '$'
5. Size has mixed units (MB, KB, text)
6. Some columns are irrelevant for analysis

Most dangerous column: Size  
Column never for ML: App  
Cleanest column: Category

## Task 2: Missing Value Strategy
- Rating: Filled with median
- Content Rating: Filled with "Unknown"
- Current Ver: Dropped
- Indicator column added for Rating

Blindly dropping missing values would bias against new or less popular apps.

## Task 4: Data Type Fixing
Wrong data types break sorting, filtering, and ML models.
Numeric conversions were applied to Installs, Price, and Size.
