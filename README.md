# Ibb_Iklim_ve_Mobilite_Datathonu_CyberBros
## Istanbul Public Transport Data Analysis

## About the Competition

In this competition, we were awarded 2nd place for our innovative solutions to address the existing problems in Istanbul using data from the Istanbul Municipality's Open Data Portal. The data we utilized is sourced from [Istanbul Municipality's Open Data Portal](https://data.ibb.gov.tr/dataset/hourly-public-transport-data-set/resource/f0c798c8-bab4-479e-841b-e82422e38e7f).

## Data Preprocessing
We started with hourly data for public transportation lines in Istanbul, covering various variables such as "number_of_passage" and "number_of_passengers" on a daily basis. Our goal was to transform this data into a monthly format for each line. We applied this process to data for the years 2021 and 2022, enabling us to compile monthly data for a two-year period.

Based on this data, we introduced a new column called 'intensity'. The 'intensity' column calculates the monthly intensity of a line, which is derived from the ratio of the monthly number of trips to the number of passengers for that line. Additionally, we categorized the CO2 consumption based on the number of trips.

## Objectives
Our project has several key objectives:

- Improve bus services for cost-efficiency and resource optimization.
- Enhance public transportation services on busy routes to ease urban transit.
- Optimize services on less busy routes to maximize resource allocation.
- Predict the intensity of newly planned routes.
- Mitigate CO2 emissions to reduce environmental impact and combat climate change.

## Model Description
Our model predicts the intensity of a public transportation line for the following month, providing valuable insights into its performance and resource allocation.

## Team Members
- [Serhat Kılıç](https://github.com/s192275) - Team Member
- [Umutcan Mert](https://github.com/UmutcanMert) - Team Member
- [Muhammed Nihat Aydın](https://github.com/Nihat-AYDIN) - Team Member
- [Muhammet Hamza Yavuz](https://github.com/hamza37yavuz) - Team Member
  
We are proud to contribute to the improvement of Istanbul's public transportation system and environmental sustainability through data analysis and innovative solutions.
