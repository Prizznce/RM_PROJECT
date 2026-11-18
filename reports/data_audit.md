# Data Stream Audit Report: Multi-Fuel Hourly Electricity Stream (2015-2018)

- **Data File**: `data/energy_dataset_clean.csv`
- **Row Count**: 35064 records (100% hourly intervals)
- **Date Range**: 2015-01-01 00:00:00 to 2018-12-31 23:00:00
- **Total Duration**: 35064 expected chronological hours

## 1. Temporal Continuity, Duplicates & Missing Hours
- **Duplicate Timestamps**: 4
- **Duplicate Full Rows**: 0
- **Missing Hourly Timestamps**: 4
- **Gaps Detected**:
  - 2015-03-29 01:00:00 -> 2015-03-29 03:00:00 (1 missing hours)
  - 2016-03-27 01:00:00 -> 2016-03-27 03:00:00 (1 missing hours)
  - 2017-03-26 01:00:00 -> 2017-03-26 03:00:00 (1 missing hours)
  - 2018-03-25 01:00:00 -> 2018-03-25 03:00:00 (1 missing hours)
## 2. Generation & Demand Metrics Summary
| Feature | Count | Mean (MW) | Std (MW) | Min (MW) | 25% | 50% (Median) | 75% | Max (MW) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Demand (MW)** | 35064 | 28698.28 | 4575.83 | 18041.00 | 24807.00 | 28902.00 | 32194.25 | 41015.00 |
| **Total Generation (MW)** | 35064 | 28163.50 | 4368.66 | 6899.00 | 24768.00 | 27914.00 | 31272.00 | 42310.00 |
| **Nuclear (MW)** | 35064 | 6263.48 | 840.27 | 0.00 | 5759.00 | 6564.00 | 7025.00 | 7117.00 |
| **Gas (MW)** | 35064 | 5622.70 | 2201.51 | 0.00 | 4126.00 | 4969.50 | 6429.00 | 20034.00 |
| **Hard Coal (MW)** | 35064 | 4256.53 | 1961.99 | 0.00 | 2527.00 | 4475.00 | 5839.00 | 8359.00 |
| **Solar (MW)** | 35064 | 1432.82 | 1679.96 | 0.00 | 71.00 | 616.00 | 2579.00 | 5792.00 |
| **Wind (MW)** | 35064 | 5464.98 | 3213.59 | 0.00 | 2933.00 | 4849.50 | 7399.50 | 17436.00 |

