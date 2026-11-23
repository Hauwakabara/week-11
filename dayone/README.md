# week-11
Missing Data Approach and Risks
For this lab, I filled missing values as follows:
continent: filled from the same country in other years
lifeExp and gdpPercap: filled with the median value for that continent
Risks of this approach:
1- Median imputation hides real variation - every missing country in a continent gets the exact same value, so the data looks less varied than it really is.
2- It can make relationships look stronger than they are - because imputed values sit in the middle, correlations can be misleading.
3- Continent median may not fit a specific country - a poor country and a rich country in the same continent are very different, but they get the same median.
4- Missingness may not be random - if poorer countries are more likely to be missing data, using the median will overestimate them.
For this lab, this approach is acceptable because only 17 values were missing (out of 96 rows), and the missingness flags (lifeExp_was_missing, gdpPercap_was_missing) were kept so we can tell which values were imputed.
