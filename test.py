import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

data_filepath = 'SARIMA_AIC_for_all_pdq_and_seasonal_pdq.csv'
data_aic = pd.read_csv(data_filepath)
five_lowest_AICS = data_aic.sort_values(by='AIC', ascending=True, ignore_index=True).head()
print(f"="*99,"\n",
      f"top 5 combinations with lowest AIC\n"
      f"{five_lowest_AICS}\n"
      f"="*99,"\n",
      f"lowest AIC is for parameters: {five_lowest_AICS.iloc[0]}"
      )

print(len(data_aic))