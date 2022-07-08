# %%

import os
import pandas as pd
import numpy as np
from pandas.tseries.offsets import Second
import sklearn
from summarytools.summarytools import dfSummary
from matplotlib import pyplot as plt

# %% CONFIG
MEMBERS = 'Members'
PRICES = 'Prices'
REFUELING = 'Refueling'

# %%
mem = pd.read_csv(r"C:\Users\fabia\Documents\Uni\TUM\QTEM\QDC\Anagrafica_ClubQ8.csv", sep=";", decimal = ',')
pri = pd.read_csv(r"C:\Users\fabia\Documents\Uni\TUM\QTEM\QDC\Premi_ClubQ8.csv", sep=";", decimal = ',')
refu = pd.read_csv(r"C:\Users\fabia\Documents\Uni\TUM\QTEM\QDC\Rifornimenti_Carburante_ClubQ8.csv", sep=";")



# %% data exploration
if True:
    mem.head()
    pri.head()
    refu.head()

    print(mem.head(10))
    print(pri.head(10))
    print(refu.head(10))


    print("Members count rows: " + str(len(mem)))
    print("Prices count rows: " + str(len(pri)))
    print("refueling count rows: " + str(len(refu)))

    # columns
    print(mem.columns)
    print(pri.columns)
    print(refu.columns)


    print(mem.dtypes)
    print(pri.dtypes)
    print(refu.dtypes)

    # na rows
    print(f"{MEMBERS} has {mem.isna().sum().sum()} rows with na values which corresponds to {round(mem.isna().sum().sum()/len(mem) * 100, 2)} %")
    print(f"{PRICES} has {pri.isna().sum().sum()} rows with na values which corresponds to {round(pri.isna().sum().sum()/len(pri) * 100, 2)} %")
    print(f"{REFUELING} has {refu.isna().sum().sum()} rows with na values which corresponds to {round(refu.isna().sum().sum()/len(refu) * 100, 2)} %")

    # na rows per column
    print(f"Length of {MEMBERS} = {len(mem)}. Columns na values:")
    print(mem.isna().sum())
    print(f"Length of {PRICES} = {len(pri)}. Columns na values:")
    print(pri.isna().sum())
    print(f"Length of {REFUELING} = {len(refu)}. Columns na values:")
    print(refu.isna().sum())

    percent_missing_mem = mem.isnull().sum() * 100 / len(mem)
    missing_value_mem = pd.DataFrame({'count': mem.isna().sum(),
                                    'percent_missing': percent_missing_mem})
    percent_missing_pri = pri.isnull().sum() * 100 / len(pri)
    missing_value_pri = pd.DataFrame({'count': pri.isna().sum(),
                                    'percent_missing': percent_missing_pri})
    percent_missing_refu = refu.isnull().sum() * 100 / len(refu)
    missing_value_refu = pd.DataFrame({'count': refu.isna().sum(),
                                    'percent_missing': percent_missing_refu})


    # check whether NA values in the location columns are all in the same rows -> yes they are
    mem[mem['REGIONE'].isna()]
    a = np.where(mem['REGIONE'].isna() & mem['PROVINCIA'].isna() & mem['COMUNE'].isna(), True, False)
    np.unique(a, return_counts=True)

    # check whether NA values in mem SALDO_PUNTI and pri ‘CONTRIBUTO_CLIENTE_CON_IVA’ relate to the column being 0
    mem[mem['SALDO_PUNTI']==0]
    pri[pri['CONTRIBUTO_CLIENTE_CON_IVA']==0]

    # levels

    mem_level_cols = ['SEX', 'REGIONE', 'PROVINCIA', 'COMUNE', 'TIPO_CARTA']
    pri_level_cols = ['LUOGO_PRENOTAZIONE_PREMIO', 'CATEGORIA', 'RAGGRUPPAMENTO_MERCEOLOGICO']
    refu_level_cols = ['PRODOTTO', 'MODALITA_VENDITA']

    for col in mem_level_cols:
        print(f"""
        {col} with {mem[col].nunique()} values""")
        print(mem[col].value_counts())

    for col in pri_level_cols:
        print(f"""
        {col} with {pri[col].nunique()} values""")
        print(pri[col].value_counts())

    for col in refu_level_cols:
        print(f"""
        {col} with {refu[col].nunique()} values""")
        print(refu[col].value_counts())


    # lowest, highest, standard deviation per numeric column
    dfSummary(mem)

    # some histograms
    n, bins, patches = plt.hist(mem['DATA_BATTESIMO'], density=False, facecolor='g', alpha=0.75)
    plt.show()
    n, bins, patches = plt.hist(refu['DATA_OPERAZIONE'], density=False, facecolor='g', alpha=0.75)
    plt.show()
    n, bins, patches = plt.hist(pri['DATA_OPERAZIONE'], density=False, facecolor='g', alpha=0.75)
    plt.show()
    # find out whether customer contribution for prices are fixed or flexible

# %% cleansing
# convert columns to appropiate datatype
mem = mem[~mem['SALDO_PUNTI'].isna()] # filter out members with NA values for SALDO_PUNTI TODO: does this make sense?
mem['SALDO_PUNTI'] = mem['SALDO_PUNTI'].astype(int) # convert to int
mem['DATA_NASCITA'] = pd.to_datetime(mem['DATA_NASCITA']) # convert to datetime
mem['DATA_BATTESIMO'] = pd.to_datetime(mem['DATA_BATTESIMO']) # convert to datetime
mem_str_cols = ['SEX', 'REGIONE', 'PROVINCIA', 'COMUNE', 'TIPO_CARTA']
for col in mem_str_cols:
    mem[col] = mem[col].astype('string')

refu['DATA_OPERAZIONE'] = pd.to_datetime(refu['DATA_OPERAZIONE'])
refu_str_cols = ['PRODOTTO', 'MODALITA_VENDITA']
for col in refu_str_cols:
    refu[col] = refu[col].astype('string')

pri['DATA_OPERAZIONE'] = pd.to_datetime(pri['DATA_OPERAZIONE'])
pri_str_cols = ['LUOGO_PRENOTAZIONE_PREMIO', 'CATEGORIA', 'RAGGRUPPAMENTO_MERCEOLOGICO', 'DESCRIZIONE']
for col in pri_str_cols:
    pri[col] = pri[col].astype('string')

# %% Data description and visualization
## Members
# members age distribution
n, bins, patches = plt.hist(mem['DATA_NASCITA'], bins=50, density=False, facecolor='b', alpha=0.75)
plt.xlabel('Year of birth')
plt.ylabel('Count')
plt.show()

# members time in program distribution
n, bins, patches = plt.hist(mem['DATA_BATTESIMO'], bins=50, density=False, facecolor='b', alpha=0.75)
plt.xlabel('Year of program entrance')
plt.ylabel('Count')
plt.show()

# members points min, max, mean
min_points = mem['SALDO_PUNTI'].min()
max_points = mem['SALDO_PUNTI'].max()
mean_points = mem['SALDO_PUNTI'].mean()

## Refueling
# most common PRODOTTO
refu_prodotto = refu.groupby('PRODOTTO').size().reset_index().rename({0:'count'}, axis=1).sort_values('count', ascending=False)


# most common MODALITA_VENDITA
refu_modalita = refu.groupby('MODALITA_VENDITA').size().reset_index().rename({0:'count'}, axis=1).sort_values('count', ascending=False)
plt.bar(refu_modalita['MODALITA_VENDITA'], height=refu_modalita['count'], color='blue')
plt.xlabel('Type of Service')
plt.ylabel('Count in Million')
plt.show()

# refueling litres distribution
n, bins, patches = plt.hist(refu['LITRI'], bins=50, density=False, facecolor='g', alpha=0.75)
plt.show()

# refueling points distribution
n, bins, patches = plt.hist(refu['PUNTI_CARBURANTE'], bins=30, density=False, facecolor='b', alpha=0.75)
plt.xlabel('Received points')
plt.ylabel('Count in Million')
plt.show()

# LITRI min, max mean
min_litri = refu['LITRI'].min()
max_litri = refu['LITRI'].max()
mean_litri = refu['LITRI'].mean()

## Prices
# most common LUOGO_PRENOTAZIONE_PREMIO -> bar chart
pri_luogo = pri.groupby('LUOGO_PRENOTAZIONE_PREMIO').size().reset_index().rename({0:'count'}, axis=1).sort_values('count', ascending=False)
plt.bar(pri_luogo['LUOGO_PRENOTAZIONE_PREMIO'], height=pri_luogo['count'], color = 'blue')
plt.xlabel('Place of Price')
plt.ylabel('Count')
plt.show()

# most common CATEGORIA -> bar chart
pri_categoria = pri.groupby('CATEGORIA').size().reset_index().rename({0:'count'}, axis=1).sort_values('count', ascending=False)
plt.bar(pri_categoria['CATEGORIA'], height=pri_categoria['count'])
plt.xlabel('Type of Price')
plt.ylabel('Count')
plt.show()

# most common RAGGRUPPAMENTO_MERCEOLOGICO
pri_raggrupamento = pri.groupby('RAGGRUPPAMENTO_MERCEOLOGICO').size().reset_index().rename({0:'count'}, axis=1).sort_values('count', ascending=False)

# most common DESCRIZIONE
pri_descrizione = pri.groupby('DESCRIZIONE').size().reset_index().rename({0:'count'}, axis=1).sort_values('count', ascending=False).head(15)

# seasonality of populat price
price = 'CONSEGNA PASTA GAROFALO'
pri_seasonality = pri[pri['DESCRIZIONE']=='CONSEGNA PASTA GAROFALO']
# pri_seasonality_res = pd.DataFrame
list_month = []
list_count = []
for i in range(0,12,1):
    index = (i+8)%12+1
    pri_filtered = pri_seasonality[pri_seasonality['DATA_OPERAZIONE'].apply(lambda x: x.month)==index]
    count = len(pri_filtered)
    list_count.append(count)
    list_month.append(index)
    print(f'Prices claimed in month {index}: {count}')
pri_seasonality_res = pd.DataFrame(columns=['Month', 'count'])
pri_seasonality_res['Month'] = [str(int) for int in list_month]
pri_seasonality_res['count'] = list_count

plt.bar(pri_seasonality_res['Month'], height=pri_seasonality_res['count'])
plt.xlabel('Month')
plt.ylabel('Count')
plt.show()

# %%
# write out to csv for ppt
if True:
    path = r'C:\Users\fabia\Documents\Uni\TUM\QTEM\QDC\Tables for ppt'
    name = 'pri_seasonality_res'
    filename = f'{name}.txt'
    path = os.path.join(path, filename)
    write_out = pri_seasonality_res
    write_out.to_csv(path, index=False)


 # %%
# filter out truck cards
mem = mem[~(mem['TIPO_CARTA'] == 'STAR_TRUCK')] 

# cleanse the negative saldo_punti
# print(mem[mem['SALDO_PUNTI'] < 0])
mem.loc[mem['SALDO_PUNTI'] < 0, 'SALDO_PUNTI'] = 0
# print(mem[mem['SALDO_PUNTI'] < 0])

# set columns to zero where na
mem['SALDO_PUNTI'] = np.where(mem['SALDO_PUNTI'].isna(), 0, mem['SALDO_PUNTI'])
pri['CONTRIBUTO_CLIENTE_CON_IVA'] = np.where(pri['CONTRIBUTO_CLIENTE_CON_IVA'].isna(), 0, pri['CONTRIBUTO_CLIENTE_CON_IVA'])

# check price categories RAGGRUPPAMENTO_MERCEOLOGICO -> put the similar ones into one category -> e.g. bambini & per il tuo bambino
# first check top 3 most requested prices per category

dict_repl_cat = {
    'PER IL TUO BAMBINO':'BAMBINI',
    'PER IL  TUO CUCCIOLO':'AMICI A 4 ZAMPE',
    'BELLEZZA & BENESSERE':'PER IL TUO BENESSERE'
    }
for key, val in dict_repl_cat.items():
    pri['RAGGRUPPAMENTO_MERCEOLOGICO'] = pri['RAGGRUPPAMENTO_MERCEOLOGICO'].replace(key, val)



# %% merging the dataframes
# now rows on transaction level
mem_pri = pd.merge(mem, pri, on='COD_PAN_DA_POS')
mem_refu = pd.merge(mem, refu, on='COD_PAN_DA_POS')

# %% feature engineering and aggregation
df = mem.copy()

# convert period of membership to int day number
duration_membership = mem.loc[:,['COD_PAN_DA_POS', 'DATA_BATTESIMO']]
today = pd.to_datetime('today')
duration_membership['duration_membership'] = (today - duration_membership['DATA_BATTESIMO'])\
    .apply(lambda x: x.days)
duration_membership['duration_membership'] = np.where(duration_membership['duration_membership'].isna(),
    np.nan, duration_membership['duration_membership'].astype('Int64'))
duration_membership = duration_membership.drop('DATA_BATTESIMO', axis=1)
df = df.merge(duration_membership, how='left', left_on='COD_PAN_DA_POS', right_on='COD_PAN_DA_POS')



# e.g. aggregate prices to customer level -> e.g. choose most frequent price category, get price with most required points, average number of points
# number of prices requested
nr_pri = mem_pri.loc[:,['COD_PAN_DA_POS', 'DATA_OPERAZIONE']]
nr_pri = nr_pri.groupby('COD_PAN_DA_POS').size().reset_index().rename({0:'total_prices'}, axis=1)
df = df.merge(nr_pri, how='left', left_on='COD_PAN_DA_POS', right_on='COD_PAN_DA_POS')
df['total_prices'] = df['total_prices'].fillna(0).apply(np.int64)

# most frequent RAGGRUPPAMENTO_MERCEOLOGICO per customer
most_frequent_raggrupamento = mem_pri.loc[:,['COD_PAN_DA_POS', 'RAGGRUPPAMENTO_MERCEOLOGICO']]
most_frequent_raggrupamento = most_frequent_raggrupamento.groupby('COD_PAN_DA_POS').agg(lambda x: pd.Series.mode(x)[0]).rename({'RAGGRUPPAMENTO_MERCEOLOGICO':'most_frequent_raggrupamento'}, axis=1)
df = df.merge(most_frequent_raggrupamento, how='left', left_on='COD_PAN_DA_POS', right_on='COD_PAN_DA_POS')

# most frequent CATEGORIA per customer
most_frequent_category = mem_pri.loc[:,['COD_PAN_DA_POS', 'CATEGORIA']]
most_frequent_category = most_frequent_category.groupby('COD_PAN_DA_POS').agg(lambda x: pd.Series.mode(x)[0]).rename({'CATEGORIA':'most_frequent_category'}, axis=1)
df = df.merge(most_frequent_category, how='left', left_on='COD_PAN_DA_POS', right_on='COD_PAN_DA_POS')


# most frequent price per customer
most_frequent_price = mem_pri.loc[:,['COD_PAN_DA_POS', 'DESCRIZIONE']]
most_frequent_price = most_frequent_price.groupby('COD_PAN_DA_POS').agg(lambda x: pd.Series.mode(x)[0]).rename({'DESCRIZIONE':'most_frequent_price'}, axis=1)
df = df.merge(most_frequent_price, how='left', left_on='COD_PAN_DA_POS', right_on='COD_PAN_DA_POS')

# least expensive price in terms of points per customer
points_least_expensive = mem_pri.loc[:,['COD_PAN_DA_POS', 'PUNTI_RICHIESTI']]
points_least_expensive = points_least_expensive.groupby('COD_PAN_DA_POS').agg(lambda x: pd.Series.min(x)).rename({'PUNTI_RICHIESTI':'points_least_expensive'}, axis=1)
df = df.merge(points_least_expensive, how='left', left_on='COD_PAN_DA_POS', right_on='COD_PAN_DA_POS')

# most expensive price in terms of points per customer
points_most_expensive = mem_pri.loc[:,['COD_PAN_DA_POS', 'PUNTI_RICHIESTI']]
points_most_expensive = points_most_expensive.groupby('COD_PAN_DA_POS').agg(lambda x: pd.Series.max(x)).rename({'PUNTI_RICHIESTI':'points_most_expensive'}, axis=1)
df = df.merge(points_most_expensive, how='left', left_on='COD_PAN_DA_POS', right_on='COD_PAN_DA_POS')

# average cost of price in terms of points per customer
points_average = mem_pri.loc[:,['COD_PAN_DA_POS', 'PUNTI_RICHIESTI']]
points_average = points_average.groupby('COD_PAN_DA_POS').mean().rename({'PUNTI_RICHIESTI':'points_average'}, axis=1)
df = df.merge(points_average, how='left', left_on='COD_PAN_DA_POS', right_on='COD_PAN_DA_POS')

# least expensive price in terms of monetary contribution per customer
money_least_expensive = mem_pri.loc[:,['COD_PAN_DA_POS', 'CONTRIBUTO_CLIENTE_CON_IVA']]
money_least_expensive = money_least_expensive.groupby('COD_PAN_DA_POS').agg(lambda x: pd.Series.min(x)).rename({'CONTRIBUTO_CLIENTE_CON_IVA':'money_least_expensive'}, axis=1)
df = df.merge(money_least_expensive, how='left', left_on='COD_PAN_DA_POS', right_on='COD_PAN_DA_POS')

# most expensive price in terms of monetary contribution per customer
money_most_expensive = mem_pri.loc[:,['COD_PAN_DA_POS', 'CONTRIBUTO_CLIENTE_CON_IVA']]
money_most_expensive = money_most_expensive.groupby('COD_PAN_DA_POS').agg(lambda x: pd.Series.max(x)).rename({'CONTRIBUTO_CLIENTE_CON_IVA':'money_most_expensive'}, axis=1)
df = df.merge(money_most_expensive, how='left', left_on='COD_PAN_DA_POS', right_on='COD_PAN_DA_POS')

# average cost of price in terms of monetary contribution per customer
money_average = mem_pri.loc[:,['COD_PAN_DA_POS', 'CONTRIBUTO_CLIENTE_CON_IVA']]
money_average = money_average.groupby('COD_PAN_DA_POS').mean().rename({'CONTRIBUTO_CLIENTE_CON_IVA':'money_average'}, axis=1)
df = df.merge(money_average, how='left', left_on='COD_PAN_DA_POS', right_on='COD_PAN_DA_POS')



# total number of refueling transactions
nr_trans = mem_refu.loc[:,['COD_PAN_DA_POS', 'DATA_OPERAZIONE']]
nr_trans = nr_trans.groupby('COD_PAN_DA_POS').size().reset_index()
nr_trans.rename({0:'total_refuelings'}, axis=1, inplace=True)
df = df.merge(nr_trans, how='left', left_on='COD_PAN_DA_POS', right_on='COD_PAN_DA_POS')
df['total_refuelings'] = df['total_refuelings'].fillna(0).apply(np.int64)


# most frequent MODALITA_VENDITA
most_frequent_modalita_vendita = refu.loc[:,['COD_PAN_DA_POS', 'MODALITA_VENDITA']]
most_frequent_modalita_vendita = most_frequent_modalita_vendita.groupby('COD_PAN_DA_POS').agg(lambda x: pd.Series.mode(x)[0]).rename({'MODALITA_VENDITA':'most_frequent_modalita_vendita'}, axis=1)
df = df.merge(most_frequent_modalita_vendita, how='left', left_on='COD_PAN_DA_POS', right_on='COD_PAN_DA_POS')

# most frequent PRODOTTO
most_frequent_prodotto = refu.loc[:,['COD_PAN_DA_POS', 'PRODOTTO']]
most_frequent_prodotto = most_frequent_prodotto.groupby('COD_PAN_DA_POS').agg(lambda x: pd.Series.mode(x)[0]).rename({'PRODOTTO':'most_frequent_prodotto'}, axis=1)
df = df.merge(most_frequent_prodotto, how='left', left_on='COD_PAN_DA_POS', right_on='COD_PAN_DA_POS')

# most frequent LUOGO_PRENOTAZIONE_PREMIO
most_frequent_luogo_prenotazione = pri.loc[:,['COD_PAN_DA_POS', 'LUOGO_PRENOTAZIONE_PREMIO']]
most_frequent_luogo_prenotazione = most_frequent_luogo_prenotazione.groupby('COD_PAN_DA_POS').agg(lambda x: pd.Series.mode(x)[0]).rename({'LUOGO_PRENOTAZIONE_PREMIO':'most_frequent_luogo_prenotazione'}, axis=1)
df = df.merge(most_frequent_luogo_prenotazione, how='left', left_on='COD_PAN_DA_POS', right_on='COD_PAN_DA_POS')

# activity column -> 1 when at least one refueling in last 3 months, 0 otherwise
active = refu.loc[:,['COD_PAN_DA_POS', 'DATA_OPERAZIONE']]
active = active[(active['DATA_OPERAZIONE']>='2021-06-01') & (active['DATA_OPERAZIONE']<='2021-08-31')]
active = active.groupby('COD_PAN_DA_POS').size().reset_index().rename({0:'Active'}, axis=1)
active['Active'] = np.where(active['Active']>=1, 1, 0)
df = df.merge(active, how='left', left_on='COD_PAN_DA_POS', right_on='COD_PAN_DA_POS')
df['Active'] = df['Active'].fillna(0)


# number of transactions in period of x weeks -> how active customers are in this period
if True:
    total_nr_days = (refu['DATA_OPERAZIONE'].max() - refu['DATA_OPERAZIONE'].min()).days
    interval_len = 21
    interval_offset = 7
    interval_start = 0
    interval_end = interval_start + interval_len
    interval_count = 0
    date_start = pd.to_datetime(refu['DATA_OPERAZIONE'].min()).replace(hour=0, minute=0, second=0)
    # date_start = pd.to_datetime(refu['DATA_OPERAZIONE'].min(), unit='s')

    nr_trans_intval = mem.loc[:,['COD_PAN_DA_POS']].copy()
    while interval_end < total_nr_days:
        # print(f'interval from {date_start} to {date_start+pd.Timedelta(days=interval_len)}')
        col_name = f'nr_trans_intval_{str(date_start.month).zfill(2)}{str(date_start.day).zfill(2)}_{str((date_start+pd.Timedelta(days=interval_len)).month).zfill(2)}{str((date_start+pd.Timedelta(days=interval_len)).day).zfill(2)}'
        # col_name = f'nr_trans_intval_{interval_count+1}'
        # filter on interval
        refu_filtered = refu.loc[:,['COD_PAN_DA_POS', 'DATA_OPERAZIONE']].copy()
        refu_filtered = refu_filtered[(refu_filtered['DATA_OPERAZIONE']>=date_start) & (refu_filtered['DATA_OPERAZIONE']<=date_start+pd.Timedelta(days=7))]
        # aggregate number of transactions
        refu_filtered = refu_filtered.groupby(['COD_PAN_DA_POS']).size().reset_index().rename({0:col_name}, axis=1)
        # add to dataframe
        nr_trans_intval = nr_trans_intval.merge(refu_filtered, how='left', left_on='COD_PAN_DA_POS', right_on='COD_PAN_DA_POS')
        # set to 0 where NaN and convert to int
        nr_trans_intval[col_name] = nr_trans_intval[col_name].fillna(0).apply(np.int64)
        # update variables
        date_start = date_start+pd.Timedelta(days=interval_offset)
        interval_start += interval_offset
        interval_end = min(interval_start + interval_offset, total_nr_days)
        interval_count += 1

    print(f'Interval count: {interval_count}')


# %% features on prices level
# find out most requested price in terms of numbers
df_pri = pri.copy()
df_pri.groupby(['DESCRIZIONE']).agg()


# %%
# write out to csv
if True:
    path = r'C:\Users\fabia\Documents\Uni\TUM\QTEM\QDC'
    today = pd.to_datetime('today')
    filename = f'df_{today.date()}.txt'
    path = os.path.join(path, filename)
    df.to_csv(path, index=False)

# %%
# write out to csv for ppt
if True:
    path = r'C:\Users\fabia\Documents\Uni\TUM\QTEM\QDC\Tables for ppt'
    name = 'pri_head'
    filename = f'{name}.txt'
    path = os.path.join(path, filename)
    write_out = pri.head(15)
    write_out.to_csv(path, index=False)

# %% check whether csv works
test_df = pd.read_csv(path)
