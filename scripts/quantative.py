{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "c4e52880",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "021b3a90",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "506065c0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "141035e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_data(file_paths):\n",
    "    dfs = []\n",
    "    for path in file_paths:\n",
    "        df = pd.read_csv(path)\n",
    "        df['Date'] = pd.to_datetime(df['Date'])\n",
    "        ticker = os.path.basename(path).split(\".\")[0].upper()\n",
    "        df = df[['Date', 'Adj Close']].copy()\n",
    "        df.rename(columns={'Adj Close': ticker}, inplace=True)\n",
    "        dfs.append(df)\n",
    "    \n",
    "    # Merge on Date\n",
    "    combined_df = dfs[0]\n",
    "    for other_df in dfs[1:]:\n",
    "        combined_df = pd.merge(combined_df, other_df, on='Date', how='outer')\n",
    "    \n",
    "    combined_df.sort_values('Date', inplace=True)\n",
    "    combined_df.set_index('Date', inplace=True)\n",
    "    return combined_df"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": ".venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
