# src/eda.py
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from adjustText import adjust_text
import pandas as pd
from src.config import PATHS

def run_exploratory_analysis(df):
    """Generates insights through data visualization."""
    
    # 1. Feature Correlation Study
    plt.figure(figsize=(12, 8))
    corr_data = df.corr(numeric_only=True)
    sns.heatmap(data=corr_data, annot=True, cmap='coolwarm', fmt=".2f",
                vmin=-1, vmax=1, linewidths=1.5, linecolor='white')
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{PATHS['eda']}/01_correlation_matrix.png")
    plt.show()

    # Observation: Laps and Race Progress are naturally highly correlated. 
    # LapTime shows significant links with tyre degradation and lap deltas.

    # 2. Race Winners Analysis (2023 vs 2024)
    # Identifying the last lap of each race per driver
    last_lap_idx = df.groupby(['Year', 'Race', 'Driver'])['LapNumber'].idxmax()
    df_finals = df.loc[last_lap_idx].copy()
    
    gp_winners = df_finals[df_finals['Position'] == 1]
    win_counts = gp_winners.groupby(['Year', 'Driver']).size().reset_index(name='Wins')
    win_counts = win_counts.sort_values(['Year', 'Wins'], ascending=[True, False])

    plt.figure(figsize=(12, 6))
    sns.barplot(data=win_counts, x='Driver', y='Wins', hue='Year', palette='magma', edgecolor='black')
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('Grand Prix Wins: 2023 vs 2024', fontsize=15, pad=20)
    plt.xlabel('Pilot', fontsize=12)
    plt.ylabel('Number of Wins', fontsize=12)
    plt.legend(title='Season', loc='upper right')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{PATHS['eda']}/02_race_wins.png")
    plt.show()
    # Observation: Verstappen dominated 2023 (19 wins), while 2024 shows a more balanced distribution.

    # 3. Championship Points (Total Season Standing)
    points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    df_finals['Points'] = df_finals['Position'].map(points_map).fillna(0)
    annual_standings = df_finals.groupby(['Year', 'Driver'])['Points'].sum().reset_index()
    annual_standings = annual_standings.sort_values(['Year', 'Points'], ascending=[True, False])

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=annual_standings, x='Year', y='Points', hue='Driver', marker='o')
    plt.title('Total Championship Points per Season', fontsize=15)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('World Championship Points', fontsize=12)
    plt.xticks([2023, 2024])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Drivers")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PATHS['eda']}/03_points_standing.png")
    plt.show()
    # Observation: In this picture, we can see that Vertappen won the F1 championship both in 2023 and 2024. The
    # difference in points is vast in 2023.

    # 4. Tyre Life Analysis by Compound
    avg_tyre_life = df.groupby('Compound')['TyreLife'].mean().reset_index()
    order = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
    avg_tyre_life['Compound'] = pd.Categorical(avg_tyre_life['Compound'], categories=order, ordered=True)
    avg_tyre_life = avg_tyre_life.sort_values('Compound')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=avg_tyre_life, x='Compound', y='TyreLife', palette='magma')
    plt.title('Average Tyre Life by Compound', fontsize=15)
    plt.xlabel('Type of Tyre', fontsize = 12)
    plt.ylabel('Average Laps per Stint', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PATHS['eda']}/04_tyre_life.png")
    plt.show()
    # Observation : Tyres with a longer average lifespan are hard tyres, followed by intermediante tyres (used in
    # raininy conditions) that has more average life is hard tyres, whose follor intermediate (used in raining days), medium tyres,
    # soft tyres and and wet tyres.

    # 5. Fastest Lap Times by GP
    fastest_laps_idx = df.groupby(['Race', 'Year'])['LapTime (s)'].idxmin()
    fastest_laps = df.loc[fastest_laps_idx].copy()

    lap_counts = fastest_laps.sort_values(['Race', 'Year'])

    plt.figure(figsize=(12, 10))
    ax = sns.scatterplot(data= lap_counts, x='LapTime (s)', y='Race', hue= 'Year', style= 'Year', s = 150,
                     palette= 'magma', edgecolor = 'black', zorder = 3)
    
    texts = []
    for i in range(len(lap_counts)):
        row = lap_counts.iloc[i]
        texts.append(plt.text(row['LapTime (s)'], row['Race'], row['Driver'], fontsize=8, fontweight='bold'))
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    
    min_t = lap_counts['LapTime (s)'].min()
    max_t = lap_counts['LapTime (s)'].max()
    plt.xlim(min_t - 5, max_t + 10)

    plt.title('Fastest Lap Time per GP and Year', fontsize=16, pad=20)
    plt.xlabel('Lap Time (s)', fontsize=12)
    plt.ylabel('GP', fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{PATHS['eda']}/05_fastest_laps.png")
    plt.show()
    # Observation : In this picture we can see the fastests lap time of the different pilots per Gran Prix and year. There are
    # a few differences between them, with the fastest lap time being about 69 seconds in the Austrian Gran Prix and
    # the slowest lap time being about 109 seconds in the Belgian Grand Prix.
    
    # 6. Pit Stop Timing Analysis
    df_pits = df[df['PitStop'] == 1]
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=df_pits, x='RaceProgress', hue='Compound', fill=True, common_norm=False, palette='viridis', alpha=0.3)
    plt.title('Race Progress vs. Pit Stop Density', fontsize=16)
    plt.xlabel('Race Progress (%)', fontsize=12)
    plt.xlabel('Pit Stop Density', fontsize=12)
    plt.axvline(0.33, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(0.66, color='gray', linestyle='--', alpha=0.3)
    plt.xlim(0, 1)
    plt.xticks([0, 0.25, 0.5, 0.75, 1], ['Start', '25%', 'Halfway', '75%', 'Finish'])
    plt.grid(True, axis='x', alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{PATHS['eda']}/06_pit_stop_timing.png")
    plt.show()
    # Observation: This picture shows the stage at which pit stops are made to fit the different tyres during the race. 
    # As we can see, the soft tyres are fitted at around the 65 % mark, as these are used for the final sprint to help
    # the cars accelerate. 